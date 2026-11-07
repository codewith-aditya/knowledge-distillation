"""
Multi-Teacher Orchestrator — parallel querying of all active teachers.

Manages:
- One LLMClient per teacher endpoint
- Parallel queries with asyncio.Semaphore for concurrency control
- Partial failure tolerance (if 1 of N fails, still process the rest)
- Per-teacher rate limiting
- Response time tracking fed back to TeacherRegistry health
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from distill.llm.client import LLMClient
from distill.core.teacher_registry import TeacherRegistry, TeacherConfig


@dataclass
class TeacherResponse:
    """Response from a single teacher"""
    teacher_key: str
    teacher_name: str
    question: str
    content: str                   # Raw response text
    reasoning: str                 # Extracted reasoning portion
    answer: str                    # Extracted answer portion
    tokens: int
    model: str
    response_time_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class MultiTeacherResult:
    """Aggregated result from all teachers"""
    question: str
    topic: str
    difficulty: str
    responses: dict[str, TeacherResponse]   # key → response
    successful_count: int
    failed_count: int
    total_time_ms: float
    all_succeeded: bool

    @property
    def any_succeeded(self) -> bool:
        return self.successful_count > 0


class MultiTeacherOrchestrator:
    """
    Orchestrates parallel queries to all active teachers.

    Usage:
        orchestrator = MultiTeacherOrchestrator(registry, concurrency=6)
        result = await orchestrator.query_all(
            question="What is a B+ tree?",
            topic="Database Internals",
            difficulty="intermediate"
        )
    """

    def __init__(
        self,
        registry: TeacherRegistry,
        concurrency: int = 6,
        query_timeout: float = 300.0,
    ):
        self.registry = registry
        self.concurrency = concurrency
        self.query_timeout = query_timeout
        self._semaphore = asyncio.Semaphore(concurrency)

        # One LLMClient per teacher, lazily created
        self._clients: dict[str, LLMClient] = {}
        # Per-teacher rate limiters (requests per second)
        self._rate_limiters: dict[str, asyncio.Semaphore] = {}

    def _get_or_create_client(self, key: str, cfg: TeacherConfig) -> LLMClient:
        """Get or create an LLMClient for a teacher"""
        if key not in self._clients:
            self._clients[key] = LLMClient(
                base_url=cfg.base_url,
                api_key=cfg.resolve_api_key(),
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=self.query_timeout,
            )
            # Simple rate limiter: 2 concurrent requests per endpoint
            self._rate_limiters[key] = asyncio.Semaphore(2)
        return self._clients[key]

    async def query_all(
        self,
        question: str,
        topic: str,
        difficulty: str,
        context: Optional[str] = None,
        prompt_builder: callable = None,
    ) -> MultiTeacherResult:
        """
        Query all active teachers with the same question.

        Args:
            question: The question to ask
            topic: Knowledge domain
            difficulty: Difficulty level
            context: Optional additional context
            prompt_builder: Optional callable(teacher_key, cfg, question, topic, difficulty, context) → str
                            If None, uses a default comprehensive prompt.

        Returns:
            MultiTeacherResult with all responses
        """
        active = self.registry.get_active()
        if not active:
            raise RuntimeError("No active teachers available")

        start_time = time.monotonic()

        # Build prompts for each teacher (potentially custom per teacher)
        teacher_tasks = []
        for key, cfg in active:
            prompt = question  # default — prompt_builder should wrap this
            if prompt_builder:
                prompt = prompt_builder(key, cfg, question, topic, difficulty, context)

            task = self._query_single(key, cfg, question, topic, prompt)
            teacher_tasks.append(task)

        # Run all in parallel with concurrency limit
        results_list = await asyncio.gather(*teacher_tasks, return_exceptions=True)

        total_time_ms = (time.monotonic() - start_time) * 1000

        # Build result dict
        responses: dict[str, TeacherResponse] = {}
        successful = 0
        failed = 0

        for i, (key, cfg) in enumerate(active):
            result = results_list[i]
            if isinstance(result, Exception):
                failed += 1
                responses[key] = TeacherResponse(
                    teacher_key=key,
                    teacher_name=cfg.name,
                    question=question,
                    content="",
                    reasoning="",
                    answer="",
                    tokens=0,
                    model=cfg.model,
                    response_time_ms=0,
                    success=False,
                    error=str(result),
                )
            elif result is not None:
                if result.success:
                    successful += 1
                else:
                    failed += 1
                responses[key] = result
            else:
                failed += 1
                responses[key] = TeacherResponse(
                    teacher_key=key,
                    teacher_name=cfg.name,
                    question=question,
                    content="",
                    reasoning="",
                    answer="",
                    tokens=0,
                    model=cfg.model,
                    response_time_ms=0,
                    success=False,
                    error="No response",
                )

        return MultiTeacherResult(
            question=question,
            topic=topic,
            difficulty=difficulty,
            responses=responses,
            successful_count=successful,
            failed_count=failed,
            total_time_ms=total_time_ms,
            all_succeeded=(failed == 0),
        )

    async def _query_single(
        self,
        key: str,
        cfg: TeacherConfig,
        question: str,
        topic: str,
        prompt: str,
    ) -> Optional[TeacherResponse]:
        """Query a single teacher with rate limiting and error handling"""
        async with self._semaphore:
            rate_limiter = self._rate_limiters.get(key)
            if rate_limiter:
                async with rate_limiter:
                    return await self._do_query(key, cfg, question, topic, prompt)
            return await self._do_query(key, cfg, question, topic, prompt)

    async def _do_query(
        self,
        key: str,
        cfg: TeacherConfig,
        question: str,
        topic: str,
        prompt: str,
    ) -> Optional[TeacherResponse]:
        """Execute a single teacher query with timing and health tracking"""
        client = self._get_or_create_client(key, cfg)

        start = time.monotonic()
        try:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            self.registry.record_success(key, elapsed_ms)

            content = (response.get("content") or "").strip()
            channel_reasoning = (response.get("reasoning") or "").strip()

            if not content and not channel_reasoning:
                raise RuntimeError(
                    f"Empty response from {cfg.model} (finish_reason="
                    f"{response.get('finish_reason')}) — likely ran out of tokens mid-reasoning"
                )

            if channel_reasoning:
                # Endpoint exposes a real thinking channel — keep the two
                # artifacts separate instead of guessing a split.
                reasoning = channel_reasoning
                answer = content or channel_reasoning
            else:
                reasoning, answer = self._extract_reasoning_answer(content)

            return TeacherResponse(
                teacher_key=key,
                teacher_name=cfg.name,
                question=question,
                content=content,
                reasoning=reasoning,
                answer=answer,
                tokens=response["usage"].total_tokens,
                model=response.get("model", cfg.model),
                response_time_ms=elapsed_ms,
                success=True,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            self.registry.record_failure(key)
            # Log the failure — error strings were previously swallowed into
            # TeacherResponse.error and never surfaced, making teacher
            # outages invisible in the logs.
            import logging
            logging.getLogger("distill.orchestrator").warning(
                "Teacher query FAILED: %s: %s (%s @ %s)",
                type(e).__name__, str(e)[:200], key, cfg.base_url,
            )
            return TeacherResponse(
                teacher_key=key,
                teacher_name=cfg.name,
                question=question,
                content="",
                reasoning="",
                answer="",
                tokens=0,
                model=cfg.model,
                response_time_ms=elapsed_ms,
                success=False,
                error=str(e),
            )

    def _extract_reasoning_answer(self, content: str) -> tuple[str, str]:
        """
        Extract reasoning and answer portions from a teacher response.

        Uses the ## header convention from teacher_client.py:
        - Reasoning: everything before the first ## header
        - Answer: everything from the first ## header onward
        """
        if "## " in content:
            idx = content.index("## ")
            reasoning = content[:idx].strip()
            answer = content[idx:].strip()
            return reasoning, answer

        # Fallback: first third is reasoning, rest is answer
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 3:
            split = max(1, len(paragraphs) // 3)
            reasoning = '\n\n'.join(paragraphs[:split])
            answer = '\n\n'.join(paragraphs[split:])
            return reasoning, answer

        return "", content

    async def close(self):
        """Close all LLM clients"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    def get_stats(self) -> dict:
        """Get orchestrator statistics"""
        return {
            "active_teachers": self.registry.get_active_count(),
            "total_queries": sum(c.request_count for c in self._clients.values()),
            "total_tokens": sum(c.total_usage.total_tokens for c in self._clients.values()),
        }
