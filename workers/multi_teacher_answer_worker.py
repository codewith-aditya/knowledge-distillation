"""
Multi-Teacher Answer Worker — queries teachers, merges, hands off for scoring.

IMPORTANT: this worker no longer writes to the dataset. It builds the sample
payload and enqueues it for quality verification; only the quality worker
persists a sample, and only after it passes. Previously every answer was written
to disk before scoring, so a 707MB dataset contained only rejected samples.

Flow:
1. Dequeue question from answer_queue
2. Query all active teachers in parallel (via MultiTeacherOrchestrator)
3. Merge — only when 2+ teachers answered. With a single teacher the merger just
   re-emits the same answer and injects meta-commentary into  thinking.
4. Enqueue the payload to quality_queue (no disk write)
5. Re-enqueue the topic to question_queue (continue loop)
"""

import asyncio
from dataclasses import asdict

from distill.config import Config
from distill.llm.client import LLMClient
from distill.core.teacher_registry import TeacherRegistry
from distill.core.multi_teacher_orchestrator import MultiTeacherOrchestrator
from distill.core.teacher_merger import TeacherMerger
from distill.core.teacher_client import build_teacher_prompt
from distill.orchestration.queue_manager import QueueManager
from distill.monitoring.logger import get_logger


class MultiTeacherAnswerWorker:
    """Collects answers from all active teachers, merges, defers storage."""

    def __init__(self, worker_id: int, config: Config):
        self.worker_id = worker_id
        self.config = config
        self.running = False

        self.merger_llm = None
        self.teacher_registry = None
        self.orchestrator = None
        self.merger = None
        self.queue_manager = None
        self.logger = get_logger(config.logs_base_path, config.log_level)

        self.stats = {
            "answers_collected": 0,
            "teachers_queried": 0,
            "mergers_performed": 0,
            "short_answers": 0,
            "errors": 0,
        }

    async def start(self):
        teacher_config_path = getattr(self.config, "teachers_config_path", "config/teachers.yaml")
        self.teacher_registry = TeacherRegistry(teacher_config_path)

        health = await self.teacher_registry.health_check_all(timeout=15.0)
        active_count = self.teacher_registry.get_active_count()
        self.logger.info(
            f"Multi-Teacher Answer Worker {self.worker_id} starting",
            active_teachers=active_count,
            health=health,
        )

        if active_count == 0:
            self.logger.error("No active teachers! Worker cannot proceed.")

        self.merger_llm = LLMClient(
            base_url=self.config.research_base_url,
            api_key=self.config.research_api_key,
            model=self.config.research_model,
            max_tokens=12000,
            temperature=0.35,
            timeout=getattr(self.config, "teacher_query_timeout", 300),
        )

        self.orchestrator = MultiTeacherOrchestrator(
            registry=self.teacher_registry,
            concurrency=getattr(self.config, "teacher_concurrency_limit", 4),
            query_timeout=getattr(self.config, "teacher_query_timeout", 300),
        )

        self.merger = TeacherMerger(
            llm_client=self.merger_llm,
            registry=self.teacher_registry,
            consensus_threshold=getattr(self.config, "merger_consensus_threshold", 2),
            min_teachers_required=getattr(self.config, "min_teachers_required", 1),
        )

        self.queue_manager = QueueManager(self.config.redis_url)
        await self.queue_manager.connect()

        self.running = True
        self.logger.info(f"Multi-Teacher Answer Worker {self.worker_id} started")

        while self.running:
            try:
                task = await self.queue_manager.dequeue_answer(timeout=5)
                if task:
                    await self._process_task(task)
            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error("Multi-teacher answer worker error", error=e, worker_id=self.worker_id)
                await asyncio.sleep(1)

    async def _process_task(self, task: dict):
        try:
            question = task["question"]
            topic = task["topic"]
            difficulty = task["difficulty"]
            question_id = task["question_id"]
            retry_count = task.get("retry_count", 0)

            active = self.teacher_registry.get_active()
            if len(active) < 1:
                self.logger.error("No active teachers — health check triggered")
                await self.teacher_registry.health_check_all(timeout=15.0)
                active = self.teacher_registry.get_active()
                if len(active) < 1:
                    await self.queue_manager.mark_failed(task["task_id"], "No active teachers")
                    return

            result = await self.orchestrator.query_all(
                question=question,
                topic=topic,
                difficulty=difficulty,
                prompt_builder=self._build_teacher_prompt,
            )
            self.stats["teachers_queried"] += len(result.responses)

            if not result.any_succeeded:
                await self.queue_manager.mark_failed(task["task_id"], "All teachers failed")
                return

            successful = [r for r in result.responses.values() if r.success]

            merged = None
            if len(successful) >= 2:
                merged = await self.merger.merge(result)
                if merged:
                    self.stats["mergers_performed"] += 1

            best = max(successful, key=lambda r: len(r.answer), default=None)
            if merged:
                answer_text = merged.primary_answer
                reasoning_text = merged.reasoning
            elif best:
                answer_text = best.answer
                reasoning_text = best.reasoning
            else:
                await self.queue_manager.mark_failed(task["task_id"], "No usable answer")
                return

            # Hard gate: minimum answer length — discard before any queue hop
            if len(answer_text) < self.config.min_answer_length:
                self.stats["short_answers"] += 1
                self.logger.warning(
                    "Answer too short — retrying",
                    length=len(answer_text),
                    min_length=self.config.min_answer_length,
                    topic=topic,
                    retry=retry_count,
                )
                if retry_count < self.config.quality_retry_attempts:
                    await self.queue_manager.enqueue_answer(
                        question=question,
                        topic=topic,
                        difficulty=difficulty,
                        question_id=question_id,
                        retry_count=retry_count + 1,
                    )
                    await self.queue_manager.mark_completed(task["task_id"])
                else:
                    await self.queue_manager.mark_failed(
                        task["task_id"], f"Answer too short ({len(answer_text)} chars)"
                    )
                return

            self.stats["answers_collected"] += 1

            # Payload for the quality worker — it owns persistence.
            qa_pair = {
                "question": question,
                "answer": answer_text,
                "reasoning": reasoning_text,
                "topic": topic,
                "difficulty": difficulty,
                "tokens": merged.total_input_tokens if merged else (best.tokens if best else 0),
                "teachers_contributed": (
                    merged.teachers_contributed if merged else [best.teacher_key]
                ),
                "teachers_failed": (merged.teachers_failed if merged else []),
                "agreement_score": merged.agreement_score if merged else 0.0,
                "contradictions": (
                    [asdict(c) for c in merged.contradictions] if merged else []
                ),
                "is_multi_teacher": merged is not None,
                "model": (
                    "multi-teacher-merged"
                    if merged else (best.teacher_name if best else "unknown")
                ),
                "examples": merged.examples if merged else [],
                "code_blocks": merged.code_blocks if merged else [],
                "production_notes": merged.production_notes if merged else "",
                "best_practices": merged.best_practices if merged else "",
                "failure_cases": merged.failure_cases if merged else "",
                "optimization_notes": merged.optimization_notes if merged else "",
                "references": merged.references if merged else [],
                "retry_count": retry_count,
            }
            await self.queue_manager.enqueue_quality(qa_pair)

            # Continue the loop — next question on this topic
            await self.queue_manager.enqueue_question(
                topic=topic,
                difficulty=difficulty,
                context={"previous_qa": [{"question": question, "answer": answer_text}]},
            )

            self.logger.info(
                "Answer collected — queued for quality gate",
                topic=topic,
                difficulty=difficulty,
                teachers=len(successful),
                agreement=round(merged.agreement_score, 3) if merged else 0,
                chars=len(answer_text),
                retry=retry_count,
            )

            await self.queue_manager.mark_completed(task["task_id"])

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Multi-teacher answer processing error", error=e, task_id=task.get("task_id"))
            await self.queue_manager.mark_failed(task["task_id"], str(e))

    def _build_teacher_prompt(self, key, cfg, question, topic, difficulty, context):
        """One prompt builder for every teacher — wording lives in teacher_client."""
        return build_teacher_prompt(question, topic, difficulty, context)

    async def stop(self):
        self.running = False
        if self.orchestrator:
            await self.orchestrator.close()
        if self.merger_llm:
            await self.merger_llm.close()
        if self.queue_manager:
            await self.queue_manager.close()
        self.logger.info(f"Multi-Teacher Answer Worker {self.worker_id} stopped")

    def get_stats(self) -> dict:
        registry_stats = self.teacher_registry.get_stats() if self.teacher_registry else {}
        return {
            "worker_id": self.worker_id,
            "status": "running" if self.running else "stopped",
            "active_teachers": registry_stats.get("total_active", 0),
            **self.stats,
        }
