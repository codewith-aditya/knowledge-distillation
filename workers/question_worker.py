"""
Question Worker — generates questions using the Research Agent (student model).

Flow:
1. Dequeue question task (topic + difficulty)
2. Enforce the per-topic cap → retire the topic and pull a fresh one
3. Generate question (student model), validate it (readable, ends with '?',
   no invented concepts, not truncated)
4. Cross-worker duplicate check (Redis-backed)
5. Enqueue to answer_queue; the answer worker continues the same topic loop
"""

import asyncio
import re

from distill.config import Config
from distill.llm.client import LLMClient
from distill.core.research_agent import ResearchAgent
from distill.orchestration.queue_manager import QueueManager
from distill.quality.duplicate_detector import DuplicateDetector
from distill.monitoring.logger import get_logger

# Fabricated-concept tells: the old run produced 13k questions asking the teacher
# to "formally derive **<made-up primitive>**". These never belong in a dataset.
FABRICATION_PATTERNS = [
    r"the missing\s+\*{0,2}[\w-]+",
    r"formally derive\s+\*{2}",
    r"does not yet exist",
    r"no standard named",
]


class QuestionWorker:
    """Worker that generates (and validates) questions for topics."""

    def __init__(self, worker_id: int, config: Config):
        self.worker_id = worker_id
        self.config = config
        self.running = False

        self.llm_client = None
        self.research_agent = None
        self.queue_manager = None
        self.duplicate_detector = None
        self.logger = get_logger(config.logs_base_path, config.log_level)

        self.stats = {
            "questions_generated": 0,
            "duplicates_found": 0,
            "invalid_questions": 0,
            "topics_exhausted": 0,
            "topics_rotated": 0,
            "errors": 0,
        }

    async def start(self):
        self.llm_client = LLMClient(
            base_url=self.config.research_base_url,
            api_key=self.config.research_api_key,
            model=self.config.research_model,
            max_tokens=self.config.research_max_tokens,
            temperature=self.config.research_temperature,
        )

        self.research_agent = ResearchAgent(self.llm_client, self.config)
        self.queue_manager = QueueManager(self.config.redis_url)
        await self.queue_manager.connect()

        self.duplicate_detector = DuplicateDetector(self.config)

        self.running = True
        self.logger.info(f"Question Worker {self.worker_id} started")

        while self.running:
            try:
                task = await self.queue_manager.dequeue_question(timeout=5)
                if task:
                    await self._process_task(task)
            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error("Question worker error", error=e, worker_id=self.worker_id)
                await asyncio.sleep(1)

    # ── Topic bookkeeping ────────────────────────────────────

    async def _bump_topic_counter(self, topic: str) -> int:
        """Increment and return how many questions this topic has produced."""
        key = f"distill:topiccount:{DuplicateDetector._topic_slug(topic)}"
        try:
            n = await self.queue_manager.redis.incr(key)
            return int(n)
        except Exception:
            return 1

    async def _retire_topic(self, topic: str, reason: str):
        """Mark a topic done and refill the topic queue with a fresh topic."""
        self.stats["topics_exhausted"] += 1
        redis = self.queue_manager.redis
        try:
            await redis.sadd("distill:topics_retired", topic)
        except Exception:
            pass

        next_topic = await self._pull_next_topic()
        if next_topic:
            await self.queue_manager.enqueue_topic(next_topic, difficulty="beginner", priority=5)
            self.stats["topics_rotated"] += 1
            self.logger.info(
                "Topic retired — queued next topic",
                topic=topic, reason=reason, next_topic=next_topic,
            )
        else:
            # Pool drained: fall back to discovery-style expansion so the
            # pipeline never stalls with an empty topic queue.
            await self.queue_manager.enqueue_topic(
                f"Related to: {topic}", difficulty="beginner", priority=5
            )
            self.logger.info("Topic retired — pool empty, queued related topic", topic=topic, reason=reason)

    async def _pull_next_topic(self):
        """Pop an unused seed/rotating topic from the shared pool."""
        redis = self.queue_manager.redis
        pool_key = "distill:topic_pool"
        try:
            for _ in range(200):
                raw = await redis.rpop(pool_key)
                if raw is None:
                    return None
                topic = raw if isinstance(raw, str) else raw.decode()
                already = await redis.sismember("distill:topics_retired", topic)
                if not already:
                    return topic
        except Exception as e:
            self.logger.warning("topic pool pull failed", error=e)
        return None

    # ── Question validation ──────────────────────────────────

    def _validate_question(self, question: str) -> tuple[bool, str]:
        """Reject truncated, rambling or fabricated questions before they cost
        a teacher call and poison the dataset."""
        q = (question or "").strip()

        if len(q) < 20:
            return False, "too_short"
        if len(q) >= 1990:
            return False, "truncated_at_cap"
        if not q.endswith("?"):
            return False, "no_question_mark"
        if len(q.split()) > 160:
            return False, "too_rambling"
        for pat in FABRICATION_PATTERNS:
            if re.search(pat, q, re.I):
                return False, "fabricated_concept"
        return True, "ok"

    async def _generate_valid_question(self, topic, difficulty, previous_qa, context):
        """Generate up to 3 times until the question passes validation."""
        last_reason = "unknown"
        for attempt in range(3):
            result = await self.research_agent.generate_question(
                topic=topic,
                difficulty=difficulty,
                previous_qa_pairs=previous_qa,
                context=context,
            )
            if not result:
                return None, "generation_failed"
            if result.get("exhausted"):
                return result, "exhausted"

            ok, reason = self._validate_question(result.get("question", ""))
            if ok:
                return result, "ok"

            last_reason = reason
            self.stats["invalid_questions"] += 1
            self.logger.warning(
                "Rejected generated question — regenerating",
                topic=topic, reason=reason, attempt=attempt + 1,
                question=(result.get("question") or "")[:180],
            )
            # Nudge the agent: force a shorter, real-topic question next round.
            previous_qa = list(previous_qa or []) + [{
                "question": result.get("question", ""),
                "answer": f"[rejected: {reason}] ask something real, short and specific",
            }]
        return None, last_reason

    async def _process_task(self, task: dict):
        try:
            topic = task["topic"]
            context = task.get("context", {}) or {}
            previous_qa = context.get("previous_qa", [])

            # 1. Per-topic cap → rotate instead of looping on one topic forever
            asked = await self._bump_topic_counter(topic)
            cap = getattr(self.config, "max_questions_per_topic", 30)
            if asked > cap:
                await self._retire_topic(topic, f"cap reached ({asked - 1}/{cap})")
                await self.queue_manager.mark_completed(task["task_id"])
                return

            # 2. Spread difficulty across levels instead of all-beginner
            levels = self.config.difficulty_levels or ["beginner"]
            difficulty = levels[(asked - 1) % len(levels)]

            # 3. Generate + validate
            result, status = await self._generate_valid_question(
                topic, difficulty, previous_qa, context.get("additional_context")
            )

            if status == "exhausted":
                self.logger.info("Topic exhausted by research agent", topic=topic)
                await self._retire_topic(topic, "agent reported exhausted")
                await self.queue_manager.mark_completed(task["task_id"])
                return

            if not result or not result.get("question"):
                self.logger.warning("Question generation failed", topic=topic, reason=status)
                await self.queue_manager.mark_failed(task["task_id"], f"question_{status}")
                # Keep the topic alive rather than dropping it.
                await self.queue_manager.enqueue_question(topic, difficulty, context)
                return

            question = result["question"]

            # 4. Shared duplicate check (spans workers + restarts)
            dup = await self.duplicate_detector.check_and_register(
                question, topic, self.queue_manager.redis
            )
            if dup["is_duplicate"]:
                self.stats["duplicates_found"] += 1
                self.logger.info(
                    "Duplicate question detected", topic=topic, method=dup["method"],
                    similarity=dup.get("similarity"),
                )
                await self.queue_manager.enqueue_question(topic, difficulty, context)
                await self.queue_manager.mark_completed(task["task_id"])
                return

            # 5. Hand off to the answer queue
            await self.queue_manager.enqueue_answer(
                question=question,
                topic=topic,
                difficulty=difficulty,
                question_id=task["task_id"],
            )

            self.stats["questions_generated"] += 1
            self.logger.info(
                "Question generated", topic=topic, difficulty=difficulty,
                topic_count=asked, q_chars=len(question),
            )
            await self.queue_manager.mark_completed(task["task_id"])

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Question processing error", error=e, task_id=task.get("task_id"))
            await self.queue_manager.mark_failed(task["task_id"], str(e))

    async def stop(self):
        self.running = False
        if self.llm_client:
            await self.llm_client.close()
        if self.queue_manager:
            await self.queue_manager.close()
        self.logger.info(f"Question Worker {self.worker_id} stopped")

    def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "status": "running" if self.running else "stopped",
            **self.stats,
        }
