"""
Quality Worker - Scores Q&A pairs and retries low-quality answers
"""

import asyncio
from distill.config import Config
from distill.llm.client import LLMClient
from distill.quality.scorer import QualityScorer
from distill.orchestration.queue_manager import QueueManager
from distill.monitoring.logger import get_logger


class QualityWorker:
    """
    Worker that scores Q&A pairs for quality

    Flow:
    1. Dequeue Q&A pair from quality_queue
    2. Score using QualityScorer
    3. If quality < threshold: re-enqueue to answer_queue for retry
    4. If quality >= threshold: accept and continue
    5. Log quality metrics
    """

    def __init__(self, worker_id: int, config: Config):
        self.worker_id = worker_id
        self.config = config
        self.running = False

        self.llm_client = None
        self.scorer = None
        self.queue_manager = None
        self.logger = get_logger(config.logs_base_path, config.log_level)

        self.stats = {
            "pairs_scored": 0,
            "pairs_accepted": 0,
            "pairs_rejected": 0,
            "errors": 0
        }

    async def start(self):
        """Start quality worker"""
        self.llm_client = LLMClient(
            base_url=self.config.research_base_url,
            api_key=self.config.research_api_key,
            model=self.config.research_model,
            max_tokens=2048,
            temperature=0.3
        )

        self.scorer = QualityScorer(self.llm_client, self.config)
        self.queue_manager = QueueManager(self.config.redis_url)
        await self.queue_manager.connect()

        self.running = True
        self.logger.info(f"Quality Worker {self.worker_id} started")

        while self.running:
            try:
                task = await self.queue_manager.dequeue_quality(timeout=5)

                if task:
                    await self._process_task(task)

            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error(f"Quality worker error", error=e, worker_id=self.worker_id)
                await asyncio.sleep(1)

    async def _process_task(self, task: dict):
        """Process quality scoring task"""
        try:
            qa_pair = task["qa_pair"]

            question = qa_pair["question"]
            answer = qa_pair["answer"]
            reasoning = qa_pair.get("reasoning", "")

            score_result = await self.scorer.score_qa_pair(question, answer, reasoning)

            self.stats["pairs_scored"] += 1

            if score_result["passed"]:
                self.stats["pairs_accepted"] += 1
                self.logger.info(
                    f"Q&A pair accepted",
                    score=score_result["overall_score"],
                    topic=qa_pair.get("topic", "")
                )
                await self.queue_manager.mark_completed(task["task_id"])

            else:
                self.stats["pairs_rejected"] += 1
                self.logger.warning(
                    f"Q&A pair rejected - low quality",
                    score=score_result["overall_score"],
                    reasons=", ".join(score_result["reasons"]),
                    topic=qa_pair.get("topic", "")
                )

                retry_count = qa_pair.get("retry_count", 0)

                if retry_count < self.config.quality_retry_attempts:
                    self.logger.info(f"Re-enqueueing for retry", retry=retry_count + 1)
                    await self.queue_manager.enqueue_answer(
                        question=question,
                        topic=qa_pair.get("topic", ""),
                        difficulty=qa_pair.get("difficulty", "intermediate"),
                        question_id=task["task_id"]
                    )
                else:
                    self.logger.warning(f"Max retries reached - discarding low-quality answer")

                await self.queue_manager.mark_completed(task["task_id"])

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Quality processing error", error=e, task_id=task.get("task_id"))
            await self.queue_manager.mark_failed(task["task_id"], str(e))

    async def stop(self):
        """Stop worker"""
        self.running = False
        if self.llm_client:
            await self.llm_client.close()
        if self.queue_manager:
            await self.queue_manager.close()
        self.logger.info(f"Quality Worker {self.worker_id} stopped")

    def get_stats(self) -> dict:
        """Get worker statistics"""
        acceptance_rate = round(
            self.stats["pairs_accepted"] / self.stats["pairs_scored"] * 100, 2
        ) if self.stats["pairs_scored"] > 0 else 0.0

        return {
            "worker_id": self.worker_id,
            "status": "running" if self.running else "stopped",
            "acceptance_rate": acceptance_rate,
            **self.stats
        }
