"""
Quality Verification Worker — scores, and is the ONLY writer of the dataset.

Flow:
1. Dequeue Q&A payload from quality queue (nothing has touched disk yet)
2. Score: completeness, correctness, reasoning depth, implementation depth,
   novelty, factual consistency, hallucination probability
3. ACCEPT  → persist to RAW TXT + JSONL, mark completed
4. REJECT  → re-queue the answer for retry (no write), discard after max retries

Nothing rejected ever reaches the dataset file. The old flow wrote first and
scored second, so a 707MB dataset was 100% rejected samples.
"""

import asyncio

from distill.config import Config
from distill.llm.client import LLMClient
from distill.quality.scorer import QualityScorer
from distill.orchestration.queue_manager import QueueManager
from distill.storage.raw_txt_storage import RawTxtStorage
from distill.storage.jsonl_backup_writer import JsonlBackupWriter
from distill.monitoring.logger import get_logger


class QualityVerificationWorker:
    """Scores Q&A samples and persists only the ones that pass."""

    def __init__(self, worker_id: int, config: Config):
        self.worker_id = worker_id
        self.config = config
        self.running = False

        self.llm_client = None
        self.scorer = None
        self.queue_manager = None
        self.storage = None
        self.jsonl = None
        self.logger = get_logger(config.logs_base_path, config.log_level)

        self.stats = {
            "pairs_scored": 0,
            "pairs_accepted": 0,
            "pairs_rejected": 0,
            "retries_triggered": 0,
            "discarded": 0,
            "write_failures": 0,
            "errors": 0,
        }

    async def start(self):
        self.llm_client = LLMClient(
            base_url=self.config.research_base_url,
            api_key=self.config.research_api_key,
            model=self.config.research_model,
            max_tokens=2048,
            temperature=0.3,
        )

        self.scorer = QualityScorer(self.llm_client, self.config)
        self.queue_manager = QueueManager(self.config.redis_url)
        await self.queue_manager.connect()

        self.storage = RawTxtStorage(
            base_path=self.config.dataset_base_path,
            rotation_size_gb=self.config.raw_txt_rotation_size_gb,
        )
        self.jsonl = JsonlBackupWriter(
            base_path=str(self.config.dataset_base_path) + "/jsonl"
        )

        self.running = True
        self.logger.info(
            f"Quality Verification Worker {self.worker_id} started",
            dataset_file=self.storage.get_current_file_path(),
        )

        while self.running:
            try:
                task = await self.queue_manager.dequeue_quality(timeout=5)
                if task:
                    await self._process_task(task)
            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error("Quality verification error", error=e, worker_id=self.worker_id)
                await asyncio.sleep(1)

    # ── Persistence ──────────────────────────────────────────

    def _persist(self, qa_pair: dict, scores: dict) -> bool:
        """Write an accepted sample. Returns True only if it hit disk."""
        common = dict(
            topic=qa_pair.get("topic", ""),
            difficulty=qa_pair.get("difficulty", "intermediate"),
            question=qa_pair["question"],
            reasoning=qa_pair.get("reasoning", ""),
            answer=qa_pair["answer"],
            model=qa_pair.get("model", "unknown"),
            tokens=qa_pair.get("tokens", 0),
        )

        # Always use the rich format so quality dimension scores land in the
        # RAW file's METADATA block. The legacy write_sample() has no quality
        # fields, which left accepted samples with no audit trail.
        ok = self.storage.write_merged_sample(
            **common,
            teachers_contributed=qa_pair.get("teachers_contributed") or [common["model"]],
            teachers_failed=qa_pair.get("teachers_failed", []),
            agreement_score=qa_pair.get("agreement_score", 0.0),
            quality_scores=scores,
            examples=qa_pair.get("examples", []),
            code_blocks=qa_pair.get("code_blocks", []),
            production_notes=qa_pair.get("production_notes", ""),
            best_practices=qa_pair.get("best_practices", ""),
            failure_cases=qa_pair.get("failure_cases", ""),
            optimization_notes=qa_pair.get("optimization_notes", ""),
            references=qa_pair.get("references", []),
            contradictions=qa_pair.get("contradictions", []),
            retry_count=qa_pair.get("retry_count", 0),
            low_quality_flag=False,
        )

        if ok:
            try:
                self.jsonl.write_sample({
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "reasoning": qa_pair.get("reasoning", ""),
                    "topic": qa_pair.get("topic", ""),
                    "difficulty": qa_pair.get("difficulty", ""),
                    "model": qa_pair.get("model", ""),
                    "tokens": qa_pair.get("tokens", 0),
                    "teachers": qa_pair.get("teachers_contributed", []),
                    "agreement_score": qa_pair.get("agreement_score", 0.0),
                    "quality": scores,
                })
            except Exception as e:
                self.logger.warning("JSONL backup write failed", error=e)

        return ok

    # ── Main ─────────────────────────────────────────────────

    async def _process_task(self, task: dict):
        try:
            qa_pair = task["qa_pair"]
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            reasoning = qa_pair.get("reasoning", "")
            topic = qa_pair.get("topic", "")
            is_multi = qa_pair.get("is_multi_teacher", False)
            retry_count = qa_pair.get("retry_count", 0)

            score_result = await self.scorer.score_qa_pair(
                question, answer, reasoning, topic=topic,
                is_multi_teacher=is_multi,
            )
            self.stats["pairs_scored"] += 1

            threshold = self.config.min_quality_score
            if is_multi:
                threshold = max(threshold, self.config.min_quality_score)

            if score_result["passed"]:
                scores = {
                    "overall": score_result["overall_score"],
                    "completeness": score_result["completeness"],
                    "correctness": score_result["correctness"],
                    "reasoning_depth": score_result["reasoning_depth"],
                    "implementation_depth": score_result["implementation_depth"],
                    "novelty": score_result["novelty"],
                    "factual_consistency": score_result["factual_consistency"],
                    "hallucination_probability": score_result["hallucination_probability"],
                }
                written = self._persist(qa_pair, scores)

                if not written:
                    self.stats["write_failures"] += 1
                    self.logger.error("Accepted sample FAILED to persist", topic=topic)
                    await self.queue_manager.mark_failed(task["task_id"], "storage write failed")
                    return

                self.stats["pairs_accepted"] += 1
                self.logger.info(
                    "Q&A pair ACCEPTED and saved",
                    score=score_result["overall_score"],
                    topic=topic,
                    chars=len(answer),
                    difficulty=qa_pair.get("difficulty"),
                    multi_teacher=is_multi,
                    file=self.storage.get_current_file_path(),
                )
                await self.queue_manager.mark_completed(task["task_id"])
                return

            # ── Rejected: never written ──
            self.stats["pairs_rejected"] += 1
            max_retries = self.config.quality_retry_attempts

            self.logger.warning(
                "Q&A pair REJECTED",
                score=score_result["overall_score"],
                reasons=", ".join(score_result["reasons"]) or "below threshold",
                topic=topic,
                retry=retry_count + 1,
            )

            if retry_count < max_retries:
                self.stats["retries_triggered"] += 1
                await self.queue_manager.enqueue_answer(
                    question=question,
                    topic=topic,
                    difficulty=qa_pair.get("difficulty", "intermediate"),
                    question_id=task["task_id"],
                    retry_count=retry_count + 1,
                )
            else:
                self.stats["discarded"] += 1
                self.logger.info(
                    "Max retries reached — sample discarded (not written)",
                    topic=topic,
                    final_score=score_result["overall_score"],
                )

            await self.queue_manager.mark_completed(task["task_id"])

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Quality processing error", error=e, task_id=task.get("task_id"))
            await self.queue_manager.mark_failed(task["task_id"], str(e))

    async def stop(self):
        self.running = False
        if self.llm_client:
            await self.llm_client.close()
        if self.queue_manager:
            await self.queue_manager.close()
        if self.storage:
            self.storage.close()
        if self.jsonl:
            self.jsonl.close()
        self.logger.info(f"Quality Verification Worker {self.worker_id} stopped")

    def get_stats(self) -> dict:
        acceptance_rate = round(
            self.stats["pairs_accepted"] / self.stats["pairs_scored"] * 100, 2
        ) if self.stats["pairs_scored"] > 0 else 0.0

        return {
            "worker_id": self.worker_id,
            "status": "running" if self.running else "stopped",
            "acceptance_rate": acceptance_rate,
            **self.stats,
        }
