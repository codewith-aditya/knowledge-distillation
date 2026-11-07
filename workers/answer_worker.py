"""
Answer Worker - Collects maximum quality answers from Teacher
"""

import asyncio
from distill.config import Config
from distill.llm.client import LLMClient
from distill.core.teacher_client import TeacherClient
from distill.orchestration.queue_manager import QueueManager
from distill.storage.raw_txt_storage import RawTxtStorage
from distill.monitoring.logger import get_logger


class AnswerWorker:
    """
    Worker that collects answers from Teacher model

    Flow:
    1. Dequeue question from answer_queue
    2. Use Teacher Client to get maximum quality answer
    3. Write to RAW TXT storage with immediate sync
    4. Enqueue to quality_queue for scoring
    5. Re-enqueue topic to question_queue (continue loop)
    """

    def __init__(self, worker_id: int, config: Config):
        self.worker_id = worker_id
        self.config = config
        self.running = False

        self.llm_client = None
        self.teacher_client = None
        self.queue_manager = None
        self.storage = None
        self.logger = get_logger(config.logs_base_path, config.log_level)

        self.stats = {
            "answers_collected": 0,
            "bytes_written": 0,
            "short_answers": 0,
            "errors": 0
        }

    async def start(self):
        """Start answer worker"""
        self.llm_client = LLMClient(
            base_url=self.config.teacher_base_url,
            api_key=self.config.teacher_api_key,
            model=self.config.teacher_model,
            max_tokens=self.config.teacher_max_tokens,
            temperature=self.config.teacher_temperature
        )

        self.teacher_client = TeacherClient(self.llm_client, self.config)
        self.queue_manager = QueueManager(self.config.redis_url)
        await self.queue_manager.connect()

        self.storage = RawTxtStorage(
            base_path=self.config.dataset_base_path,
            rotation_size_gb=self.config.raw_txt_rotation_size_gb
        )

        self.running = True
        self.logger.info(f"Answer Worker {self.worker_id} started")

        while self.running:
            try:
                task = await self.queue_manager.dequeue_answer(timeout=5)

                if task:
                    await self._process_task(task)

            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error(f"Answer worker error", error=e, worker_id=self.worker_id)
                await asyncio.sleep(1)

    async def _process_task(self, task: dict):
        """Process answer collection task"""
        try:
            question = task["question"]
            topic = task["topic"]
            difficulty = task["difficulty"]
            question_id = task["question_id"]

            result = await self.teacher_client.get_answer(
                question=question,
                topic=topic,
                difficulty=difficulty,
                context=None
            )

            if not result:
                await self.queue_manager.mark_failed(task["task_id"], "Failed to get answer")
                return

            reasoning = result["reasoning"]
            answer = result["answer"]
            tokens = result["tokens"]
            model = result["model"]
            retry_count = task.get("retry_count", 0)

            # Hard gate: minimum answer length — never write thin samples
            if len(answer) < self.config.min_answer_length:
                if retry_count < self.config.quality_retry_attempts:
                    self.stats["short_answers"] += 1
                    self.logger.warning(
                        "Answer too short — requesting regeneration",
                        length=len(answer),
                        min_length=self.config.min_answer_length,
                        retry=retry_count + 1,
                        topic=topic,
                    )
                    await self.queue_manager.enqueue_answer(
                        question=question,
                        topic=topic,
                        difficulty=difficulty,
                        question_id=question_id,
                        retry_count=retry_count + 1,
                    )
                    await self.queue_manager.mark_completed(task["task_id"])
                else:
                    self.stats["short_answers"] += 1
                    self.logger.error(
                        "Answer too short after max retries — discarded",
                        length=len(answer),
                        min_length=self.config.min_answer_length,
                        topic=topic,
                    )
                    await self.queue_manager.mark_failed(
                        task["task_id"], f"Answer too short ({len(answer)} chars)"
                    )
                return

            success = self.storage.write_sample(
                topic=topic,
                difficulty=difficulty,
                question=question,
                reasoning=reasoning,
                answer=answer,
                model=model,
                tokens=tokens
            )

            if not success:
                self.logger.error(f"Failed to write sample to storage", task_id=task["task_id"])
                await self.queue_manager.mark_failed(task["task_id"], "Storage write failed")
                return

            self.stats["answers_collected"] += 1
            self.stats["bytes_written"] += len(answer.encode('utf-8'))

            await self.queue_manager.enqueue_quality({
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
                "topic": topic,
                "difficulty": difficulty,
                "tokens": tokens,
                "retry_count": retry_count
            })

            await self.queue_manager.enqueue_question(
                topic=topic,
                difficulty=difficulty,
                context={"previous_qa": [{"question": question, "answer": answer}]}
            )

            self.logger.info(
                f"Answer collected and saved",
                topic=topic,
                difficulty=difficulty,
                tokens=tokens
            )

            await self.queue_manager.mark_completed(task["task_id"])

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Answer processing error", error=e, task_id=task.get("task_id"))
            await self.queue_manager.mark_failed(task["task_id"], str(e))

    async def stop(self):
        """Stop worker"""
        self.running = False
        if self.llm_client:
            await self.llm_client.close()
        if self.queue_manager:
            await self.queue_manager.close()
        if self.storage:
            self.storage.close()
        self.logger.info(f"Answer Worker {self.worker_id} stopped")

    def get_stats(self) -> dict:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "status": "running" if self.running else "stopped",
            **self.stats
        }
