"""
Topic Worker - Manages topic lifecycle and discovery
"""

import asyncio
from distill.config import Config
from distill.llm.client import LLMClient
from distill.core.topic_discovery import TopicDiscoveryEngine
from distill.orchestration.queue_manager import QueueManager
from distill.monitoring.logger import get_logger


class TopicWorker:
    """
    Worker that manages topics and discovers new ones

    Flow:
    1. Dequeue topic from topic_queue
    2. Initialize topic for exploration
    3. Enqueue first question generation task
    4. Monitor topic progress
    5. When topic exhausted, discover related topics
    """

    def __init__(self, worker_id: int, config: Config):
        self.worker_id = worker_id
        self.config = config
        self.running = False

        self.llm_client = None
        self.discovery_engine = None
        self.queue_manager = None
        self.logger = get_logger(config.logs_base_path, config.log_level)

        self.stats = {
            "topics_processed": 0,
            "topics_discovered": 0,
            "errors": 0
        }

    async def start(self):
        """Start topic worker"""
        self.llm_client = LLMClient(
            base_url=self.config.research_base_url,
            api_key=self.config.research_api_key,
            model=self.config.research_model,
            max_tokens=self.config.research_max_tokens,
            temperature=0.8
        )

        self.discovery_engine = TopicDiscoveryEngine(self.llm_client, self.config)
        self.queue_manager = QueueManager(self.config.redis_url)
        await self.queue_manager.connect()

        self.running = True
        self.logger.info(f"Topic Worker {self.worker_id} started")

        while self.running:
            try:
                task = await self.queue_manager.dequeue_topic(timeout=5)

                if task:
                    await self._process_task(task)

            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error(f"Topic worker error", error=e, worker_id=self.worker_id)
                await asyncio.sleep(1)

    async def _process_task(self, task: dict):
        """Process topic task"""
        try:
            topic = task["topic"]
            # Spread topics across all configured difficulty levels instead of
            # locking everything to beginner — generates a balanced dataset
            # (beginner, intermediate, expert, advanced, research).
            levels = self.config.difficulty_levels or ["beginner"]
            difficulty = levels[self.stats["topics_processed"] % len(levels)]

            self.logger.info(f"Starting topic exploration", topic=topic, difficulty=difficulty)

            await self.queue_manager.enqueue_question(
                topic=topic,
                difficulty=difficulty,
                context={"previous_qa": []}
            )

            self.stats["topics_processed"] += 1
            await self.queue_manager.mark_completed(task["task_id"])

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Topic processing error", error=e, task_id=task.get("task_id"))
            await self.queue_manager.mark_failed(task["task_id"], str(e))

    async def stop(self):
        """Stop worker"""
        self.running = False
        if self.llm_client:
            await self.llm_client.close()
        if self.queue_manager:
            await self.queue_manager.close()
        self.logger.info(f"Topic Worker {self.worker_id} stopped")

    def get_stats(self) -> dict:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "status": "running" if self.running else "stopped",
            **self.stats
        }
