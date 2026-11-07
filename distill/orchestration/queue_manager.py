"""
Redis queue manager for distributed task processing
"""

import asyncio
import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
# pyrefly: ignokaise re [missing-import]
import redis.asyncio as aioredis


class QueueManager:
    """
    Redis-backed queue system for task distribution

    Queues:
    - topic_queue: Topics to explore
    - question_queue: Questions to generate
    - answer_queue: Questions awaiting answers
    - quality_queue: Answers to score
    """

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None

        self.TOPIC_QUEUE = "distill:queue:topic"
        self.QUESTION_QUEUE = "distill:queue:question"
        self.ANSWER_QUEUE = "distill:queue:answer"
        self.QUALITY_QUEUE = "distill:queue:quality"
        self.MERGE_QUEUE = "distill:queue:merge"              # Phase 1
        self.VERIFICATION_QUEUE = "distill:queue:verification" # Phase 1
        self.KG_UPDATE_QUEUE = "distill:queue:kg_update"      # Phase 4
        self.RETRY_QUEUE = "distill:queue:retry"              # Phase 1

        self.PROCESSING_SET = "distill:processing"
        self.COMPLETED_SET = "distill:completed"
        self.FAILED_SET = "distill:failed"

    async def connect(self):
        """Connect to Redis"""
        # socket_timeout MUST exceed the longest blocking dequeue wait (5s).
        # redis-py 8.x defaults it to 5s, which races BRPOP's 5s server-side
        # hold and produces "Timeout reading from localhost:6379" every cycle.
        self.redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_timeout=30.0,
            socket_connect_timeout=10.0,
        )

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()

    async def enqueue_topic(self, topic: str, difficulty: str = "beginner", priority: int = 0) -> str:
        """Add topic to topic queue"""
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "type": "topic",
            "topic": topic,
            "difficulty": difficulty,
            "created_at": datetime.utcnow().isoformat(),
            "priority": priority
        }

        await self.redis.zadd(self.TOPIC_QUEUE, {json.dumps(task): -priority})
        return task_id

    async def enqueue_question(self, topic: str, difficulty: str, context: Dict) -> str:
        """Add question generation task to question queue"""
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "type": "question",
            "topic": topic,
            "difficulty": difficulty,
            "context": context,
            "created_at": datetime.utcnow().isoformat()
        }

        await self.redis.lpush(self.QUESTION_QUEUE, json.dumps(task))
        return task_id

    async def enqueue_answer(
        self,
        question: str,
        topic: str,
        difficulty: str,
        question_id: str,
        retry_count: int = 0,
    ) -> str:
        """Add answer collection task to answer queue"""
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "type": "answer",
            "question": question,
            "topic": topic,
            "difficulty": difficulty,
            "question_id": question_id,
            "retry_count": retry_count,
            "created_at": datetime.utcnow().isoformat()
        }

        await self.redis.lpush(self.ANSWER_QUEUE, json.dumps(task))
        return task_id

    async def enqueue_quality(self, qa_pair: Dict) -> str:
        """Add quality scoring task to quality queue"""
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "type": "quality",
            "qa_pair": qa_pair,
            "created_at": datetime.utcnow().isoformat()
        }

        await self.redis.lpush(self.QUALITY_QUEUE, json.dumps(task))
        return task_id

    async def dequeue_topic(self, timeout: int = 5) -> Optional[Dict]:
        """Get next topic task (priority queue)"""
        result = await self.redis.bzpopmax(self.TOPIC_QUEUE, timeout=timeout)
        if result:
            _, task_json, _ = result
            task = json.loads(task_json)
            await self.redis.sadd(self.PROCESSING_SET, task["task_id"])
            return task
        return None

    async def dequeue_question(self, timeout: int = 5) -> Optional[Dict]:
        """Get next question task"""
        result = await self.redis.brpop(self.QUESTION_QUEUE, timeout=timeout)
        if result:
            _, task_json = result
            task = json.loads(task_json)
            await self.redis.sadd(self.PROCESSING_SET, task["task_id"])
            return task
        return None

    async def dequeue_answer(self, timeout: int = 5) -> Optional[Dict]:
        """Get next answer task"""
        result = await self.redis.brpop(self.ANSWER_QUEUE, timeout=timeout)
        if result:
            _, task_json = result
            task = json.loads(task_json)
            await self.redis.sadd(self.PROCESSING_SET, task["task_id"])
            return task
        return None

    async def dequeue_quality(self, timeout: int = 5) -> Optional[Dict]:
        """Get next quality scoring task"""
        result = await self.redis.brpop(self.QUALITY_QUEUE, timeout=timeout)
        if result:
            _, task_json = result
            task = json.loads(task_json)
            await self.redis.sadd(self.PROCESSING_SET, task["task_id"])
            return task
        return None

    async def mark_completed(self, task_id: str):
        """Mark task as completed"""
        await self.redis.srem(self.PROCESSING_SET, task_id)
        await self.redis.sadd(self.COMPLETED_SET, task_id)

    async def mark_failed(self, task_id: str, error: str):
        """Mark task as failed"""
        await self.redis.srem(self.PROCESSING_SET, task_id)
        await self.redis.sadd(self.FAILED_SET, task_id)
        await self.redis.set(f"distill:error:{task_id}", error, ex=86400)

    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        return {
            "topic_queue": await self.redis.zcard(self.TOPIC_QUEUE),
            "question_queue": await self.redis.llen(self.QUESTION_QUEUE),
            "answer_queue": await self.redis.llen(self.ANSWER_QUEUE),
            "quality_queue": await self.redis.llen(self.QUALITY_QUEUE),
            "merge_queue": await self.redis.llen(self.MERGE_QUEUE),
            "verification_queue": await self.redis.llen(self.VERIFICATION_QUEUE),
            "kg_update_queue": await self.redis.llen(self.KG_UPDATE_QUEUE),
            "retry_queue": await self.redis.llen(self.RETRY_QUEUE),
            "processing": await self.redis.scard(self.PROCESSING_SET),
            "completed": await self.redis.scard(self.COMPLETED_SET),
            "failed": await self.redis.scard(self.FAILED_SET)
        }

    async def clear_all_queues(self):
        """Clear all queues (for testing/reset)"""
        await self.redis.delete(
            self.TOPIC_QUEUE,
            self.QUESTION_QUEUE,
            self.ANSWER_QUEUE,
            self.QUALITY_QUEUE,
            self.MERGE_QUEUE,
            self.VERIFICATION_QUEUE,
            self.KG_UPDATE_QUEUE,
            self.RETRY_QUEUE,
            self.PROCESSING_SET,
            self.COMPLETED_SET,
            self.FAILED_SET
        )
