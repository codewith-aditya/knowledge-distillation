"""
Main Coordinator — Frontier-scale orchestration of the entire distillation system.

Manages:
- 8 worker types (topic, question, answer, multi-teacher answer, quality, verification, KG, retry)
- Knowledge Graph initialization and periodic persistence
- Multi-teacher pipeline (registry, orchestrator, merger)
- Periodic checkpointing with full state preservation
- System health monitoring
- Graceful shutdown with state preservation
- Auto-resume from checkpoint
- Live dashboard with multi-teacher and KG stats
"""

import asyncio
import signal
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from distill.config import Config
from distill.core.seed_topics import SEED_TOPICS
from distill.orchestration.queue_manager import QueueManager
from distill.storage.checkpoint_manager import CheckpointManager
from distill.storage.raw_txt_storage import RawTxtStorage
from distill.storage.jsonl_backup_writer import JsonlBackupWriter
from distill.core.teacher_registry import TeacherRegistry
from distill.knowledge_graph.kg_manager import KnowledgeGraph
from distill.knowledge_graph.coverage_tracker import CoverageTracker
from distill.knowledge_graph.gap_analyzer import GapAnalyzer
from distill.monitoring.logger import get_logger
from distill.monitoring.dashboard import Dashboard

from workers.topic_worker import TopicWorker
from workers.question_worker import QuestionWorker
from workers.answer_worker import AnswerWorker
from workers.multi_teacher_answer_worker import MultiTeacherAnswerWorker
from workers.quality_verification_worker import QualityVerificationWorker


class Coordinator:
    """
    Frontier-scale coordinator for autonomous multi-teacher distillation.

    Handles:
    - Teacher registry loading and health checks
    - Knowledge Graph initialization
    - Coverage tracking and gap analysis
    - All worker lifecycle (launch, monitor, stop)
    - Periodic checkpointing and KG persistence
    - Live dashboard updates
    """

    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.start_time: Optional[datetime] = None

        # Core services
        self.queue_manager: Optional[QueueManager] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.storage: Optional[RawTxtStorage] = None
        self.jsonl_writer: Optional[JsonlBackupWriter] = None

        # Multi-teacher
        self.teacher_registry: Optional[TeacherRegistry] = None

        # Knowledge Graph
        self.kg: Optional[KnowledgeGraph] = None
        self.coverage_tracker: Optional[CoverageTracker] = None
        self.gap_analyzer: Optional[GapAnalyzer] = None

        # Dashboard
        self.dashboard: Optional[Dashboard] = None

        # Logging
        self.logger = get_logger(config.logs_base_path, config.log_level)

        # Workers
        self.workers: list[tuple[str, object]] = []
        self.worker_tasks: list[asyncio.Task] = []

        # Stats
        self.stats = {
            "start_time": None,
            "uptime_seconds": 0,
            "current_topic": "Initializing...",
            "questions_generated": 0,
            "answers_collected": 0,
            "multi_teacher_answers": 0,
            "samples_accepted": 0,
            "samples_rejected": 0,
            "errors_count": 0,
            "kg_concepts": 0,
            "kg_coverage": 0.0,
        }

    # ── Start / Stop ─────────────────────────────────────────

    async def start(self, seed_topics: list = None):
        """Start the frontier-scale distillation system"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Frontier-Scale Autonomous AI Knowledge Distillation System")
        self.logger.info("=" * 60)

        # 1. Ensure paths
        self.config.ensure_paths()

        # 2. Initialize Redis queues
        self.queue_manager = QueueManager(self.config.redis_url)
        await self.queue_manager.connect()
        self.logger.info("Redis queue manager connected")

        # 3. Initialize storage
        self.storage = RawTxtStorage(
            base_path=self.config.dataset_base_path,
            rotation_size_gb=self.config.raw_txt_rotation_size_gb,
        )
        self.logger.info(f"RAW TXT storage initialized: {self.storage.get_current_file_path()}")

        # JSONL backup
        jsonl_path = str(Path(self.config.dataset_base_path) / "jsonl")
        self.jsonl_writer = JsonlBackupWriter(base_path=jsonl_path)
        self.logger.info(f"JSONL backup writer initialized: {jsonl_path}")

        # 4. Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_base_path)

        # 5. Load teacher registry
        self.teacher_registry = TeacherRegistry(
            config_path=getattr(self.config, 'teachers_config_path', 'config/teachers.yaml')
        )
        active = self.teacher_registry.get_active_count()
        self.logger.info(f"Teacher registry loaded: {active} active teachers")

        # Health check teachers
        if active > 0:
            health = await self.teacher_registry.health_check_all(timeout=15.0)
            healthy_count = sum(1 for v in health.values() if v)
            self.logger.info(f"Teacher health check: {healthy_count}/{len(health)} healthy", health=health)

        # 6. Initialize Knowledge Graph
        kg_path = str(Path(self.config.checkpoint_base_path) / "knowledge_graph.json")
        self.kg = KnowledgeGraph(persistence_path=kg_path)
        self.coverage_tracker = CoverageTracker(self.kg)
        self.gap_analyzer = GapAnalyzer(self.kg, self.coverage_tracker)
        self.logger.info(f"Knowledge Graph loaded: {len(self.kg)} concepts")

        # 7. Resume or seed
        # Checkpoint resume only restores counters — queue contents are in
        # Redis, not the checkpoint. If the queues are empty (e.g. tasks died
        # with workers), re-seed so the system always has fuel on startup.
        resumed = False
        if self.config.auto_resume_on_start:
            checkpoint = await self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                self.logger.info("Checkpoint found — resuming from previous state")
                await self._resume_from_checkpoint(checkpoint)
                resumed = True

        topic_count = await self.queue_manager.get_queue_stats()
        if topic_count.get("topic_queue", 0) == 0:
            if resumed:
                self.logger.info("Queues empty after resume — re-seeding topics")
            await self._initialize_seed_topics(seed_topics or [])

        # 8. Launch workers
        await self._launch_workers()

        # 9. Start running
        self.running = True
        self.start_time = datetime.utcnow()
        self.stats["start_time"] = self.start_time.isoformat() + 'Z'

        self._setup_signal_handlers()

        self.logger.info("All workers started successfully")
        self.logger.info("System running — Press Ctrl+C to stop")

        # 10. Main loop
        await self._main_loop()

    async def stop(self):
        """Stop the system gracefully"""
        self.logger.info("Stopping all workers...")
        self.running = False

        # Stop workers
        for _, worker in self.workers:
            await worker.stop()

        for task in self.worker_tasks:
            task.cancel()

        # Final checkpoint
        state = await self._get_system_state()
        await self.checkpoint_manager.save_checkpoint(state)

        # Persist KG
        if self.kg:
            self.kg.save()
            self.logger.info("Knowledge Graph persisted")

        # Close storage
        if self.storage:
            self.storage.close()
        if self.jsonl_writer:
            self.jsonl_writer.close()

        # Close queues
        if self.queue_manager:
            await self.queue_manager.close()

        self.logger.info("System stopped successfully")
        self.logger.info("=" * 60)

    # ── Internal ─────────────────────────────────────────────

    async def _initialize_seed_topics(self, seed_topics: list):
        """Initialize queues with seed topics + a rotation pool.

        A handful of topics go straight into the topic queue; the rest sit in
        `distill:topic_pool` and are pulled one at a time by the question worker
        when a topic hits its per-topic cap. Without this the pipeline loops on
        a single topic forever (18k of 20k samples landed on one topic).
        """
        if not seed_topics:
            seed_topics = SEED_TOPICS

        self.logger.info(f"Initializing {len(seed_topics)} seed topics")

        # Fresh pool each boot (queue contents live in Redis, pool is rebuilt)
        pool_key = "distill:topic_pool"
        await self.queue_manager.redis.delete(pool_key)

        bootstrap = seed_topics[:5]
        rotation_pool = seed_topics[5:]

        for topic in bootstrap:
            await self.queue_manager.enqueue_topic(topic, difficulty="beginner", priority=10)
            self.kg.add_concept(topic, is_seed=True)

        for topic in rotation_pool:
            await self.queue_manager.redis.rpush(pool_key, topic)
            self.kg.add_concept(topic, is_seed=True)

        self.kg.save()
        self.logger.info(
            "Seed topics initialized",
            queued=len(bootstrap),
            rotation_pool=len(rotation_pool),
        )

    async def _launch_workers(self):
        """Launch all worker processes"""
        self.logger.info("Launching workers...")
        use_multi_teacher = getattr(self.config, 'enable_multi_teacher', True)

        # Topic workers
        for i in range(self.config.num_topic_workers):
            worker = TopicWorker(i + 1, self.config)
            self.workers.append(("topic", worker))
            self.worker_tasks.append(asyncio.create_task(worker.start()))

        # Question workers
        for i in range(self.config.num_question_workers):
            worker = QuestionWorker(i + 1, self.config)
            self.workers.append(("question", worker))
            self.worker_tasks.append(asyncio.create_task(worker.start()))

        # Answer workers — multi-teacher or legacy
        if use_multi_teacher:
            for i in range(self.config.num_answer_workers):
                worker = MultiTeacherAnswerWorker(i + 1, self.config)
                self.workers.append(("multi_teacher_answer", worker))
                self.worker_tasks.append(asyncio.create_task(worker.start()))
        else:
            for i in range(self.config.num_answer_workers):
                worker = AnswerWorker(i + 1, self.config)
                self.workers.append(("answer", worker))
                self.worker_tasks.append(asyncio.create_task(worker.start()))

        # Quality verification workers
        for i in range(self.config.num_quality_workers):
            worker = QualityVerificationWorker(i + 1, self.config)
            self.workers.append(("quality_verification", worker))
            self.worker_tasks.append(asyncio.create_task(worker.start()))

        await asyncio.sleep(2)
        self.logger.info(f"All {len(self.workers)} workers launched")

    async def _main_loop(self):
        """Main coordinator loop with background tasks"""
        checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        stats_task = asyncio.create_task(self._stats_loop())
        kg_persist_task = asyncio.create_task(self._kg_persist_loop())

        if self.config.enable_rich_dashboard:
            self.dashboard = Dashboard(refresh_rate=self.config.dashboard_refresh_rate)
            dashboard_task = asyncio.create_task(self._dashboard_loop())

        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            for t in [checkpoint_task, stats_task, kg_persist_task]:
                t.cancel()
            if self.config.enable_rich_dashboard:
                dashboard_task.cancel()

    async def _checkpoint_loop(self):
        """Periodic checkpointing"""
        while self.running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval_minutes * 60)
                state = await self._get_system_state()
                success = await self.checkpoint_manager.save_checkpoint(state)
                if success:
                    self.logger.info("Checkpoint saved")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Checkpoint error", error=e)

    async def _kg_persist_loop(self):
        """Periodic knowledge graph persistence"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                self.kg.save()
                self.coverage_tracker.take_snapshot()
                self.logger.info("KG persisted and coverage snapshot taken")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("KG persist error", error=e)

    async def _stats_loop(self):
        """Periodic stats collection"""
        while self.running:
            try:
                await asyncio.sleep(5)

                # Worker stats are cumulative counters — snapshot the SUM across
                # workers instead of accumulating (+=) every cycle, which would
                # double-count and inflate the totals.
                questions = answers = mt_answers = 0
                accepted = rejected = errors = 0
                for worker_type, worker in self.workers:
                    ws = worker.get_stats()

                    if worker_type == "question":
                        questions += ws.get("questions_generated", 0)
                    elif worker_type in ("answer", "multi_teacher_answer"):
                        answers += ws.get("answers_collected", 0)
                        if worker_type == "multi_teacher_answer":
                            mt_answers += ws.get("answers_collected", 0)
                    elif worker_type == "quality_verification":
                        accepted += ws.get("pairs_accepted", 0)
                        rejected += ws.get("pairs_rejected", 0)

                    errors += ws.get("errors", 0)

                self.stats["questions_generated"] = questions
                self.stats["answers_collected"] = answers
                self.stats["multi_teacher_answers"] = mt_answers
                self.stats["samples_accepted"] = accepted
                self.stats["samples_rejected"] = rejected
                self.stats["errors_count"] = errors

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Stats collection error", error=e)

    async def _dashboard_loop(self):
        """Update dashboard with real-time stats"""
        while self.running:
            try:
                if self.start_time:
                    self.stats["uptime_seconds"] = int(
                        (datetime.utcnow() - self.start_time).total_seconds()
                    )

                storage_stats = self.storage.get_stats()
                queue_stats = await self.queue_manager.get_queue_stats()
                kg_stats = self.kg.get_stats() if self.kg else {}
                registry_stats = self.teacher_registry.get_stats() if self.teacher_registry else {}

                total, used, free = shutil.disk_usage(self.config.dataset_base_path)
                disk_free_gb = free / (1024 ** 3)

                # Worker stats
                workers_dict = {}
                for worker_type, worker in self.workers:
                    ws = worker.get_stats()
                    key = f"{worker_type}_{ws['worker_id']}"
                    workers_dict[key] = {
                        "status": ws.get("status", "unknown"),
                        "tasks_completed": (
                            ws.get("questions_generated", 0) +
                            ws.get("answers_collected", 0) +
                            ws.get("pairs_scored", 0) +
                            ws.get("mergers_performed", 0)
                        ),
                        "errors": ws.get("errors", 0),
                    }

                self.stats["kg_concepts"] = kg_stats.get("total_concepts", 0)
                self.stats["kg_coverage"] = kg_stats.get("avg_coverage", 0.0)

                self.dashboard.update_stats({
                    "workers": workers_dict,
                    "current_topic": self.stats["current_topic"],
                    "questions_generated": self.stats["questions_generated"],
                    "answers_collected": self.stats["answers_collected"],
                    "multi_teacher_answers": self.stats["multi_teacher_answers"],
                    "samples_accepted": self.stats["samples_accepted"],
                    "samples_rejected": self.stats["samples_rejected"],
                    "dataset_size_mb": storage_stats["total_mb_written"],
                    "disk_free_gb": disk_free_gb,
                    "queue_stats": queue_stats,
                    "errors_count": self.stats["errors_count"],
                    "uptime_seconds": self.stats["uptime_seconds"],
                    "tokens_used": 0,
                    "api_latency_ms": 0,
                    "quality_scores": {"avg": 0.0, "min": 0.0, "max": 0.0},
                    "kg_stats": {
                        "concepts": kg_stats.get("total_concepts", 0),
                        "coverage": kg_stats.get("avg_coverage", 0.0),
                        "fully_covered": kg_stats.get("fully_covered", 0),
                        "unexplored": kg_stats.get("unexplored", 0),
                    },
                    "teachers": {
                        "active": registry_stats.get("total_active", 0),
                        "healthy": registry_stats.get("total_healthy", 0),
                        "total": registry_stats.get("total_registered", 0),
                    },
                })

                await asyncio.sleep(self.config.dashboard_refresh_rate)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Dashboard error", error=e)

    async def _get_system_state(self) -> dict:
        """Get complete system state for checkpointing"""
        queue_stats = await self.queue_manager.get_queue_stats()
        storage_stats = self.storage.get_stats()
        kg_stats = self.kg.get_stats() if self.kg else {}

        return {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "uptime_seconds": self.stats["uptime_seconds"],
            "questions_generated": self.stats["questions_generated"],
            "answers_collected": self.stats["answers_collected"],
            "samples_accepted": self.stats["samples_accepted"],
            "queue_stats": queue_stats,
            "storage_stats": storage_stats,
            "kg_stats": kg_stats,
        }

    async def _resume_from_checkpoint(self, checkpoint: dict):
        """Resume system from checkpoint"""
        self.logger.info(f"Resuming from checkpoint: {checkpoint.get('checkpoint_time')}")
        self.stats.update({
            "questions_generated": checkpoint.get("questions_generated", 0),
            "answers_collected": checkpoint.get("answers_collected", 0),
            "samples_accepted": checkpoint.get("samples_accepted", 0),
        })

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def handler(sig, frame):
            self.logger.info("Shutdown signal received — stopping gracefully...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
