"""
Rich terminal dashboard — upgraded for frontier-scale multi-teacher monitoring.

Shows:
- Workers status (all 7+ types)
- Current topic and KG coverage
- Multi-teacher stats (active, healthy, contributions)
- Questions generated, answers collected, samples accepted/rejected
- Dataset size and disk usage
- Queue stats (all queues)
- Quality score distribution
- Knowledge graph metrics
- Errors and uptime
"""

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from datetime import datetime
from typing import Dict
import asyncio


class Dashboard:
    """
    Real-time dashboard for frontier-scale multi-teacher distillation.

    Layout:
    ┌────────────────────────────────────┐
    │         HEADER (system + topic)     │
    ├──────────────────┬─────────────────┤
    │  WORKERS STATUS  │  SYSTEM STATS   │
    │                  │                 │
    │  TEACHER HEALTH  │  QUALITY STATS  │
    │                  │                 │
    │  QUEUE DEPTHS    │  KG COVERAGE    │
    ├──────────────────┴─────────────────┤
    │              FOOTER                │
    └────────────────────────────────────┘
    """

    def __init__(self, refresh_rate: float = 1.0):
        self.console = Console()
        self.refresh_rate = refresh_rate
        self.stats = {
            "workers": {},
            "current_topic": "Initializing...",
            "questions_generated": 0,
            "answers_collected": 0,
            "multi_teacher_answers": 0,
            "samples_accepted": 0,
            "samples_rejected": 0,
            "dataset_size_mb": 0.0,
            "disk_free_gb": 0.0,
            "queue_stats": {},
            "errors_count": 0,
            "uptime_seconds": 0,
            "tokens_used": 0,
            "api_latency_ms": 0,
            "quality_scores": {"avg": 0.0, "min": 0.0, "max": 0.0},
            "kg_stats": {"concepts": 0, "coverage": 0.0, "fully_covered": 0, "unexplored": 0},
            "teachers": {"active": 0, "healthy": 0, "total": 0},
        }

    def update_stats(self, stats: Dict):
        """Update dashboard statistics"""
        self.stats.update(stats)

    def generate_layout(self) -> Layout:
        """Generate the full dashboard layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        # Left side: workers + teachers
        layout["left"].split_column(
            Layout(name="workers_panel"),
            Layout(name="teachers_panel"),
            Layout(name="queues_panel"),
        )

        # Right side: stats + quality + KG
        layout["right"].split_column(
            Layout(name="stats_panel"),
            Layout(name="quality_panel"),
            Layout(name="kg_panel"),
        )

        layout["header"].update(self._create_header())
        layout["workers_panel"].update(self._create_workers_panel())
        layout["teachers_panel"].update(self._create_teachers_panel())
        layout["queues_panel"].update(self._create_queues_panel())
        layout["stats_panel"].update(self._create_stats_panel())
        layout["quality_panel"].update(self._create_quality_panel())
        layout["kg_panel"].update(self._create_kg_panel())
        layout["footer"].update(self._create_footer())

        return layout

    def _create_header(self) -> Panel:
        """Create header panel"""
        uptime = self._format_uptime(self.stats["uptime_seconds"])
        mt_answers = self.stats.get("multi_teacher_answers", 0)

        text = Text()
        text.append("🤖 Frontier-Scale Multi-Teacher Knowledge Distillation", style="bold cyan")
        text.append(f"  |  Uptime: {uptime}", style="yellow")
        text.append(f"  |  MT Answers: {mt_answers}", style="green")
        text.append(f"  |  Topic: {self.stats['current_topic'][:40]}", style="green")

        return Panel(text, style="bold")

    def _create_workers_panel(self) -> Panel:
        """Create workers status panel"""
        table = Table(title="👷 Workers", show_header=True, header_style="bold")
        table.add_column("Worker", style="cyan", width=22)
        table.add_column("Status", width=10)
        table.add_column("Tasks", justify="right", width=8)
        table.add_column("Errs", justify="right", style="red", width=6)

        for name, ws in self.stats.get("workers", {}).items():
            status = "🟢" if ws.get("status") == "running" else "🔴"
            tasks = str(ws.get("tasks_completed", 0))
            errors = str(ws.get("errors", 0))
            table.add_row(name, status, tasks, errors)

        return Panel(table, border_style="blue")

    def _create_teachers_panel(self) -> Panel:
        """Create teacher health panel"""
        teachers = self.stats.get("teachers", {})
        table = Table(title="🎓 Teachers", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")

        table.add_row("Active", str(teachers.get("active", 0)))
        table.add_row("Healthy", str(teachers.get("healthy", 0)))
        table.add_row("Total Registered", str(teachers.get("total", 0)))

        status_color = "green" if teachers.get("active", 0) > 0 else "red"
        status = "✅ All OK" if teachers.get("active", 0) > 0 else "❌ No teachers!"

        return Panel(f"[{status_color}]{status}[/{status_color}]", title="🎓 Teacher Status", border_style="blue")

    def _create_queues_panel(self) -> Panel:
        """Create queue depth panel"""
        qs = self.stats.get("queue_stats", {})
        table = Table(title="📬 Queues", show_header=True, header_style="bold")
        table.add_column("Queue", style="cyan")
        table.add_column("Depth", justify="right", style="yellow")

        queue_names = [
            ("Topics", "topic_queue"),
            ("Questions", "question_queue"),
            ("Answers", "answer_queue"),
            ("Quality", "quality_queue"),
            ("Verification", "verification_queue"),
            ("Processing", "processing"),
            ("Completed", "completed"),
            ("Failed", "failed"),
        ]

        for label, key in queue_names:
            table.add_row(label, str(qs.get(key, 0)))

        return Panel(table, border_style="blue")

    def _create_stats_panel(self) -> Panel:
        """Create system statistics panel"""
        table = Table(title="📊 System Stats", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")

        rows = [
            ("Questions Generated", f"{self.stats['questions_generated']:,}"),
            ("Answers Collected", f"{self.stats['answers_collected']:,}"),
            ("Multi-Teacher Answers", f"{self.stats.get('multi_teacher_answers', 0):,}"),
            ("Samples Accepted", f"{self.stats.get('samples_accepted', 0):,}"),
            ("Samples Rejected", f"{self.stats.get('samples_rejected', 0):,}"),
            ("", ""),
            ("Dataset Size", f"{self.stats['dataset_size_mb']:.1f} MB"),
            ("Disk Free", f"{self.stats['disk_free_gb']:.1f} GB"),
            ("", ""),
            ("Total Errors", str(self.stats['errors_count'])),
        ]

        for metric, value in rows:
            if metric:
                table.add_row(metric, value)
            else:
                table.add_row("─" * 18, "─" * 12)

        return Panel(table, border_style="green")

    def _create_quality_panel(self) -> Panel:
        """Create quality score panel"""
        qs = self.stats.get("quality_scores", {})
        total = self.stats.get("samples_accepted", 0) + self.stats.get("samples_rejected", 0)
        accept_rate = (
            round(self.stats.get("samples_accepted", 0) / total * 100, 1)
            if total > 0 else 0.0
        )

        table = Table(title="✅ Quality", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")

        table.add_row("Avg Score", f"{qs.get('avg', 0):.3f}")
        table.add_row("Min Score", f"{qs.get('min', 0):.3f}")
        table.add_row("Max Score", f"{qs.get('max', 0):.3f}")
        table.add_row("Acceptance Rate", f"{accept_rate}%")

        return Panel(table, border_style="green")

    def _create_kg_panel(self) -> Panel:
        """Create knowledge graph panel"""
        kg = self.stats.get("kg_stats", {})

        concepts = kg.get("concepts", 0)
        coverage = kg.get("coverage", 0.0)
        fully = kg.get("fully_covered", 0)
        unexplored = kg.get("unexplored", 0)

        # Coverage bar
        bar_len = 20
        filled = int(coverage * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        text = f"""
Concepts: {concepts}  |  Coverage: {coverage:.1%}
[{bar}]
Fully Covered: {fully}  |  Unexplored: {unexplored}
"""
        return Panel(text.strip(), title="🧠 Knowledge Graph", border_style="green")

    def _create_footer(self) -> Panel:
        """Create footer panel"""
        text = Text()
        text.append("Ctrl+C to stop", style="bold red")
        text.append("  |  ", style="white")
        text.append(f"Updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC", style="dim")
        text.append("  |  ", style="white")
        text.append("Frontier-Scale", style="bold cyan")

        return Panel(text, style="dim")

    def _format_uptime(self, seconds: int) -> str:
        """Format uptime as hh:mm:ss"""
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}h {m:02d}m {s:02d}s"

    async def start(self):
        """Start live dashboard"""
        with Live(
            self.generate_layout(),
            refresh_per_second=1 / self.refresh_rate,
            console=self.console,
        ) as live:
            while True:
                live.update(self.generate_layout())
                await asyncio.sleep(self.refresh_rate)
