"""
Coverage Tracker — Per-concept and global coverage metrics.

Coverage = f(questions_asked, depth_reached, quality_scores, time_spent)

Concepts with coverage > 0.95 are "saturated" and not revisited
unless new information emerges from related topics.

Tracks:
- Per-concept coverage percentage
- Depth level achieved per concept
- Exploration priority queue
- Coverage trends over time
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from distill.knowledge_graph.kg_manager import KnowledgeGraph, ConceptNode


@dataclass
class CoverageSnapshot:
    """A point-in-time coverage snapshot"""
    timestamp: str
    total_concepts: int
    avg_coverage: float
    fully_covered: int
    unexplored: int
    frontier_size: int


class CoverageTracker:
    """
    High-level coverage tracking and trend analysis.

    Wraps the KnowledgeGraph to provide:
    - Coverage trends (snapshots over time)
    - Saturation detection
    - Priority queue for what to explore next
    - Coverage gap identification
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.snapshots: list[CoverageSnapshot] = []
        self._max_snapshots = 1000

    def take_snapshot(self):
        """Record current coverage state"""
        stats = self.kg.get_coverage_stats()
        frontier = self.kg.get_exploration_frontier(limit=100)

        snapshot = CoverageSnapshot(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            total_concepts=stats["total_concepts"],
            avg_coverage=stats["avg_coverage"],
            fully_covered=stats["fully_covered"],
            unexplored=stats["unexplored"],
            frontier_size=len(frontier),
        )
        self.snapshots.append(snapshot)

        # Trim old snapshots
        if len(self.snapshots) > self._max_snapshots:
            self.snapshots = self.snapshots[-self._max_snapshots:]

    def get_trend(self, num_snapshots: int = 10) -> dict:
        """Get coverage trend over recent snapshots"""
        recent = self.snapshots[-num_snapshots:]
        if not recent:
            return {"direction": "unknown", "change": 0.0}

        if len(recent) < 2:
            return {"direction": "stable", "change": 0.0}

        first = recent[0].avg_coverage
        last = recent[-1].avg_coverage
        change = last - first

        if change > 0.01:
            direction = "improving"
        elif change < -0.01:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "change": round(change, 4),
            "first_coverage": first,
            "last_coverage": last,
            "snapshots_analyzed": len(recent),
        }

    def is_concept_saturated(self, name: str) -> bool:
        """Check if a concept is fully covered (saturated)"""
        node = self.kg.get_concept(name)
        if not node:
            return False
        return node.is_exhausted or node.coverage_score >= 0.95

    def get_priority_queue(self, limit: int = 10) -> list[dict]:
        """
        Get priority-ranked topics to explore next.

        Priority = (1 - coverage_score) * depth_reached_weight * recency_factor
        Higher priority → explore sooner.
        """
        frontier = self.kg.get_exploration_frontier(limit=limit * 3)

        prioritized = []
        now = datetime.utcnow()

        for name, coverage, depth in frontier:
            node = self.kg.get_concept(name)
            if not node or node.is_exhausted:
                continue

            # Base priority: inverse of coverage
            base_priority = 1.0 - coverage

            # Recency factor: prefer topics not explored recently
            recency_factor = 1.0
            if node.last_explored:
                try:
                    last = datetime.fromisoformat(node.last_explored.replace('Z', '+00:00'))
                    days_since = (now - last.replace(tzinfo=None)).total_seconds() / 86400
                    recency_factor = min(days_since / 7.0, 3.0)  # Cap at 3x after a week
                except (ValueError, TypeError):
                    pass

            # Seed topics get a small boost
            seed_boost = 1.1 if node.is_seed else 1.0

            priority = base_priority * recency_factor * seed_boost

            prioritized.append({
                "name": name,
                "coverage": coverage,
                "depth_reached": depth,
                "priority": round(priority, 4),
                "last_explored": node.last_explored or "never",
                "is_seed": node.is_seed,
            })

        prioritized.sort(key=lambda x: x["priority"], reverse=True)
        return prioritized[:limit]

    def get_gap_summary(self) -> dict:
        """Summarize coverage gaps across the knowledge graph"""
        all_concepts = self.kg._concepts

        if not all_concepts:
            return {"total_gaps": 0, "critical_gaps": [], "moderate_gaps": []}

        critical = []  # coverage < 0.2
        moderate = []  # coverage 0.2-0.5

        for name, node in all_concepts.items():
            if node.is_exhausted:
                continue
            if node.coverage_score < 0.2:
                critical.append({"name": name, "coverage": node.coverage_score, "depth": node.depth_reached})
            elif node.coverage_score < 0.5:
                moderate.append({"name": name, "coverage": node.coverage_score, "depth": node.depth_reached})

        return {
            "total_gaps": len(critical) + len(moderate),
            "critical_gaps": sorted(critical, key=lambda x: x["coverage"])[:20],
            "moderate_gaps": sorted(moderate, key=lambda x: x["coverage"])[:20],
        }

    def estimate_completion_time(self, avg_samples_per_hour: float) -> dict:
        """
        Estimate time to full coverage based on current rate.

        Args:
            avg_samples_per_hour: Average samples generated per hour
        """
        gap = self.get_gap_summary()
        total_gaps = gap["total_gaps"]

        if total_gaps == 0 or avg_samples_per_hour <= 0:
            return {"estimable": False, "reason": "No gaps or no rate data"}

        # Assume ~30 samples needed per gap concept to reach 0.95 coverage
        samples_needed = total_gaps * 30
        hours_needed = samples_needed / avg_samples_per_hour
        days_needed = hours_needed / 24

        return {
            "estimable": True,
            "total_gaps": total_gaps,
            "estimated_samples_needed": int(samples_needed),
            "estimated_hours": round(hours_needed, 1),
            "estimated_days": round(days_needed, 1),
            "estimated_completion": (
                datetime.utcnow().isoformat() + 'Z'
            ),
        }

    def get_stats(self) -> dict:
        """Get comprehensive coverage stats"""
        kg_stats = self.kg.get_coverage_stats()
        trend = self.get_trend()
        gap = self.get_gap_summary()

        return {
            **kg_stats,
            "trend": trend,
            "critical_gaps": len(gap["critical_gaps"]),
            "moderate_gaps": len(gap["moderate_gaps"]),
            "total_snapshots": len(self.snapshots),
        }
