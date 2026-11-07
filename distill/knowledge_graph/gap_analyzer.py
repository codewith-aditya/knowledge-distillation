"""
Gap Analyzer — Identifies unexplored areas in the knowledge graph.

Analyzes:
- Concepts with low coverage
- Concepts referenced in answers but never explored as primary topics
- Concepts with no exploration at all (dead zones)
- Subtopics mentioned but not decomposed
- Cross-topic connections that reveal unexplored territory

Feeds results back to topic discovery for prioritization.
"""

from dataclasses import dataclass, field
from typing import Optional

from distill.knowledge_graph.kg_manager import KnowledgeGraph
from distill.knowledge_graph.coverage_tracker import CoverageTracker
from distill.knowledge_graph.topic_hierarchy import TopicHierarchy


@dataclass
class GapReport:
    """Report of coverage gaps"""
    critical_gaps: list[dict] = field(default_factory=list)
    moderate_gaps: list[dict] = field(default_factory=list)
    unexplored_concepts: list[str] = field(default_factory=list)
    suggested_next_topics: list[dict] = field(default_factory=list)
    dead_zones: list[str] = field(default_factory=list)
    cross_references: list[dict] = field(default_factory=list)
    total_gaps: int = 0
    estimated_hours_to_complete: float = 0.0


class GapAnalyzer:
    """
    Systematic gap analysis for the knowledge graph.

    Identifies WHERE and WHAT to explore next for maximum coverage gains.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        coverage_tracker: CoverageTracker,
        hierarchy: Optional[TopicHierarchy] = None,
    ):
        self.kg = kg
        self.tracker = coverage_tracker
        self.hierarchy = hierarchy

    def analyze(self, samples_per_hour: float = 0) -> GapReport:
        """
        Perform comprehensive gap analysis.

        Returns a GapReport with prioritized suggestions.
        """
        report = GapReport()

        # 1. Get coverage-based gaps
        gap_summary = self.tracker.get_gap_summary()
        report.critical_gaps = gap_summary["critical_gaps"]
        report.moderate_gaps = gap_summary["moderate_gaps"]
        report.total_gaps = gap_summary["total_gaps"]

        # 2. Find completely unexplored concepts (dead zones)
        report.dead_zones = self._find_dead_zones()

        # 3. Find cross-references (concepts mentioned but not explored)
        report.cross_references = self._find_cross_references()

        # 4. Generate suggested next topics (prioritized)
        report.suggested_next_topics = self._suggest_next_topics()

        # 5. Estimate completion
        if samples_per_hour > 0:
            est = self.tracker.estimate_completion_time(samples_per_hour)
            report.estimated_hours_to_complete = est.get("estimated_hours", 0)

        return report

    def _find_dead_zones(self) -> list[str]:
        """
        Find concepts that exist in the graph but have ZERO exploration.
        These are the highest-priority gaps.
        """
        dead = []
        for name, node in self.kg._concepts.items():
            if node.num_samples == 0 and not node.is_exhausted:
                dead.append(name)
        return dead

    def _find_cross_references(self) -> list[dict]:
        """
        Find concepts referenced in metadata or related to explored topics
        but never explored as primary topics.
        """
        cross_refs = []

        for name, node in self.kg._concepts.items():
            # Check if this concept is referenced in other concepts' metadata
            referenced_by = []
            for other_name, other_node in self.kg._concepts.items():
                if other_name == name:
                    continue
                # Check if name appears in metadata
                metadata_str = str(other_node.metadata).lower()
                if name.lower() in metadata_str:
                    referenced_by.append(other_name)

            if referenced_by and node.coverage_score < 0.3:
                cross_refs.append({
                    "concept": name,
                    "coverage": node.coverage_score,
                    "referenced_by": referenced_by[:5],
                    "depth": node.depth_reached,
                })

        cross_refs.sort(key=lambda x: len(x["referenced_by"]), reverse=True)
        return cross_refs[:20]

    def _suggest_next_topics(self) -> list[dict]:
        """
        Generate prioritized suggestions for what to explore next.

        Priority factors:
        1. Critical gaps (coverage < 0.2) — highest priority
        2. Dead zones (no exploration) — second highest
        3. Moderate gaps (coverage 0.2-0.5)
        4. Cross-referenced unexplored concepts
        """
        suggestions = self.tracker.get_priority_queue(limit=15)
        return suggestions

    def get_priority_score(self, concept_name: str) -> float:
        """
        Calculate a priority score for exploring a specific concept.

        Higher score = should explore sooner.
        Score = (1 - coverage) * depth_boost * dead_zone_boost * cross_ref_boost
        """
        node = self.kg.get_concept(concept_name)
        if not node:
            return 0.0

        if node.is_exhausted:
            return 0.0

        # Base: inverse coverage
        base = 1.0 - node.coverage_score

        # Dead zone boost
        dead_boost = 2.0 if node.num_samples == 0 else 1.0

        # Hierarchy depth boost (prefer shallower topics first)
        depth_boost = 1.0
        if self.hierarchy:
            d = self.hierarchy.get_depth(concept_name)
            depth_boost = max(0.5, 1.0 - d * 0.1)

        # Cross-reference boost
        cross_ref_boost = 1.0
        cross_refs = self._find_cross_references()
        for cr in cross_refs:
            if cr["concept"] == concept_name:
                cross_ref_boost = 1.0 + len(cr["referenced_by"]) * 0.1
                break

        return round(base * dead_boost * depth_boost * cross_ref_boost, 4)

    def get_stats(self) -> dict:
        """Get gap analyzer statistics"""
        dead = self._find_dead_zones()
        cross = self._find_cross_references()

        return {
            "dead_zones": len(dead),
            "cross_references": len(cross),
            "coverage_stats": self.tracker.get_stats(),
        }
