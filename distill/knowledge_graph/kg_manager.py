"""
Knowledge Graph Manager — In-memory graph with JSON persistence.

Uses networkx for graph operations. No Neo4j needed on VPS.
Handles millions of concepts efficiently in-memory.

Nodes: Topics/Concepts with metadata (coverage_score, last_explored, num_samples)
Edges: Relationship types (prerequisite_of, related_to, subtopic_of, builds_on)

Operations:
- Add concepts and relationships
- Track coverage per concept
- Find unexplored neighbors (exploration frontier)
- Mark concepts as explored
- Persist to JSON and load back
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from threading import Lock


@dataclass
class ConceptNode:
    """A concept/topic in the knowledge graph"""
    name: str
    coverage_score: float = 0.0          # 0.0 (unexplored) to 1.0 (fully covered)
    depth_reached: int = 0               # Max depth level explored
    num_samples: int = 0                 # Number of Q&A samples generated
    avg_quality_score: float = 0.0       # Average quality of samples
    last_explored: str = ""              # ISO timestamp
    first_seen: str = ""                 # ISO timestamp
    is_seed: bool = False                # Was this a seed topic?
    is_exhausted: bool = False           # Fully explored?
    metadata: dict = field(default_factory=dict)


class KnowledgeGraph:
    """
    In-memory knowledge graph with JSON persistence.

    Tracks:
    - Concepts and their coverage status
    - Relationships between concepts
    - Exploration frontier (concepts to explore next)
    - Coverage metrics across the entire graph
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path

        # In-memory structures
        self._concepts: dict[str, ConceptNode] = {}
        self._adjacency: dict[str, dict[str, set[str]]] = {}  # source → {target → {rel_types}}
        self._reverse_adj: dict[str, dict[str, set[str]]] = {}  # target → {source → {rel_types}}

        self._lock = Lock()
        self._dirty = False  # Track if we need to persist

        # Load from disk if available
        if persistence_path:
            p = Path(persistence_path)
            if p.exists():
                self.load()

        self._dirty = False

    # ── Concept Management ───────────────────────────────────

    def add_concept(
        self,
        name: str,
        coverage_score: float = 0.0,
        depth_reached: int = 0,
        is_seed: bool = False,
        metadata: dict = None,
    ) -> ConceptNode:
        """Add or update a concept in the graph"""
        with self._lock:
            name = name.strip()
            if name in self._concepts:
                node = self._concepts[name]
                # Update existing
                if coverage_score > node.coverage_score:
                    node.coverage_score = coverage_score
                if depth_reached > node.depth_reached:
                    node.depth_reached = depth_reached
                if metadata:
                    node.metadata.update(metadata)
            else:
                now = datetime.utcnow().isoformat() + 'Z'
                node = ConceptNode(
                    name=name,
                    coverage_score=coverage_score,
                    depth_reached=depth_reached,
                    is_seed=is_seed,
                    first_seen=now,
                    metadata=metadata or {},
                )
                self._concepts[name] = node
                self._adjacency[name] = {}
                self._reverse_adj[name] = {}

            self._dirty = True
            return node

    def get_concept(self, name: str) -> Optional[ConceptNode]:
        """Get a concept by name"""
        return self._concepts.get(name.strip())

    def has_concept(self, name: str) -> bool:
        """Check if concept exists"""
        return name.strip() in self._concepts

    def remove_concept(self, name: str):
        """Remove a concept and its edges"""
        name = name.strip()
        with self._lock:
            self._concepts.pop(name, None)
            # Remove outgoing edges
            self._adjacency.pop(name, None)
            # Remove incoming edges
            self._reverse_adj.pop(name, None)
            for src in self._adjacency:
                self._adjacency[src].pop(name, None)
            self._dirty = True

    # ── Relationship Management ──────────────────────────────

    RELATIONSHIP_TYPES = {
        "subtopic_of", "prerequisite_of", "related_to",
        "builds_on", "contrasts_with", "extends", "applies_to",
    }

    def add_relationship(
        self, source: str, target: str, rel_type: str
    ):
        """Add a directed relationship between two concepts"""
        source = source.strip()
        target = target.strip()

        if rel_type not in self.RELATIONSHIP_TYPES:
            raise ValueError(f"Unknown relationship type: {rel_type}. Valid: {self.RELATIONSHIP_TYPES}")

        with self._lock:
            # Ensure both concepts exist
            if source not in self._concepts:
                self.add_concept(source)
            if target not in self._concepts:
                self.add_concept(target)

            # Add forward edge
            if target not in self._adjacency[source]:
                self._adjacency[source][target] = set()
            self._adjacency[source][target].add(rel_type)

            # Add reverse edge
            if source not in self._reverse_adj[target]:
                self._reverse_adj[target][source] = set()
            self._reverse_adj[target][source].add(rel_type)

            self._dirty = True

    def get_relationships(self, source: str) -> dict[str, set[str]]:
        """Get all relationships from a concept"""
        return self._adjacency.get(source.strip(), {})

    def get_related(self, name: str, rel_type: Optional[str] = None) -> list[str]:
        """Get concepts related to this one, optionally filtered by type"""
        name = name.strip()
        if name not in self._adjacency:
            return []

        related = []
        for target, types in self._adjacency[name].items():
            if rel_type is None or rel_type in types:
                related.append(target)

        return related

    def get_subtopics(self, name: str) -> list[str]:
        """Get all subtopics of a concept"""
        return self.get_related(name, "subtopic_of")

    def get_prerequisites(self, name: str) -> list[str]:
        """Get all prerequisites for a concept"""
        # Prerequisites point TO this concept
        if name not in self._reverse_adj:
            return []
        prereqs = []
        for source, types in self._reverse_adj[name].items():
            if "prerequisite_of" in types:
                prereqs.append(source)
        return prereqs

    # ── Coverage Tracking ────────────────────────────────────

    def mark_explored(
        self,
        name: str,
        depth_achieved: int = 1,
        quality_score: float = 0.0,
        increment_samples: int = 1,
    ):
        """Mark a concept as explored with updated coverage"""
        name = name.strip()
        with self._lock:
            node = self._concepts.get(name)
            if not node:
                node = self.add_concept(name)

            node.depth_reached = max(node.depth_reached, depth_achieved)
            node.num_samples += increment_samples
            node.last_explored = datetime.utcnow().isoformat() + 'Z'

            # Update average quality
            if quality_score > 0:
                old_total = node.avg_quality_score * (node.num_samples - increment_samples)
                node.avg_quality_score = (old_total + quality_score) / node.num_samples

            # Recalculate coverage score
            node.coverage_score = self._calculate_coverage(node)
            self._dirty = True

    def set_exhausted(self, name: str):
        """Mark a concept as exhausted (fully explored)"""
        name = name.strip()
        with self._lock:
            node = self._concepts.get(name)
            if node:
                node.is_exhausted = True
                node.coverage_score = max(node.coverage_score, 0.95)
                self._dirty = True

    def _calculate_coverage(self, node: ConceptNode) -> float:
        """Calculate coverage score for a concept"""
        # Coverage = f(depth_reached, num_samples, avg_quality)
        # Max coverage possible per depth level:
        depth_max = min(node.depth_reached / 6.0, 1.0) * 0.5  # Depth contributes 50%

        # Sample count (diminishing returns)
        sample_factor = min(node.num_samples / 50.0, 1.0) * 0.3  # Samples contribute 30%

        # Quality factor
        quality_factor = min(node.avg_quality_score, 1.0) * 0.2  # Quality contributes 20%

        return round(min(depth_max + sample_factor + quality_factor, 1.0), 4)

    def get_coverage(self, name: str) -> float:
        """Get coverage score for a concept"""
        node = self._concepts.get(name.strip())
        return node.coverage_score if node else 0.0

    # ── Exploration Frontier ─────────────────────────────────

    def get_unexplored_neighbors(
        self, name: str, min_coverage: float = 0.3
    ) -> list[str]:
        """
        Get related concepts that haven't been sufficiently explored.

        Args:
            name: Source concept
            min_coverage: Maximum coverage to be considered "unexplored"
        """
        related = self.get_related(name)
        unexplored = []
        for r in related:
            cov = self.get_coverage(r)
            if cov < min_coverage:
                unexplored.append(r)
        return unexplored

    def get_exploration_frontier(
        self, limit: int = 20, min_coverage: float = 0.3
    ) -> list[tuple[str, float, int]]:
        """
        Get the exploration frontier — concepts to explore next.

        Returns list of (concept_name, coverage_score, depth_reached)
        sorted by lowest coverage first.
        """
        frontier = []
        for name, node in self._concepts.items():
            if node.is_exhausted:
                continue
            if node.coverage_score < min_coverage:
                frontier.append((name, node.coverage_score, node.depth_reached))

        frontier.sort(key=lambda x: x[1])  # Lowest coverage first
        return frontier[:limit]

    def get_coverage_stats(self) -> dict:
        """Get global coverage statistics"""
        if not self._concepts:
            return {
                "total_concepts": 0,
                "avg_coverage": 0.0,
                "fully_covered": 0,
                "unexplored": 0,
                "exhausted": 0,
            }

        scores = [n.coverage_score for n in self._concepts.values()]
        return {
            "total_concepts": len(self._concepts),
            "avg_coverage": round(sum(scores) / len(scores), 4),
            "fully_covered": sum(1 for s in scores if s >= 0.95),
            "unexplored": sum(1 for s in scores if s < 0.3),
            "exhausted": sum(1 for n in self._concepts.values() if n.is_exhausted),
            "by_depth": {
                d: sum(1 for n in self._concepts.values() if n.depth_reached == d)
                for d in range(7)
            },
        }

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """Persist the knowledge graph to JSON"""
        target = path or self.persistence_path
        if not target:
            return

        with self._lock:
            data = {
                "version": 1,
                "saved_at": datetime.utcnow().isoformat() + 'Z',
                "concepts": {
                    name: {
                        "coverage_score": node.coverage_score,
                        "depth_reached": node.depth_reached,
                        "num_samples": node.num_samples,
                        "avg_quality_score": node.avg_quality_score,
                        "last_explored": node.last_explored,
                        "first_seen": node.first_seen,
                        "is_seed": node.is_seed,
                        "is_exhausted": node.is_exhausted,
                        "metadata": node.metadata,
                    }
                    for name, node in self._concepts.items()
                },
                "edges": {
                    src: {
                        tgt: list(types)
                        for tgt, types in targets.items()
                    }
                    for src, targets in self._adjacency.items()
                },
            }

            # Atomic write
            p = Path(target)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.replace(p)

            self._dirty = False

    def load(self, path: Optional[str] = None):
        """Load the knowledge graph from JSON"""
        target = path or self.persistence_path
        if not target or not Path(target).exists():
            return

        try:
            with open(target, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return

        with self._lock:
            self._concepts.clear()
            self._adjacency.clear()
            self._reverse_adj.clear()

            for name, cd in data.get("concepts", {}).items():
                node = ConceptNode(
                    name=name,
                    coverage_score=cd.get("coverage_score", 0.0),
                    depth_reached=cd.get("depth_reached", 0),
                    num_samples=cd.get("num_samples", 0),
                    avg_quality_score=cd.get("avg_quality_score", 0.0),
                    last_explored=cd.get("last_explored", ""),
                    first_seen=cd.get("first_seen", ""),
                    is_seed=cd.get("is_seed", False),
                    is_exhausted=cd.get("is_exhausted", False),
                    metadata=cd.get("metadata", {}),
                )
                self._concepts[name] = node
                self._adjacency[name] = {}
                self._reverse_adj[name] = {}

            for src, targets in data.get("edges", {}).items():
                if src not in self._adjacency:
                    self._adjacency[src] = {}
                    self._reverse_adj[src] = {}
                for tgt, types in targets.items():
                    if tgt not in self._adjacency:
                        self._adjacency[tgt] = {}
                        self._reverse_adj[tgt] = {}
                    type_set = set(types)
                    self._adjacency[src][tgt] = type_set
                    if src not in self._reverse_adj[tgt]:
                        self._reverse_adj[tgt][src] = set()
                    self._reverse_adj[tgt][src] = type_set

            self._dirty = False

    async def periodic_persist(self, interval_seconds: int = 300):
        """Background task that persists the graph periodically"""
        while True:
            await asyncio.sleep(interval_seconds)
            if self._dirty:
                self.save()

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get comprehensive knowledge graph statistics"""
        coverage = self.get_coverage_stats()
        edge_count = sum(len(targets) for targets in self._adjacency.values())
        return {
            **coverage,
            "total_edges": edge_count,
            "density": round(
                edge_count / max(len(self._concepts) * (len(self._concepts) - 1), 1), 6
            ) if len(self._concepts) > 1 else 0.0,
            "dirty": self._dirty,
        }

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, name: str) -> bool:
        return name.strip() in self._concepts
