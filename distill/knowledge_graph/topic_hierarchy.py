"""
Topic Hierarchy — Manages parent-child topic decomposition.

A broad topic "Machine Learning" decomposes into sub-topics,
each with their own coverage scores. Hierarchical coverage =
weighted average of sub-topic coverage.

Used for:
- Systematic topic decomposition
- Hierarchical coverage scoring
- Breadth-first exploration within a topic tree
- Parent exhaustion when all children exhausted
"""

from dataclasses import dataclass, field
from typing import Optional

from distill.knowledge_graph.kg_manager import KnowledgeGraph


@dataclass
class TopicNode:
    """A node in the topic hierarchy tree"""
    name: str
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    coverage_score: float = 0.0
    is_exhausted: bool = False
    priority: int = 5  # 1-10


class TopicHierarchy:
    """
    Manages hierarchical topic decomposition.

    Example:
        Machine Learning
        ├── Supervised Learning
        │   ├── Linear Regression
        │   ├── Decision Trees
        │   └── Neural Networks
        ├── Unsupervised Learning
        │   ├── Clustering
        │   └── Dimensionality Reduction
        └── Reinforcement Learning
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self._roots: list[str] = []          # Top-level topics
        self._topics: dict[str, TopicNode] = {}  # All topics

    def add_topic(
        self,
        name: str,
        parent: Optional[str] = None,
        priority: int = 5,
    ) -> TopicNode:
        """Add a topic to the hierarchy"""
        name = name.strip()

        if name in self._topics:
            node = self._topics[name]
            if parent:
                node.parent = parent
            if priority != 5:
                node.priority = priority
            return node

        depth = 0
        if parent and parent in self._topics:
            depth = self._topics[parent].depth + 1

        node = TopicNode(
            name=name,
            parent=parent,
            depth=depth,
            priority=priority,
        )
        self._topics[name] = node

        # Register as child of parent
        if parent:
            if parent in self._topics:
                self._topics[parent].children.append(name)
            else:
                # Create parent placeholder
                parent_node = TopicNode(name=parent, depth=0)
                parent_node.children.append(name)
                self._topics[parent] = parent_node

        # If no parent, it's a root
        if not parent and name not in self._roots:
            self._roots.append(name)

        # Register in KG
        if not self.kg.has_concept(name):
            self.kg.add_concept(name, metadata={"depth": depth, "priority": priority})

        # Add subtopic_of relationship if has parent
        if parent:
            self.kg.add_relationship(name, parent, "subtopic_of")

        return node

    def get_children(self, name: str) -> list[str]:
        """Get direct children of a topic"""
        node = self._topics.get(name.strip())
        return node.children if node else []

    def get_parent(self, name: str) -> Optional[str]:
        """Get parent of a topic"""
        node = self._topics.get(name.strip())
        return node.parent if node else None

    def get_ancestors(self, name: str) -> list[str]:
        """Get all ancestors (parent chain)"""
        ancestors = []
        current = self.get_parent(name)
        while current:
            ancestors.append(current)
            current = self.get_parent(current)
        return ancestors

    def get_descendants(self, name: str) -> list[str]:
        """Get all descendants recursively"""
        descendants = []
        for child in self.get_children(name):
            descendants.append(child)
            descendants.extend(self.get_descendants(child))
        return descendants

    def get_siblings(self, name: str) -> list[str]:
        """Get sibling topics (same parent)"""
        parent = self.get_parent(name)
        if not parent:
            return [r for r in self._roots if r != name]
        return [c for c in self.get_children(parent) if c != name]

    def get_depth(self, name: str) -> int:
        """Get topic depth in hierarchy"""
        node = self._topics.get(name.strip())
        return node.depth if node else 0

    def get_hierarchical_coverage(self, name: str) -> float:
        """
        Compute weighted hierarchical coverage.

        For a parent topic: weighted average of children's coverage.
        For a leaf topic: direct coverage from KG.
        """
        node = self._topics.get(name.strip())
        if not node:
            return self.kg.get_coverage(name)

        children = node.children
        if not children:
            return self.kg.get_coverage(name)

        # Weighted average of children
        child_scores = []
        child_weights = []
        for child in children:
            cov = self.get_hierarchical_coverage(child)
            # Weight by priority (higher priority = more weight)
            child_node = self._topics.get(child)
            weight = (child_node.priority if child_node else 5) / 10.0
            child_scores.append(cov)
            child_weights.append(weight)

        if not child_scores:
            return self.kg.get_coverage(name)

        total_weight = sum(child_weights) or 1.0
        weighted_avg = sum(s * w for s, w in zip(child_scores, child_weights)) / total_weight
        return round(weighted_avg, 4)

    def is_parent_exhausted(self, name: str) -> bool:
        """Check if all children of a topic are exhausted"""
        children = self.get_children(name)
        if not children:
            return self._topics.get(name, TopicNode(name=name)).is_exhausted

        return all(
            self._topics.get(c, TopicNode(name=c)).is_exhausted
            for c in children
        )

    def mark_exhausted(self, name: str):
        """Mark a topic as exhausted and propagate upward"""
        node = self._topics.get(name.strip())
        if node:
            node.is_exhausted = True
            self.kg.set_exhausted(name)

        # Propagate upward
        parent = self.get_parent(name)
        if parent and self.is_parent_exhausted(parent):
            self.mark_exhausted(parent)

    def get_next_to_explore(self, name: str) -> Optional[str]:
        """
        Get the next child to explore (breadth-first, lowest coverage first).
        If all direct children are done, returns None.
        """
        children = self.get_children(name)
        if not children:
            node = self._topics.get(name.strip())
            if node and not node.is_exhausted:
                return name
            return None

        # Get unexplored children, sort by coverage (lowest first)
        unexplored = []
        for child in children:
            child_node = self._topics.get(child)
            if child_node and child_node.is_exhausted:
                continue
            cov = self.kg.get_coverage(child)
            unexplored.append((child, cov))

        if not unexplored:
            return None

        unexplored.sort(key=lambda x: x[1])
        return unexplored[0][0]

    def get_breadth_first_order(self) -> list[str]:
        """Get all topics in breadth-first traversal order"""
        order = []
        visited = set()

        queue = list(self._roots)
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            order.append(current)

            children = self.get_children(current)
            queue.extend(children)

        return order

    def get_roots(self) -> list[str]:
        """Get root topics"""
        return list(self._roots)

    def get_stats(self) -> dict:
        """Get hierarchy statistics"""
        return {
            "total_topics": len(self._topics),
            "roots": len(self._roots),
            "max_depth": max((n.depth for n in self._topics.values()), default=0),
            "avg_children": round(
                sum(len(n.children) for n in self._topics.values()) / max(len(self._topics), 1), 2
            ),
            "exhausted": sum(1 for n in self._topics.values() if n.is_exhausted),
            "topics_by_depth": {
                d: sum(1 for n in self._topics.values() if n.depth == d)
                for d in range(max((n.depth for n in self._topics.values()), default=0) + 1)
            },
        }
