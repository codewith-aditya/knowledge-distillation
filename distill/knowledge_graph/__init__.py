"""
Knowledge Graph & Coverage Tracking Package.

Provides:
- KnowledgeGraph: In-memory graph with JSON persistence
- CoverageTracker: Coverage metrics and trend analysis
- TopicHierarchy: Parent-child topic decomposition
- GapAnalyzer: Identify unexplored areas
"""

from distill.knowledge_graph.kg_manager import KnowledgeGraph, ConceptNode
from distill.knowledge_graph.coverage_tracker import CoverageTracker
from distill.knowledge_graph.topic_hierarchy import TopicHierarchy
from distill.knowledge_graph.gap_analyzer import GapAnalyzer
