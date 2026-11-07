"""
Topic Discovery Engine - Automatic knowledge graph expansion
Discovers related topics and continuously expands the knowledge extraction
"""

import asyncio
from typing import List, Dict, Optional
from distill.llm.client import LLMClient


class TopicDiscoveryEngine:
    """
    Automatically discovers related topics for continuous knowledge expansion

    Strategies:
    1. Extract concepts from answers
    2. Ask research agent for related topics
    3. Identify unexplored branches
    4. Build knowledge graph
    """

    def __init__(self, llm_client: LLMClient, config):
        self.llm = llm_client
        self.config = config
        self.discovered_topics = set()

    async def discover_related_topics(
        self,
        current_topic: str,
        recent_qa_pairs: List[Dict],
        max_topics: int = 5
    ) -> List[Dict]:
        """
        Discover new related topics from current conversation

        Returns list of:
            {
                "topic": str,
                "reason": str,
                "priority": int
            }
        """

        context = self._build_discovery_context(current_topic, recent_qa_pairs)

        prompt = f"""You are a Topic Discovery Engine. Your goal is to identify new related topics that should be explored to build a comprehensive knowledge base.

CURRENT TOPIC: {current_topic}

RECENT Q&A CONTEXT:
{context}

YOUR TASK:
Analyze the Q&A pairs above and identify {max_topics} NEW related topics that would be valuable to explore.

Look for:
- Concepts mentioned but not fully explored
- Prerequisites or foundational topics
- Advanced applications
- Related methodologies
- Alternative approaches
- Practical implementations
- Common problems and solutions

For each topic, provide:
1. Topic name (specific and focused)
2. Why it's valuable to explore
3. Priority (1-10, where 10 is most important)

FORMAT (one per line):
TOPIC: [topic name] | REASON: [why valuable] | PRIORITY: [1-10]

Generate {max_topics} related topics:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )

            topics = self._parse_topics(response["content"])
            return [t for t in topics if t["topic"] not in self.discovered_topics][:max_topics]

        except Exception as e:
            return []

    def _build_discovery_context(self, topic: str, qa_pairs: List[Dict]) -> str:
        """Build context from recent Q&A pairs"""
        context_parts = []
        for i, qa in enumerate(qa_pairs[-3:], 1):
            q = qa.get("question", "")
            a = qa.get("answer", "")[:300]
            context_parts.append(f"Q{i}: {q}\nA{i}: {a}...")

        return "\n\n".join(context_parts) if context_parts else "No previous Q&A"

    def _parse_topics(self, content: str) -> List[Dict]:
        """Parse topics from LLM response"""
        topics = []

        for line in content.split('\n'):
            if line.startswith("TOPIC:"):
                try:
                    parts = line.split('|')
                    topic = parts[0].replace("TOPIC:", "").strip()
                    reason = parts[1].replace("REASON:", "").strip() if len(parts) > 1 else ""
                    priority_str = parts[2].replace("PRIORITY:", "").strip() if len(parts) > 2 else "5"
                    priority = int(priority_str) if priority_str.isdigit() else 5

                    topics.append({
                        "topic": topic,
                        "reason": reason,
                        "priority": priority
                    })
                except:
                    continue

        return sorted(topics, key=lambda x: x["priority"], reverse=True)

    def mark_discovered(self, topic: str):
        """Mark topic as discovered"""
        self.discovered_topics.add(topic)

    def get_stats(self) -> Dict:
        """Get discovery statistics"""
        return {
            "total_topics_discovered": len(self.discovered_topics)
        }
