"""
Duplicate Detection — Multi-layer: Exact, Semantic, LLM-based.

Three layers of defense:
1. EXACT: SHA-256 hash of normalized question (instant)
2. SEMANTIC: Jaccard word overlap + key phrase extraction (fast)
3. LLM-BASED: Batch comparison of candidate against recent questions (slow but accurate)

Prevents generating duplicate/near-duplicate questions.
Tracks per-topic question history for targeted dedup.
"""

import hashlib
import re
import asyncio
from typing import Dict, Optional
from collections import defaultdict


class DuplicateDetector:
    """
    Multi-layer hybrid duplicate detection.

    Layer 1 (Exact): SHA-256 hash — catches identical questions instantly.
    Layer 2 (Semantic): Jaccard + key phrase overlap — catches reworded questions.
    Layer 3 (LLM): Semantic similarity via LLM — catches same intent, different words.
    """

    def __init__(self, config, llm_client=None):
        self.config = config
        self.llm = llm_client  # Optional — enables Layer 3

        # Layer 1: Exact hash set
        self.exact_hashes: set[str] = set()

        # Layer 2: Per-topic normalized questions
        self.topic_questions: dict[str, list[str]] = defaultdict(list)

        # Layer 2: Key phrase index (extracted keywords per question)
        self.topic_phrases: dict[str, list[set[str]]] = defaultdict(list)

        # Layer 3: LLM dedup cache (topic → recent questions for batch comparison)
        self._llm_dedup_buffer: dict[str, list[str]] = defaultdict(list)
        self._llm_dedup_batch_size = 5  # Check against batches of 5

        # Stats
        self.total_checks = 0
        self.total_duplicates = 0
        self.layer1_hits = 0
        self.layer2_hits = 0
        self.layer3_hits = 0

    # ── Public API ───────────────────────────────────────────

    def is_duplicate(self, question: str, topic: str) -> Dict:
        """
        Check if question is duplicate.

        Returns:
            {
                "is_duplicate": bool,
                "method": "exact" | "semantic" | "llm" | None,
                "similarity": float,
                "matched_question": str or None
            }
        """
        self.total_checks += 1

        # ── Layer 1: Exact hash ──
        q_hash = self._compute_hash(question)
        if q_hash in self.exact_hashes:
            self.total_duplicates += 1
            self.layer1_hits += 1
            return {
                "is_duplicate": True,
                "method": "exact",
                "similarity": 1.0,
                "matched_question": None,
            }

        # ── Layer 2: Semantic overlap ──
        result = self._semantic_check(question, topic)
        if result["is_duplicate"]:
            self.total_duplicates += 1
            self.layer2_hits += 1
            return result

        return {
            "is_duplicate": False,
            "method": None,
            "similarity": 0.0,
            "matched_question": None,
        }

    async def is_duplicate_async(self, question: str, topic: str) -> Dict:
        """
        Async check including LLM layer. Call this when LLM client is available.
        """
        # Run Layer 1 + 2 first
        result = self.is_duplicate(question, topic)
        if result["is_duplicate"]:
            return result

        # ── Layer 3: LLM-based semantic check ──
        if self.llm and topic in self.topic_questions:
            recent = self.topic_questions[topic][-10:]
            if recent:
                llm_result = await self._llm_check(question, recent, topic)
                if llm_result["is_duplicate"]:
                    self.total_duplicates += 1
                    self.layer3_hits += 1
                    return llm_result

        return result

    def add_question(self, question: str, topic: str):
        """Add question to all indexes"""
        normalized = self._normalize(question)

        # Layer 1
        q_hash = self._compute_hash(question)
        self.exact_hashes.add(q_hash)

        # Layer 2
        self.topic_questions[topic].append(normalized)
        self.topic_phrases[topic].append(self._extract_key_phrases(normalized))

        # Layer 3 buffer
        self._llm_dedup_buffer[topic].append(question)
        if len(self._llm_dedup_buffer[topic]) > 50:
            self._llm_dedup_buffer[topic] = self._llm_dedup_buffer[topic][-50:]

    # ── Shared (Redis-backed) dedup — survives restarts, spans workers ──

    async def check_and_register(self, question: str, topic: str, redis) -> Dict:
        """
        Cross-process duplicate check.

        The in-memory indexes above are per-worker and evaporate on restart, so
        two question workers happily generate the same question (8,691 of 20k
        samples were repeat questions). This path keeps the authoritative state
        in Redis:

          distill:qhash            SET of exact question hashes
          distill:qtopic:<slug>    LIST of the last N normalized questions

        Returns the same dict shape as is_duplicate() and registers the
        question when it is unique.
        """
        if redis is None:
            return self.is_duplicate(question, topic)

        self.total_checks += 1
        normalized = self._normalize(question)
        q_hash = self._compute_hash(question)

        # Layer 1 — exact, global across workers/restarts
        is_new = await redis.sadd("distill:qhash", q_hash)
        if is_new == 0:
            self.total_duplicates += 1
            self.layer1_hits += 1
            return {"is_duplicate": True, "method": "exact", "similarity": 1.0,
                    "matched_question": None}

        # Layer 2 — semantic against the shared per-topic history
        slug = self._topic_slug(topic)
        key = f"distill:qtopic:{slug}"
        recent = await redis.lrange(key, 0, 199)
        q_phrases = self._extract_key_phrases(normalized)
        threshold = getattr(self.config, "duplicate_semantic_threshold", 0.90)

        for existing in recent or []:
            existing = existing if isinstance(existing, str) else existing.decode()
            jaccard = self._jaccard_similarity(normalized, existing)
            phrase_overlap = self._phrase_overlap(
                q_phrases, self._extract_key_phrases(existing)
            )
            combined = jaccard * 0.4 + phrase_overlap * 0.6
            if combined >= threshold:
                self.total_duplicates += 1
                self.layer2_hits += 1
                # Roll back the hash we just registered, so a later different
                # wording of the same idea is still checked semantically.
                await redis.srem("distill:qhash", q_hash)
                return {"is_duplicate": True, "method": "semantic",
                        "similarity": round(combined, 4),
                        "matched_question": existing[:200]}

        # Unique — register in the shared history
        await redis.lpush(key, normalized[:500])
        await redis.ltrim(key, 0, 499)
        self.topic_questions[topic].append(normalized)
        self.topic_phrases[topic].append(q_phrases)
        return {"is_duplicate": False, "method": None, "similarity": 0.0,
                "matched_question": None}

    @staticmethod
    def _topic_slug(topic: str) -> str:
        import hashlib
        return hashlib.sha1(topic.lower().strip().encode("utf-8")).hexdigest()[:16]

    # ── Layer 2: Semantic ────────────────────────────────────

    def _semantic_check(self, question: str, topic: str) -> Dict:
        """Jaccard + key phrase overlap check"""
        normalized = self._normalize(question)
        q_phrases = self._extract_key_phrases(normalized)

        if topic not in self.topic_questions:
            return {"is_duplicate": False, "method": None, "similarity": 0.0, "matched_question": None}

        threshold = getattr(self.config, 'duplicate_semantic_threshold', 0.85)

        for i, existing_q in enumerate(self.topic_questions[topic]):
            # Jaccard similarity
            jaccard = self._jaccard_similarity(normalized, existing_q)

            # Key phrase overlap
            if i < len(self.topic_phrases[topic]):
                existing_phrases = self.topic_phrases[topic][i]
                phrase_overlap = self._phrase_overlap(q_phrases, existing_phrases)
            else:
                phrase_overlap = 0.0

            # Combined score (weighted)
            combined = jaccard * 0.4 + phrase_overlap * 0.6

            if combined >= threshold:
                return {
                    "is_duplicate": True,
                    "method": "semantic",
                    "similarity": round(combined, 4),
                    "matched_question": existing_q[:200],
                }

        return {"is_duplicate": False, "method": None, "similarity": 0.0, "matched_question": None}

    # ── Layer 3: LLM-based ───────────────────────────────────

    async def _llm_check(self, question: str, recent: list[str], topic: str) -> Dict:
        """
        Ask LLM: "Is this new question semantically identical to any of these?"

        Batch check — compares against up to 10 recent questions at once.
        """
        candidates = recent[-10:]

        prompt = f"""# ROLE: Duplicate Question Detector

Check if the NEW QUESTION is semantically the same as any PREVIOUS QUESTION.
"Semantically same" means: asks for the same information even if worded differently.

# NEW QUESTION
{question}

# PREVIOUS QUESTIONS
"""
        for i, q in enumerate(candidates, 1):
            prompt += f"{i}. {q[:300]}\n"

        prompt += """
# YOUR TASK
Is the NEW QUESTION asking for the SAME information as any previous question?

Reply with EXACTLY one line:
DUPLICATE: <question_number> | SIMILARITY: <0.0-1.0> | REASON: <one line>
OR
NOT_DUPLICATE

Examples:
DUPLICATE: 3 | SIMILARITY: 0.92 | REASON: Both ask about B+ tree insertion algorithms, just rephrased
NOT_DUPLICATE"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )

            content = response["content"].strip()

            if content.startswith("DUPLICATE:"):
                # Parse similarity
                sim_match = re.search(r'SIMILARITY:\s*([\d.]+)', content)
                similarity = float(sim_match.group(1)) if sim_match else 0.8

                if similarity >= 0.80:  # LLM duplicate threshold
                    # Extract matched question number
                    num_match = re.search(r'DUPLICATE:\s*(\d+)', content)
                    if num_match:
                        idx = int(num_match.group(1)) - 1
                        matched = candidates[idx][:200] if 0 <= idx < len(candidates) else None
                    else:
                        matched = None

                    return {
                        "is_duplicate": True,
                        "method": "llm",
                        "similarity": similarity,
                        "matched_question": matched,
                    }

        except Exception:
            pass

        return {"is_duplicate": False, "method": None, "similarity": 0.0, "matched_question": None}

    # ── Utility methods ──────────────────────────────────────

    def _normalize(self, question: str) -> str:
        """Normalize question: lowercase, strip punctuation, collapse whitespace"""
        q = question.lower().strip()
        q = re.sub(r'[^\w\s]', ' ', q)  # Replace punctuation with space
        q = re.sub(r'\s+', ' ', q)       # Collapse whitespace
        return q.strip()

    def _compute_hash(self, question: str) -> str:
        """SHA-256 hash of normalized question"""
        return hashlib.sha256(self._normalize(question).encode('utf-8')).hexdigest()

    def _jaccard_similarity(self, q1: str, q2: str) -> float:
        """Jaccard similarity on words"""
        w1 = set(q1.split())
        w2 = set(q2.split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def _extract_key_phrases(self, text: str) -> set[str]:
        """Extract key phrases (2-3 word ngrams of meaningful words)"""
        words = text.split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
                     'and', 'or', 'but', 'not', 'so', 'if', 'then', 'than', 'that',
                     'this', 'these', 'those', 'it', 'its', 'can', 'will', 'would',
                     'could', 'should', 'may', 'might', 'has', 'have', 'had', 'do',
                     'does', 'did', 'what', 'how', 'why', 'when', 'where', 'which',
                     'who', 'whom', 'about', 'into', 'over', 'after', 'before'}

        meaningful = [w for w in words if w not in stopwords and len(w) > 2]

        phrases = set()
        # Bigrams
        for i in range(len(meaningful) - 1):
            phrases.add(f"{meaningful[i]} {meaningful[i+1]}")
        # Trigrams
        for i in range(len(meaningful) - 2):
            phrases.add(f"{meaningful[i]} {meaningful[i+1]} {meaningful[i+2]}")

        return phrases

    def _phrase_overlap(self, phrases1: set[str], phrases2: set[str]) -> float:
        """Jaccard on key phrases (more semantic than word-level)"""
        if not phrases1 or not phrases2:
            return 0.0
        return len(phrases1 & phrases2) / len(phrases1 | phrases2)

    # ── Bulk operations ──────────────────────────────────────

    def get_topic_questions(self, topic: str) -> list[str]:
        """Get all questions for a topic"""
        return list(self.topic_questions.get(topic, []))

    def get_topic_count(self, topic: str) -> int:
        """Number of questions for a topic"""
        return len(self.topic_questions.get(topic, []))

    def clear_topic(self, topic: str):
        """Clear all data for a topic"""
        self.topic_questions.pop(topic, None)
        self.topic_phrases.pop(topic, None)
        self._llm_dedup_buffer.pop(topic, None)

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get duplicate detection statistics"""
        return {
            "total_checks": self.total_checks,
            "total_duplicates": self.total_duplicates,
            "duplicate_rate": round(
                self.total_duplicates / self.total_checks * 100, 2
            ) if self.total_checks > 0 else 0.0,
            "layer1_exact_hits": self.layer1_hits,
            "layer2_semantic_hits": self.layer2_hits,
            "layer3_llm_hits": self.layer3_hits,
            "unique_questions": len(self.exact_hashes),
            "topics_tracked": len(self.topic_questions),
            "questions_per_topic": {
                t: len(qs) for t, qs in self.topic_questions.items()
            },
        }
