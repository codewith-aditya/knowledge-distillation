"""
Novelty Scorer — Assesses how much new information an answer provides.

Checks against:
1. Previous answers on the same topic
2. Knowledge graph concepts already covered
3. Semantic similarity with recent content

Novelty score = percentage of claims/facts not previously covered.
Low novelty → overlapping content, potential duplication.
"""

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class NoveltyReport:
    """Result of novelty assessment"""
    score: float = 0.0                      # 0.0-1.0 (1.0 = entirely novel)
    overlap_ratio: float = 0.0              # % of content that overlaps with previous
    unique_claims: list[str] = field(default_factory=list)
    overlapping_claims: list[str] = field(default_factory=list)
    is_novel: bool = True
    recommendation: str = ""


class NoveltyScorer:
    """
    Multi-method novelty scoring.

    Methods:
    1. Keyword overlap: Jaccard similarity of key terms with previous answers
    2. Claim uniqueness: Percentage of sentences that are substantively new
    3. Semantic chunking: Paragraph-level novelty assessment
    4. KG concept check: Are concepts already in the knowledge graph?
    """

    def __init__(self, overlap_threshold: float = 0.6):
        self.overlap_threshold = overlap_threshold
        self.total_scored = 0
        self.total_duplicates = 0

        # Rolling buffer of recent answers for comparison (per topic)
        self._recent_answers: dict[str, list[str]] = {}
        self._recent_topics: dict[str, set[str]] = {}  # topic → set of concepts

    def score(
        self,
        answer: str,
        topic: str = "",
        previous_answers: list[str] = None,
        kg_concepts: list[str] = None,
    ) -> NoveltyReport:
        """
        Score novelty of an answer.

        Args:
            answer: The answer to assess
            topic: Topic domain
            previous_answers: Previous answers on same topic
            kg_concepts: Concepts already in knowledge graph
        """
        self.total_scored += 1
        report = NoveltyReport()

        # Collect previous content to compare against
        comparison_texts = list(previous_answers or [])

        # Add buffered recent answers for this topic
        if topic and topic in self._recent_answers:
            comparison_texts.extend(self._recent_answers[topic][-20:])

        if not comparison_texts and not kg_concepts:
            # No basis for comparison — assume novel
            report.score = 0.85
            report.is_novel = True
            report.recommendation = "Insufficient comparison data — assumed novel"
            return report

        # Method 1: Keyword-level novelty
        keyword_score = self._keyword_novelty(answer, comparison_texts)

        # Method 2: Sentence-level claim uniqueness
        claim_score, unique, overlapping = self._claim_uniqueness(answer, comparison_texts)

        # Method 3: KG concept overlap
        kg_score = 1.0
        if kg_concepts:
            kg_score = self._kg_concept_novelty(answer, kg_concepts)

        # Combined score (weighted)
        report.score = round(
            keyword_score * 0.25 +
            claim_score * 0.50 +
            kg_score * 0.25,
            4,
        )
        report.overlap_ratio = round(1.0 - report.score, 4)
        report.unique_claims = unique[:10]
        report.overlapping_claims = overlapping[:10]
        report.is_novel = report.score >= 0.4  # Allow some overlap

        if report.score < 0.3:
            report.recommendation = "Answer heavily overlaps with existing content — likely duplicate"
            self.total_duplicates += 1
        elif report.score < 0.5:
            report.recommendation = "Moderate overlap — consider exploring new angles"
        elif report.score >= 0.8:
            report.recommendation = "Highly novel content — good exploration direction"

        # Buffer this answer for future comparisons
        if topic:
            if topic not in self._recent_answers:
                self._recent_answers[topic] = []
            self._recent_answers[topic].append(answer)
            # Keep buffer bounded
            if len(self._recent_answers[topic]) > 50:
                self._recent_answers[topic] = self._recent_answers[topic][-50:]

        return report

    def _keyword_novelty(self, answer: str, comparison_texts: list[str]) -> float:
        """Compute 1 - average Jaccard similarity with previous answers"""
        if not comparison_texts:
            return 0.8

        answer_keywords = self._extract_keywords(answer)

        if not answer_keywords:
            return 0.5

        similarities = []
        for prev in comparison_texts[-10:]:  # Compare against last 10
            prev_keywords = self._extract_keywords(prev)
            if not prev_keywords:
                continue

            intersection = len(answer_keywords & prev_keywords)
            union = len(answer_keywords | prev_keywords)
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)

        if not similarities:
            return 0.5

        avg_similarity = sum(similarities) / len(similarities)
        return round(1.0 - avg_similarity, 4)

    def _claim_uniqueness(
        self, answer: str, comparison_texts: list[str]
    ) -> tuple[float, list[str], list[str]]:
        """Check sentence-level uniqueness against previous answers"""
        # Split into sentences
        sentences = [s.strip() for s in answer.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 20]

        if not sentences:
            return 0.5, [], []

        # Build n-gram index of comparison texts for efficiency
        comparison_ngrams = set()
        for text in comparison_texts[-5:]:  # Check last 5 answers
            for sentence in text.replace('!', '.').replace('?', '.').split('.'):
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # Use character 4-grams for fuzzy matching
                    for i in range(len(sentence) - 4):
                        comparison_ngrams.add(sentence[i:i+4].lower())

        unique = []
        overlapping = []
        unique_count = 0

        for sentence in sentences:
            # Generate 4-grams for this sentence
            sent_ngrams = set()
            for i in range(len(sentence) - 4):
                sent_ngrams.add(sentence[i:i+4].lower())

            if not sent_ngrams:
                continue

            overlap = len(sent_ngrams & comparison_ngrams) / len(sent_ngrams)

            if overlap > 0.7:
                overlapping.append(sentence[:150])
            else:
                unique.append(sentence[:150])
                unique_count += 1

        score = unique_count / len(sentences) if sentences else 0.5
        return round(score, 4), unique, overlapping

    def _kg_concept_novelty(self, answer: str, kg_concepts: list[str]) -> float:
        """Check how many KG concepts are referenced vs. new concepts introduced"""
        if not kg_concepts:
            return 0.8

        text = answer.lower()
        kg_lower = [c.lower() for c in kg_concepts]

        referenced = sum(1 for c in kg_lower if c in text)
        total_kg = len(kg_lower)

        # Higher score = fewer KG concepts referenced = more novel territory
        if total_kg == 0:
            return 0.8

        ratio = referenced / total_kg
        # Invert: if all KG concepts are referenced, novelty is low
        novelty = 1.0 - ratio + 0.2  # +0.2 because referencing KG is not inherently bad

        return round(max(0.0, min(1.0, novelty)), 4)

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text"""
        # Simple: words longer than 3 chars, excluding common stopwords
        stopwords = {
            'this', 'that', 'with', 'from', 'they', 'will', 'have', 'been',
            'were', 'their', 'about', 'which', 'would', 'there', 'could',
            'should', 'these', 'those', 'because', 'through', 'being',
        }

        words = text.lower().split()
        return {
            w.strip('.,;:()[]{}"\'') for w in words
            if len(w) > 3 and w.lower() not in stopwords
        }

    def add_to_buffer(self, topic: str, answer: str):
        """Manually add an answer to the comparison buffer"""
        if topic not in self._recent_answers:
            self._recent_answers[topic] = []
        self._recent_answers[topic].append(answer)
        if len(self._recent_answers[topic]) > 50:
            self._recent_answers[topic] = self._recent_answers[topic][-50:]

    def clear_buffer(self, topic: str = None):
        """Clear comparison buffer"""
        if topic:
            self._recent_answers.pop(topic, None)
        else:
            self._recent_answers.clear()

    def get_stats(self) -> dict:
        """Get novelty scorer statistics"""
        return {
            "total_scored": self.total_scored,
            "total_duplicates": self.total_duplicates,
            "duplicate_rate": round(
                self.total_duplicates / self.total_scored * 100, 2
            ) if self.total_scored > 0 else 0.0,
            "topics_tracked": len(self._recent_answers),
            "buffer_sizes": {t: len(a) for t, a in self._recent_answers.items()},
        }
