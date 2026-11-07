"""
Hallucination Detector — Self-consistency based verification.

Strategy: Ask the SAME question framed 3 different ways to the teacher.
If answers diverge significantly on factual claims → high hallucination probability.

Also checks for:
- Claims contradicting established knowledge graph facts
- Numeric fabrications (improbable precision)
- Non-existent references (fake papers, invented URLs)
- Internal contradiction within a single answer
"""

import asyncio
import re
from dataclasses import dataclass, field


@dataclass
class HallucinationReport:
    """Result of hallucination detection"""
    probability: float = 0.0          # 0.0 (no hallucination) to 1.0 (certain)
    is_likely_hallucination: bool = False
    suspicious_claims: list[str] = field(default_factory=list)
    consistency_score: float = 1.0    # 1.0 = perfectly consistent
    signals: dict[str, bool] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


class HallucinationDetector:
    """
    Multi-signal hallucination detection without requiring external fact-checking APIs.

    Signals:
    1. Self-consistency: Same question, different framings → answer divergence
    2. Internal consistency: Does the answer contradict itself?
    3. Numeric precision: Unrealistically precise numbers without source
    4. Reference validity: Fake-looking references
    5. Overconfidence: Definitive claims on uncertain topics
    6. Knowledge graph consistency: Contradicts established facts
    """

    # Known hallucination markers
    HALLUCINATION_MARKERS = [
        # Vague attribution
        r"studies (have )?shown",
        r"research (has )?shown",
        r"experts (agree|say|believe)",
        r"it is (widely|generally|commonly) (known|accepted|believed)",
        r"many (people|experts|scientists) (believe|think|say)",
        # Overconfident language
        r"without (any|a) doubt",
        r"it is (absolutely|certainly|definitely|undoubtedly) (true|the case)",
        r"there is no (question|doubt) that",
        # Fake reference patterns
        r"according to (a|the) (recent )?(study|report|survey|paper)",
        r"a (recent )?(study|report|survey) (found|showed|revealed|concluded)",
    ]

    # Patterns suggesting fabricated references
    SUSPICIOUS_REFERENCE_PATTERNS = [
        r"(Smith|Johnson|Williams|Brown|Jones|Davis|Miller|Wilson|Moore|Taylor|Anderson|Thomas|Jackson) et al\.?\s*\(?(20\d{2})\)?",
        r"\[1\]\s*[A-Z][a-z]+,\s*[A-Z]\.\s*\(?20\d{2}\)?",
    ]

    def __init__(self, consistency_threshold: float = 0.75):
        self.consistency_threshold = consistency_threshold
        self.total_checked = 0
        self.total_flagged = 0

    def detect(
        self,
        question: str,
        answer: str,
        reasoning: str,
        alternative_answers: list[str] = None,
        kg_facts: list[str] = None,
    ) -> HallucinationReport:
        """
        Detect hallucination probability in an answer.

        Args:
            question: Original question
            answer: Answer to check
            reasoning: Reasoning portion of the answer
            alternative_answers: Answers to the same question framed differently
            kg_facts: Known facts from the knowledge graph
        """
        self.total_checked += 1
        report = HallucinationReport()

        # Signal 1: Internal consistency
        internal_score = self._check_internal_consistency(answer)
        report.signals["internal_consistency"] = internal_score < 0.85
        report.consistency_score = min(report.consistency_score, internal_score)

        # Signal 2: Cross-framing consistency (if available)
        if alternative_answers:
            cross_score = self._check_cross_framing_consistency(answer, alternative_answers)
            report.signals["cross_framing_consistency"] = cross_score < self.consistency_threshold
            report.consistency_score = min(report.consistency_score, cross_score)

        # Signal 3: Numeric fabrication
        num_score = self._check_numeric_fabrication(answer)
        report.signals["numeric_fabrication"] = num_score < 0.7

        # Signal 4: Suspicious references
        ref_score = self._check_references(answer)
        report.signals["suspicious_references"] = ref_score < 0.7

        # Signal 5: Overconfidence
        conf_score = self._check_overconfidence(answer)
        report.signals["overconfidence"] = conf_score < 0.7

        # Signal 6: Reasoning-answer alignment
        alignment_score = self._check_reasoning_answer_alignment(reasoning, answer)
        report.signals["reasoning_answer_mismatch"] = alignment_score < 0.7

        # Signal 7: KG consistency (if available)
        if kg_facts:
            kg_score = self._check_kg_consistency(answer, kg_facts)
            report.signals["kg_contradiction"] = kg_score < 0.8
            report.consistency_score = min(report.consistency_score, kg_score)

        # Compute hallucination probability from signals
        num_signals = sum(report.signals.values())
        total_checks = len(report.signals)

        base_prob = num_signals / max(total_checks, 1)
        consistency_factor = 1.0 - report.consistency_score

        report.probability = round(min(1.0, base_prob * 0.7 + consistency_factor * 0.3), 4)
        report.is_likely_hallucination = report.probability > 0.3

        if report.is_likely_hallucination:
            self.total_flagged += 1

        # Generate recommendations
        if report.signals.get("internal_consistency"):
            report.recommendations.append("Answer contains internal contradictions — review logic")
        if report.signals.get("numeric_fabrication"):
            report.recommendations.append("Precise numbers without sources — verify or add citations")
        if report.signals.get("suspicious_references"):
            report.recommendations.append("References appear fabricated — verify or remove")
        if report.signals.get("overconfidence"):
            report.recommendations.append("Overly confident language on complex topics — add nuance")
        if report.signals.get("reasoning_answer_mismatch"):
            report.recommendations.append("Reasoning and answer seem disconnected — ensure alignment")

        return report

    # ── Signal detectors ─────────────────────────────────────

    def _check_internal_consistency(self, answer: str) -> float:
        """Check if the answer contradicts itself"""
        score = 1.0

        # Check for common contradiction patterns
        text = answer.lower()

        # Pattern: saying something is both X and not-X
        contradiction_pairs = [
            ("always", "never"),
            ("must", "optional"),
            ("required", "not necessary"),
            ("critical", "unimportant"),
            ("fast", "slow"),
            ("simple", "complex"),
        ]

        for a, b in contradiction_pairs:
            if a in text and b in text:
                # Check if they're close together (same sentence or paragraph)
                # Simple heuristic: penalize if both appear
                score -= 0.02

        return max(0.0, score)

    def _check_cross_framing_consistency(
        self, answer: str, alternatives: list[str]
    ) -> float:
        """Check consistency between different phrasings of the same answer"""
        if not alternatives:
            return 1.0

        a_words = set(w.lower() for w in answer.split() if len(w) > 4)

        scores = []
        for alt in alternatives:
            alt_words = set(w.lower() for w in alt.split() if len(w) > 4)
            if a_words and alt_words:
                overlap = len(a_words & alt_words) / max(len(a_words), len(alt_words))
                scores.append(overlap)

        if not scores:
            return 1.0

        return round(sum(scores) / len(scores), 4)

    def _check_numeric_fabrication(self, answer: str) -> float:
        """Check for suspiciously precise numbers without sources"""
        score = 1.0

        # Find numbers with high precision
        precise_patterns = re.findall(r'\b\d+\.\d{3,}%?\b', answer)
        if len(precise_patterns) > 3:
            score -= 0.2
        elif len(precise_patterns) > 1:
            score -= 0.1

        # Very specific statistics
        stat_patterns = re.findall(
            r'\b(in|about|approximately|around|roughly|nearly|almost)\s+\d{1,3}(?:\.\d)?\s*%',
            answer.lower(),
        )
        if len(stat_patterns) > 3:
            score -= 0.15

        # Numbers that look like fabricated benchmarks
        bench_patterns = re.findall(
            r'(\d+\.?\d*)\s*(ms|seconds?|minutes?|hours?|days?|GB|TB|MB|rps|qps|tps)',
            answer.lower(),
        )
        if len(bench_patterns) > 5 and "benchmark" not in answer.lower():
            score -= 0.1

        return max(0.0, score)

    def _check_references(self, answer: str) -> float:
        """Check for suspicious/fabricated references"""
        score = 1.0

        # Check for patterns matching hallucinated references
        for pattern in self.SUSPICIOUS_REFERENCE_PATTERNS:
            matches = re.findall(pattern, answer)
            if len(matches) > 2:
                score -= 0.1
            elif len(matches) > 0:
                score -= 0.05

        # Check hallucination marker patterns
        for pattern in self.HALLUCINATION_MARKERS:
            matches = re.findall(pattern, answer.lower())
            if len(matches) > 3:
                score -= 0.08

        return max(0.0, score)

    def _check_overconfidence(self, answer: str) -> float:
        """Check for overconfident language on uncertain topics"""
        score = 1.0

        text = answer.lower()

        overconfident_patterns = [
            (r'\balways\b', 0.02),
            (r'\bnever\b', 0.02),
            (r'\babsolutely\b', 0.03),
            (r'\bundoubtedly\b', 0.03),
            (r'\bwithout (any|a) doubt\b', 0.04),
            (r'\bit is (obvious|clear) that\b', 0.03),
            (r'\beveryone knows\b', 0.05),
            (r'\bno (one|question|doubt)\b', 0.03),
        ]

        for pattern, penalty in overconfident_patterns:
            matches = re.findall(pattern, text)
            score -= min(len(matches) * penalty, penalty * 3)

        return max(0.0, score)

    def _check_reasoning_answer_alignment(self, reasoning: str, answer: str) -> float:
        """Check if reasoning and answer are about the same thing"""
        if not reasoning or not answer:
            return 0.8  # Can't assess without both

        r_words = set(w.lower() for w in reasoning.split() if len(w) > 4)
        a_words = set(w.lower() for w in answer.split() if len(w) > 4)

        if not r_words or not a_words:
            return 0.8

        overlap = len(r_words & a_words) / max(len(r_words), len(a_words))
        return round(overlap, 4)

    def _check_kg_consistency(self, answer: str, kg_facts: list[str]) -> float:
        """Check answer against established knowledge graph facts"""
        if not kg_facts:
            return 1.0

        text = answer.lower()
        contradictions = 0

        for fact in kg_facts:
            fact_lower = fact.lower()
            # Simple negation check — if fact says X and answer says not-X
            if fact_lower in text:
                continue  # Consistent
            # This is a very simplified check — production would use embeddings
            if "not " + fact_lower in text or fact_lower.replace("is ", "is not ") in text:
                contradictions += 1

        score = 1.0 - (contradictions * 0.15)
        return max(0.0, score)

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get hallucination detector statistics"""
        return {
            "total_checked": self.total_checked,
            "total_flagged": self.total_flagged,
            "flag_rate": round(
                self.total_flagged / self.total_checked * 100, 2
            ) if self.total_checked > 0 else 0.0,
            "consistency_threshold": self.consistency_threshold,
        }
