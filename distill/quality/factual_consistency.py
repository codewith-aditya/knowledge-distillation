"""
Factual Consistency Verifier — Cross-reference verification.

Checks:
1. Internal consistency: claims within the answer don't contradict each other
2. Cross-teacher consistency: claims agree across different teachers
3. Knowledge graph consistency: claims align with established facts
4. Logical coherence: reasoning chain has no gaps or leaps
"""

from dataclasses import dataclass, field
import re


@dataclass
class ConsistencyReport:
    """Result of factual consistency verification"""
    score: float = 0.0                     # 0.0-1.0
    internal_consistency: float = 0.0
    cross_teacher_consistency: float = 0.0
    logical_coherence: float = 0.0
    is_consistent: bool = False
    issues: list[str] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)


class FactualConsistencyVerifier:
    """
    Verifies factual consistency across multiple dimensions.

    Does NOT require external fact-checking APIs.
    Uses heuristics, pattern matching, and cross-reference analysis.
    """

    def __init__(self, consistency_threshold: float = 0.80):
        self.consistency_threshold = consistency_threshold
        self.total_verified = 0
        self.total_inconsistent = 0

    def verify(
        self,
        answer: str,
        reasoning: str = "",
        other_teacher_answers: list[str] = None,
        kg_facts: list[str] = None,
    ) -> ConsistencyReport:
        """
        Run all consistency checks.

        Args:
            answer: The merged/primary answer to verify
            reasoning: Associated reasoning
            other_teacher_answers: Answers from other teachers for cross-checking
            kg_facts: Established facts from the knowledge graph
        """
        self.total_verified += 1
        report = ConsistencyReport()

        # 1. Internal consistency
        report.internal_consistency = self._check_internal(answer, reasoning)
        if report.internal_consistency < self.consistency_threshold:
            report.issues.append(f"Internal contradictions detected (score={report.internal_consistency:.2f})")

        # 2. Cross-teacher consistency
        if other_teacher_answers:
            report.cross_teacher_consistency = self._check_cross_teacher(answer, other_teacher_answers)
            if report.cross_teacher_consistency < self.consistency_threshold:
                report.issues.append(f"Cross-teacher contradictions (score={report.cross_teacher_consistency:.2f})")
        else:
            report.cross_teacher_consistency = 1.0  # Can't check

        # 3. Logical coherence
        report.logical_coherence = self._check_logical_coherence(answer, reasoning)
        if report.logical_coherence < self.consistency_threshold:
            report.issues.append(f"Logical gaps or leaps (score={report.logical_coherence:.2f})")

        # 4. KG consistency (if available)
        kg_score = 1.0
        if kg_facts:
            kg_score = self._check_kg(answer, kg_facts)
            if kg_score < self.consistency_threshold:
                report.issues.append(f"Contradicts knowledge graph facts")

        # Weighted overall
        weights = {
            "internal": 0.35,
            "cross_teacher": 0.25 if other_teacher_answers else 0,
            "logical": 0.25,
            "kg": 0.15 if kg_facts else 0,
        }

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        report.score = round(
            report.internal_consistency * weights["internal"] +
            report.cross_teacher_consistency * weights["cross_teacher"] +
            report.logical_coherence * weights["logical"] +
            kg_score * weights["kg"],
            4,
        )

        report.is_consistent = report.score >= self.consistency_threshold

        if not report.is_consistent:
            self.total_inconsistent += 1

        return report

    def _check_internal(self, answer: str, reasoning: str) -> float:
        """Check for internal contradictions within a single answer"""
        score = 1.0
        text = answer.lower()

        # Check for mutually exclusive claims
        # Pattern: "X is always Y" elsewhere "X can be not-Y"
        exclusivity_pairs = [
            (r'\bmust\b', r'\bmay\b', 0.03),
            (r'\balways\b', r'\bsometimes\b', 0.03),
            (r'\brequired\b', r'\boptional\b', 0.03),
            (r'\bonly\b', r'\balso\b', 0.02),
            (r'\bnever\b', r'\bcan\b', 0.03),
        ]

        for pattern_a, pattern_b, penalty in exclusivity_pairs:
            count_a = len(re.findall(pattern_a, text))
            count_b = len(re.findall(pattern_b, text))
            if count_a > 0 and count_b > 0:
                score -= min(penalty * min(count_a, count_b), 0.1)

        # Check for numeric contradictions
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(GB|TB|MB|ms|seconds|minutes|hours|%|rps|qps)', text)
        if len(numbers) > 1:
            # Check if similar metrics have wildly different values
            for i, (val1, unit1) in enumerate(numbers):
                for val2, unit2 in numbers[i+1:]:
                    if unit1 == unit2:
                        try:
                            v1, v2 = float(val1), float(val2)
                            if v1 > 0 and v2 > 0:
                                ratio = max(v1, v2) / min(v1, v2)
                                if ratio > 100:  # Two orders of magnitude difference
                                    score -= 0.05
                        except ValueError:
                            pass

        return max(0.0, score)

    def _check_cross_teacher(self, answer: str, others: list[str]) -> float:
        """Check consistency across different teacher answers"""
        if not others:
            return 1.0

        # Extract key claims (sentences with numbers or technical terms)
        a_claims = self._extract_key_claims(answer)
        if not a_claims:
            return 0.8

        consistent_count = 0
        total_checked = 0

        for claim in a_claims[:20]:  # Check top 20 claims
            total_checked += 1
            claim_ngrams = self._get_ngrams(claim.lower(), 4)

            support_found = False
            for other in others:
                other_ngrams = self._get_ngrams(other.lower(), 4)
                if claim_ngrams and other_ngrams:
                    overlap = len(claim_ngrams & other_ngrams) / len(claim_ngrams)
                    if overlap > 0.5:  # At least 50% of claim's n-grams found in another answer
                        support_found = True
                        break

            if support_found:
                consistent_count += 1

        if total_checked == 0:
            return 0.8

        return round(consistent_count / total_checked, 4)

    def _check_logical_coherence(self, answer: str, reasoning: str) -> float:
        """Check if reasoning chain is logically coherent"""
        score = 0.8

        # Check for logical connectors
        logical_connectors = [
            "therefore", "thus", "hence", "consequently", "as a result",
            "because", "since", "due to", "given that",
            "however", "although", "on the other hand", "in contrast",
            "first", "second", "third", "finally", "lastly",
            "for example", "specifically", "in particular",
        ]

        text = (reasoning + " " + answer).lower()
        connector_count = sum(1 for c in logical_connectors if c in text)

        # Good reasoning should have several logical connectors
        if connector_count >= 10:
            score += 0.15
        elif connector_count >= 5:
            score += 0.1
        elif connector_count >= 2:
            score += 0.05
        else:
            score -= 0.1

        # Check for logical fallacies
        fallacy_patterns = [
            (r'\bbecause.*therefore\b', 0.03),  # Circular reasoning
            (r'\beveryone knows\b', 0.05),       # Bandwagon
            (r'\bexperts agree\b', 0.03),        # Appeal to authority without citation
        ]

        for pattern, penalty in fallacy_patterns:
            if re.search(pattern, text):
                score -= penalty

        return max(0.0, min(1.0, score))

    def _check_kg(self, answer: str, kg_facts: list[str]) -> float:
        """Check answer against knowledge graph established facts"""
        if not kg_facts:
            return 1.0

        text = answer.lower()
        matches = 0
        contradictions = 0

        for fact in kg_facts:
            fact_lower = fact.lower()
            if fact_lower in text:
                matches += 1
            # Simple contradiction check
            elif f"not {fact_lower}" in text:
                contradictions += 1

        if matches + contradictions == 0:
            return 0.8  # No overlap with KG — neutral

        score = 1.0 - (contradictions / max(matches + contradictions, 1)) * 0.4
        return round(max(0.0, score), 4)

    def _extract_key_claims(self, text: str) -> list[str]:
        """Extract key factual claims from text"""
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.')]

        claims = []
        for s in sentences:
            if len(s) < 20:
                continue
            # Prefer sentences with numbers, technical terms, or keywords
            if (re.search(r'\d+', s) or
                re.search(r'\b(is|are|was|were|has|have|can|will|must|should)\b', s.lower()) or
                any(tech in s.lower() for tech in ['algorithm', 'function', 'method', 'process', 'system', 'data', 'code'])):
                claims.append(s)

        return claims

    def _get_ngrams(self, text: str, n: int) -> set[str]:
        """Generate character n-grams from text"""
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def get_stats(self) -> dict:
        """Get verifier statistics"""
        return {
            "total_verified": self.total_verified,
            "total_inconsistent": self.total_inconsistent,
            "inconsistency_rate": round(
                self.total_inconsistent / self.total_verified * 100, 2
            ) if self.total_verified > 0 else 0.0,
            "consistency_threshold": self.consistency_threshold,
        }
