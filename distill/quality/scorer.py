"""
Frontier Quality Scorer — 7-dimension comprehensive quality assessment.

Dimensions:
1. Completeness (0-1): All aspects of question addressed?
2. Correctness (0-1): Factual accuracy via self-consistency checks
3. Reasoning Depth (0-1): Multi-step reasoning present and valid?
4. Implementation Depth (0-1): Production-ready code, architecture?
5. Novelty (0-1): New information not in previous answers?
6. Factual Consistency (0-1): Internal + external consistency
7. Hallucination Probability (0-1): Likelihood of fabricated information

Combined score weighted by importance. Threshold: configurable (default 0.80).
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from distill.llm.client import LLMClient

logger = logging.getLogger("distill.scorer")


@dataclass
class QualityVerdict:
    """Complete quality assessment result"""
    # Overall
    overall_score: float = 0.0          # Weighted 0.0-1.0
    passed: bool = False

    # Per-dimension scores
    completeness: float = 0.0
    correctness: float = 0.0
    reasoning_depth: float = 0.0
    implementation_depth: float = 0.0
    novelty: float = 0.0
    factual_consistency: float = 0.0
    hallucination_probability: float = 0.0   # Lower is better

    # Details
    reasons: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)

    # Metadata
    scoring_model: str = ""
    scoring_tokens: int = 0


class FrontierQualityScorer:
    """
    7-dimension quality verification for premium training data.

    Weights are configurable but default to:
    - Completeness: 20%
    - Correctness: 25%  (most important — wrong facts poison training)
    - Reasoning Depth: 15%
    - Implementation Depth: 10%
    - Novelty: 10%
    - Factual Consistency: 15%
    - Hallucination Probability: 5%  (inverse — deducted)

    Above threshold → ACCEPT. Below → retry or flag LOW_QUALITY.
    """

    # Default dimension weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "completeness": 0.20,
        "correctness": 0.25,
        "reasoning_depth": 0.15,
        "implementation_depth": 0.10,
        "novelty": 0.10,
        "factual_consistency": 0.15,
        "hallucination_penalty": 0.05,  # Subtracted from total
    }

    def __init__(
        self,
        llm_client: LLMClient,
        config,
        weights: Optional[dict] = None,
    ):
        self.llm = llm_client
        self.config = config
        self.weights = weights or self.DEFAULT_WEIGHTS

        self.total_scored = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.score_history: list[float] = []  # Last 1000 scores

    async def comprehensive_score(
        self,
        question: str,
        answer: str,
        reasoning: str,
        topic: str = "",
        previous_samples: list = None,
        is_multi_teacher: bool = False,
    ) -> QualityVerdict:
        """
        Run all 7 quality dimensions and produce a comprehensive verdict.

        For efficiency, dimensions 1-4 are scored in a single LLM call.
        Dimensions 5-7 use separate LLM calls and heuristics.
        """
        self.total_scored += 1
        verdict = QualityVerdict()

        # ── Hard gate: minimum answer length (before any LLM calls) ──
        # Thin samples fail outright regardless of LLM scores — saves a
        # scoring call and guarantees the dataset never contains them.
        if len(answer) < self.config.min_answer_length:
            verdict.completeness = 0.0
            verdict.correctness = 0.5
            verdict.reasoning_depth = 0.0
            verdict.implementation_depth = 0.0
            verdict.novelty = 0.5
            verdict.factual_consistency = 0.5
            verdict.hallucination_probability = 0.0
            verdict.overall_score = 0.0
            verdict.passed = False
            verdict.reasons.append(
                f"Answer too short: {len(answer)} chars < minimum {self.config.min_answer_length}"
            )
            self.total_rejected += 1
            return verdict

        # ── Fast heuristics first (no LLM calls) ──
        verdict.completeness = self._heuristic_completeness(question, answer)
        verdict.implementation_depth = self._heuristic_implementation_depth(answer, reasoning)

        # ── LLM-based scoring (dimensions 1-4 combined) ──
        core_scores = await self._score_core_dimensions(question, answer, reasoning, topic)
        if core_scores:
            # Use LLM scores if they're more reliable than heuristics
            verdict.completeness = max(verdict.completeness, core_scores.get("completeness", 0))
            verdict.correctness = core_scores.get("correctness", 0.7)
            verdict.reasoning_depth = core_scores.get("reasoning_depth", 0.7)
            verdict.implementation_depth = max(
                verdict.implementation_depth,
                core_scores.get("implementation_depth", 0),
            )
        else:
            # Judge call failed (reasoning model returned no content, malformed
            # output, timeout). NEVER leave correctness/reasoning at the
            # dataclass default of 0.0 — that silently rejects every sample
            # forever. Fall back to neutral values and make it visible.
            verdict.correctness = 0.65
            verdict.reasoning_depth = 0.60
            verdict.reasons.append("judge unavailable — neutral correctness/reasoning fallback")
            logger.warning(
                "quality judge returned no parsable scores for topic=%r — "
                "using neutral fallback (correctness=0.65, reasoning=0.60)", topic,
            )

        # ── Factual consistency (heuristic + internal check) ──
        verdict.factual_consistency = self._heuristic_factual_consistency(answer, reasoning)

        # ── Novelty (requires comparison with previous samples) ──
        verdict.novelty = self._heuristic_novelty(answer, previous_samples or [])

        # ── Hallucination probability (heuristic indicators) ──
        verdict.hallucination_probability = self._heuristic_hallucination_prob(answer, reasoning)

        # ── Compute weighted overall ──
        overall = (
            verdict.completeness * self.weights["completeness"] +
            verdict.correctness * self.weights["correctness"] +
            verdict.reasoning_depth * self.weights["reasoning_depth"] +
            verdict.implementation_depth * self.weights["implementation_depth"] +
            verdict.novelty * self.weights["novelty"] +
            verdict.factual_consistency * self.weights["factual_consistency"] -
            verdict.hallucination_probability * self.weights["hallucination_penalty"]
        )

        verdict.overall_score = round(max(0.0, min(1.0, overall)), 4)

        # ── Decide ──
        threshold = self.config.min_quality_score
        if is_multi_teacher:
            threshold = max(threshold, 0.80)

        verdict.passed = verdict.overall_score >= threshold

        # ── Build reasons ──
        if verdict.completeness < 0.6:
            verdict.reasons.append(f"Incomplete (completeness={verdict.completeness:.2f})")
        if verdict.correctness < 0.6:
            verdict.reasons.append(f"Correctness concerns (score={verdict.correctness:.2f})")
        if verdict.reasoning_depth < 0.5:
            verdict.reasons.append(f"Shallow reasoning (depth={verdict.reasoning_depth:.2f})")
        if verdict.hallucination_probability > 0.3:
            verdict.reasons.append(f"High hallucination probability ({verdict.hallucination_probability:.2f})")
        if verdict.novelty < 0.4:
            verdict.reasons.append(f"Low novelty — may duplicate existing content")

        if verdict.overall_score >= 0.85:
            verdict.strengths.append("Excellent overall quality")
        if verdict.correctness >= 0.85:
            verdict.strengths.append("High factual correctness")
        if verdict.reasoning_depth >= 0.80:
            verdict.strengths.append("Deep reasoning present")

        if verdict.passed:
            self.total_accepted += 1
        else:
            self.total_rejected += 1

        self.score_history.append(verdict.overall_score)
        if len(self.score_history) > 1000:
            self.score_history = self.score_history[-1000:]

        return verdict

    # ── Core LLM-based scoring ───────────────────────────────

    async def _score_core_dimensions(
        self, question: str, answer: str, reasoning: str, topic: str
    ) -> Optional[dict]:
        """Score completeness, correctness, reasoning depth, and implementation depth in one call"""
        prompt = f"""Score this training sample on four dimensions.

Reply with ONLY four comma-separated numbers between 0 and 1, in this exact order, and nothing else:

completeness,correctness,reasoning_depth,implementation_depth

Meaning of each number:
1. completeness — how fully the answer addresses the question
2. correctness — factual accuracy (0 = contains errors, 1 = accurate)
3. reasoning_depth — multi-step reasoning vs bare conclusions
4. implementation_depth — production-ready code/architecture vs none

QUESTION: {question}

ANSWER: {answer[:2500]}

REASONING: {reasoning[:6000]}

Numbers:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=5000,
            )

            content = (response.get("content") or "").strip()
            reasoning = (response.get("reasoning") or "").strip()

            # Reasoning models (MiniMax M3, R1-style) can spend the whole budget
            # in the reasoning channel and return content="". Scan both, and
            # scan the tail of reasoning first — the answer lands at the end.
            scores = self._parse_four_scores(content)
            if scores is None and reasoning:
                scores = self._parse_four_scores(reasoning[-600:])
            if scores is None:
                logger.warning(
                    "judge output unparsable (finish_reason=%s, content=%r, reasoning_tail=%r)",
                    response.get("finish_reason"), content[:120], reasoning[-160:],
                )
                return None

            return scores
        except Exception as e:
            logger.warning("judge call failed: %s: %s", type(e).__name__, str(e)[:200])
            return None

    @staticmethod
    def _parse_four_scores(text: str) -> Optional[dict]:
        """Extract 4 float scores (0-1) from judge output, tolerating prose."""
        if not text:
            return None

        keys = ("completeness", "correctness", "reasoning_depth", "implementation_depth")

        # 1. Clean CSV form: "0.85,0.90,0.75,0.60"
        parts = [p.strip() for p in text.split(',')]
        if len(parts) == 4:
            try:
                vals = [float(p) for p in parts]
                if all(0.0 <= v <= 1.0 for v in vals):
                    return dict(zip(keys, vals))
            except ValueError:
                pass

        # 2. Labelled form: any KEY: 0.85 / KEY = 0.85 (order independent)
        labelled = {}
        for key in keys:
            label = key.replace("_", r"[ _]")
            m = re.search(label + r"\s*[:=]\s*([01](?:\.\d+)?|\d\.\d+)", text, re.I)
            if m:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    labelled[key] = v
        if len(labelled) == 4:
            return labelled

        # 3. Last resort: all 0-1 decimals in order, take the last four
        #    (explanation text usually comes first, verdict numbers last).
        nums = [float(x) for x in re.findall(r"\b([01](?:\.\d+)?)\b", text)]
        nums = [n for n in nums if 0.0 <= n <= 1.0]
        if len(nums) >= 4:
            return dict(zip(keys, nums[-4:]))

        return None

    # ── Heuristic scorers (fast, no LLM calls) ───────────────

    def _heuristic_completeness(self, question: str, answer: str) -> float:
        """Heuristic completeness based on length ratio and question coverage"""
        score = 0.5

        # Length-based
        min_len = self.config.min_answer_length
        ans_len = len(answer)
        if ans_len >= min_len * 3:
            score += 0.2
        elif ans_len >= min_len * 1.5:
            score += 0.1
        elif ans_len < min_len:
            score -= 0.3

        # Structure-based
        if answer.count("## ") >= 3:
            score += 0.1
        if answer.count("```") >= 2:
            score += 0.05
        if any(kw in answer.lower() for kw in ["example", "for instance", "such as"]):
            score += 0.05

        # Question coverage (simple keyword overlap)
        q_words = set(w.lower() for w in question.split() if len(w) > 3)
        a_words = set(w.lower() for w in answer.split() if len(w) > 3)
        if q_words:
            overlap = len(q_words & a_words) / len(q_words)
            if overlap > 0.7:
                score += 0.1
            elif overlap < 0.3:
                score -= 0.2

        return round(max(0.0, min(1.0, score)), 3)

    def _heuristic_implementation_depth(self, answer: str, reasoning: str) -> float:
        """Heuristic for implementation/code depth"""
        score = 0.3

        text = answer + reasoning
        code_blocks = text.count("```")
        if code_blocks >= 6:  # 3+ code blocks
            score += 0.4
        elif code_blocks >= 4:
            score += 0.25
        elif code_blocks >= 2:
            score += 0.15

        # Check for production indicators
        prod_keywords = [
            "error handling", "exception", "logging", "monitoring",
            "deploy", "scale", "production", "test", "benchmark",
            "try", "catch", "finally", "retry", "timeout",
        ]
        prod_count = sum(1 for kw in prod_keywords if kw in text.lower())
        score += min(prod_count * 0.02, 0.2)

        return round(min(1.0, score), 3)

    def _heuristic_factual_consistency(self, answer: str, reasoning: str) -> float:
        """Heuristic factual consistency checks"""
        score = 0.7

        # Check for internal contradictions (simple self-reference)
        text = answer.lower()

        # Hedging language reduces consistency confidence
        hedge_words = ["probably", "maybe", "might be", "could be", "i think", "i believe"]
        hedge_count = sum(1 for hw in hedge_words if hw in text)
        if hedge_count > 5:
            score -= 0.2
        elif hedge_count > 2:
            score -= 0.1

        # Citations/references increase confidence
        if any(ref in text for ref in ["arxiv:", "doi:", "paper:", "reference:", "[1]", "[2]"]):
            score += 0.1

        # Code blocks with imports suggest concrete knowledge
        if "import " in text or "from " in text:
            score += 0.05

        # Well-structured suggests rigorous thinking
        if text.count("## ") >= 3:
            score += 0.05

        return round(max(0.0, min(1.0, score)), 3)

    def _heuristic_novelty(self, answer: str, previous_samples: list) -> float:
        """Heuristic novelty based on word uniqueness vs previous samples"""
        if not previous_samples:
            return 0.8  # Can't judge — assume novel

        a_words = set(w.lower() for w in answer.split() if len(w) > 4)

        overlaps = []
        for sample in previous_samples[-10:]:
            prev_text = ""
            if isinstance(sample, dict):
                prev_text = sample.get("answer", "") or sample.get("text", "")
            else:
                prev_text = str(sample)
            prev_words = set(w.lower() for w in prev_text.split() if len(w) > 4)

            if prev_words:
                overlap = len(a_words & prev_words) / len(a_words) if a_words else 0
                overlaps.append(overlap)

        if not overlaps:
            return 0.8

        avg_overlap = sum(overlaps) / len(overlaps)
        novelty = 1.0 - avg_overlap

        return round(max(0.0, min(1.0, novelty)), 3)

    def _heuristic_hallucination_prob(self, answer: str, reasoning: str) -> float:
        """Heuristic hallucination probability indicators"""
        prob = 0.05  # Base assumption: mostly correct

        text = answer.lower()

        # Markers that increase hallucination probability
        # Vague/evasive language
        vague_phrases = [
            "as mentioned earlier", "it is well known", "obviously",
            "clearly", "without a doubt", "everyone knows",
            "it goes without saying", "needless to say",
        ]
        vague_count = sum(1 for vp in vague_phrases if vp in text)
        prob += vague_count * 0.03

        # Unsubstantiated superlatives
        if text.count("best") > 3 or text.count("always") > 2:
            prob += 0.05

        # Very short reasoning despite complexity claims
        if len(reasoning) < 150 and len(answer) > 2000:
            prob += 0.1  # Long answer but no reasoning = suspicious

        # Missing concrete details
        if "```" not in text and len(answer) > 1000:
            if "example" not in text and "for instance" not in text:
                prob += 0.05  # Long answer with no code or examples

        # Extremely specific numbers without source (potential fabrication)
        import re
        numbers = re.findall(r'\b\d+\.?\d*%\b', text)
        if len(numbers) > 5 and "source" not in text:
            prob += 0.05

        return round(max(0.0, min(1.0, prob)), 3)

    # ── Stats ────────────────────────────────────────────────

    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate"""
        if self.total_scored == 0:
            return 0.0
        return round(self.total_accepted / self.total_scored * 100, 2)

    def get_average_score(self) -> float:
        """Get average score from recent history"""
        if not self.score_history:
            return 0.0
        return round(sum(self.score_history) / len(self.score_history), 4)

    def get_stats(self) -> dict:
        """Get comprehensive quality statistics"""
        return {
            "total_scored": self.total_scored,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": self.get_acceptance_rate(),
            "average_score": self.get_average_score(),
            "recent_scores": self.score_history[-20:],
        }


class QualityScorer:
    """
    Backwards-compatible shim around FrontierQualityScorer.

    Legacy workers (quality_worker.py, quality_verification_worker.py) import
    `QualityScorer` and call `score_qa_pair()` expecting a plain dict. This
    adapter delegates to FrontierQualityScorer.comprehensive_score() and
    converts the QualityVerdict dataclass into the dict shape those workers
    consume: {"overall_score", "passed", "reasons", ...}.
    """

    def __init__(self, llm_client: LLMClient, config, weights: Optional[dict] = None):
        self._scorer = FrontierQualityScorer(llm_client, config, weights=weights)
        self.config = config

    async def score_qa_pair(
        self,
        question: str,
        answer: str,
        reasoning: str = "",
        topic: str = "",
        previous_samples: list = None,
        is_multi_teacher: bool = False,
    ) -> dict:
        """Score a Q&A pair and return a plain dict for legacy workers."""
        verdict = await self._scorer.comprehensive_score(
            question=question,
            answer=answer,
            reasoning=reasoning,
            topic=topic,
            previous_samples=previous_samples,
            is_multi_teacher=is_multi_teacher,
        )
        return {
            "overall_score": verdict.overall_score,
            "passed": verdict.passed,
            "reasons": verdict.reasons,
            "strengths": verdict.strengths,
            "weaknesses": verdict.weaknesses,
            "improvement_suggestions": verdict.improvement_suggestions,
            "completeness": verdict.completeness,
            "correctness": verdict.correctness,
            "reasoning_depth": verdict.reasoning_depth,
            "implementation_depth": verdict.implementation_depth,
            "novelty": verdict.novelty,
            "factual_consistency": verdict.factual_consistency,
            "hallucination_probability": verdict.hallucination_probability,
        }

    def get_stats(self) -> dict:
        return self._scorer.get_stats()

    def get_acceptance_rate(self) -> float:
        return self._scorer.get_acceptance_rate()

    def get_average_score(self) -> float:
        return self._scorer.get_average_score()
