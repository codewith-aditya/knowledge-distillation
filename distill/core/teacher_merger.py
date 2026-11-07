"""
Teacher Merger — the quality linchpin of the multi-teacher pipeline.

Takes independent answers from all teachers and produces one premium merged response.

Strategies (in priority order):
1. Consensus Extraction: Claims made by ≥threshold teachers (most reliable)
2. Complement Detection: Unique insights not present in other answers
3. Contradiction Detection: Flag conflicting claims between teachers
4. Best-Insight Selection: Per-section quality scoring, take the best
5. Enrichment: Collect all examples, code, references, production notes

The merged answer is richer than any single teacher's output.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from distill.llm.client import LLMClient
from distill.core.teacher_registry import TeacherRegistry
from distill.core.multi_teacher_orchestrator import TeacherResponse, MultiTeacherResult


@dataclass
class Contradiction:
    """A contradiction between two or more teacher answers"""
    claim_a: str
    claim_b: str
    teacher_a: str
    teacher_b: str
    topic: str
    resolution: str = ""          # How the contradiction was resolved (or "" if unresolved)


@dataclass
class TeacherContribution:
    """Track what each teacher contributed to the merged answer"""
    teacher_key: str
    sections_contributed: list[str] = field(default_factory=list)
    unique_insights: list[str] = field(default_factory=list)
    code_blocks: int = 0


@dataclass
class MergedAnswer:
    """The premium merged answer combining insights from all teachers"""
    question: str
    topic: str
    difficulty: str

    # Core answer
    primary_answer: str           # Consensus + best-insight merged answer
    reasoning: str                # Merged reasoning chain from all teachers

    # Enrichments
    examples: list[str] = field(default_factory=list)
    production_notes: str = ""
    best_practices: str = ""
    failure_cases: str = ""
    optimization_notes: str = ""
    code_blocks: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    # Quality metadata
    contradictions: list[Contradiction] = field(default_factory=list)
    teacher_contributions: dict[str, TeacherContribution] = field(default_factory=dict)
    agreement_score: float = 0.0     # 0-1 inter-teacher agreement
    merged_quality_score: float = 0.0  # Self-assessed quality

    # Source data
    teachers_contributed: list[str] = field(default_factory=list)
    teachers_failed: list[str] = field(default_factory=list)
    total_input_tokens: int = 0


class TeacherMerger:
    """
    Compares and merges independent answers from multiple teachers
    into one premium-quality merged response.

    The merger uses an LLM (typically the research agent or a dedicated model)
    to perform the comparison and synthesis.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: TeacherRegistry,
        consensus_threshold: int = 2,
        min_teachers_required: int = 1,
        contradiction_threshold: float = 0.7,
    ):
        self.llm = llm_client
        self.registry = registry
        self.consensus_threshold = consensus_threshold
        self.min_teachers_required = min_teachers_required
        self.contradiction_threshold = contradiction_threshold

    async def merge(self, result: MultiTeacherResult) -> Optional[MergedAnswer]:
        """
        Merge all successful teacher responses into one premium answer.

        Args:
            result: MultiTeacherResult from the orchestrator

        Returns:
            MergedAnswer or None if insufficient teachers responded
        """
        if result.successful_count < self.min_teachers_required:
            return None

        successful_responses = {
            k: r for k, r in result.responses.items() if r.success
        }

        # Phase 1: Extract consensus across teachers
        consensus_text = await self._extract_consensus(
            result.question, successful_responses, result.topic
        )

        # Phase 2: Detect contradictions
        contradictions = await self._detect_contradictions(
            result.question, successful_responses, result.topic
        )

        # Phase 3: Merge all sections with best-insight selection
        merged_sections = await self._merge_sections(
            result.question, successful_responses, result.topic, result.difficulty
        )

        # Phase 4: Enrich with examples, code, references
        enrichments = await self._enrich(
            result.question, successful_responses, result.topic
        )

        # Phase 5: Compute agreement score
        agreement_score = self._compute_agreement(successful_responses)

        # Phase 6: Build teacher contribution tracking
        contributions = self._build_contributions(successful_responses)

        # Build the merged answer
        merged = MergedAnswer(
            question=result.question,
            topic=result.topic,
            difficulty=result.difficulty,
            primary_answer=merged_sections.get("answer", ""),
            reasoning=merged_sections.get("reasoning", ""),
            examples=enrichments.get("examples", []),
            production_notes=enrichments.get("production_notes", ""),
            best_practices=enrichments.get("best_practices", ""),
            failure_cases=enrichments.get("failure_cases", ""),
            optimization_notes=enrichments.get("optimization_notes", ""),
            code_blocks=enrichments.get("code_blocks", []),
            references=enrichments.get("references", []),
            contradictions=contradictions,
            teacher_contributions=contributions,
            agreement_score=agreement_score,
            teachers_contributed=list(successful_responses.keys()),
            teachers_failed=[
                k for k, r in result.responses.items() if not r.success
            ],
            total_input_tokens=sum(
                r.tokens for r in result.responses.values()
            ),
        )

        return merged

    async def _extract_consensus(
        self,
        question: str,
        responses: dict[str, TeacherResponse],
        topic: str,
    ) -> str:
        """Extract claims where ≥consensus_threshold teachers agree"""
        if len(responses) < 2:
            # With one teacher, consensus = that teacher's answer
            only = list(responses.values())[0]
            return only.answer

        # Build comparison prompt
        answers_text = ""
        for i, (key, r) in enumerate(responses.items(), 1):
            answers_text += f"\n=== TEACHER {i}: {r.teacher_name} ===\n{r.answer[:3000]}\n"

        prompt = f"""# ROLE: Knowledge Consensus Extractor

You are merging answers from {len(responses)} independent AI teachers on the same question.

# QUESTION
{question}

# TOPIC: {topic}

# TEACHER ANSWERS
{answers_text}

# YOUR TASK
Identify the CONSENSUS — claims, facts, and explanations that are supported by multiple teachers.
With {len(responses)} teachers, a claim needs support from at least {self.consensus_threshold} teachers to be consensus.

Output the consensus answer as a coherent, well-structured response that:
1. Includes only claims with multi-teacher support
2. Is organized with clear sections
3. Uses the best explanation from any teacher for each concept
4. Notes the level of agreement (strong consensus: all teachers vs. moderate: {self.consensus_threshold}+)
5. Is comprehensive — don't leave out consensus content

# OUTPUT
Write the consensus answer directly (no meta-commentary):"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=12000,
            )
            return response["content"].strip()
        except Exception:
            # Fallback: use longest answer
            longest = max(responses.values(), key=lambda r: len(r.answer))
            return longest.answer

    async def _detect_contradictions(
        self,
        question: str,
        responses: dict[str, TeacherResponse],
        topic: str,
    ) -> list[Contradiction]:
        """Detect conflicting claims between teachers"""
        if len(responses) < 2:
            return []

        answers_text = ""
        for i, (key, r) in enumerate(responses.items(), 1):
            answers_text += f"\n=== TEACHER {i}: {r.teacher_name} ===\n{r.answer[:2000]}\n"

        prompt = f"""# ROLE: Contradiction Detector

Compare these {len(responses)} teacher answers for the same question. Find any direct contradictions.

# QUESTION
{question}

# TOPIC: {topic}

# ANSWERS
{answers_text}

# YOUR TASK
Identify any factual contradictions between teachers. A contradiction means:
- Teacher A claims X but Teacher B claims the opposite
- Numerical values differ by more than an order of magnitude
- Mutually exclusive design recommendations
- One teacher says "X is best" and another says "X should be avoided"

For each contradiction, output EXACTLY:
CONTRADICTION: <claim_a> ||| <claim_b> ||| <teacher_a> ||| <teacher_b> ||| <resolution>

If no contradictions found, output: NO_CONTRADICTIONS

Only flag genuine factual conflicts — NOT differences in depth, style, or emphasis."""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )

            content = response["content"].strip()
            contradictions = []

            for line in content.split('\n'):
                line = line.strip()
                if line.startswith("CONTRADICTION:") and "|||" in line:
                    parts = line.replace("CONTRADICTION:", "").strip().split("|||")
                    if len(parts) >= 4:
                        contradictions.append(Contradiction(
                            claim_a=parts[0].strip(),
                            claim_b=parts[1].strip(),
                            teacher_a=parts[2].strip(),
                            teacher_b=parts[3].strip(),
                            resolution=parts[4].strip() if len(parts) > 4 else "",
                            topic=topic,
                        ))

            return contradictions

        except Exception:
            return []

    async def _merge_sections(
        self,
        question: str,
        responses: dict[str, TeacherResponse],
        topic: str,
        difficulty: str,
    ) -> dict:
        """Merge all sections with best-insight selection per section"""
        # Combine reasoning from all teachers
        all_reasoning_parts = []
        all_answer_parts = []

        for key, r in responses.items():
            if r.reasoning:
                all_reasoning_parts.append(f"From {r.teacher_name}:\n{r.reasoning[:1500]}")
            if r.answer:
                all_answer_parts.append(f"From {r.teacher_name}:\n{r.answer[:3000]}")

        combined_reasoning = "\n\n---\n\n".join(all_reasoning_parts)
        combined_answers = "\n\n---\n\n".join(all_answer_parts)

        prompt = f"""# ROLE: Best-Insight Answer Merger

You have {len(responses)} independent expert answers to the same question. Your job is to produce the SINGLE BEST merged answer by selecting the best explanation for each sub-topic from any teacher.

# QUESTION
{question}

# TOPIC: {topic}
# DIFFICULTY: {difficulty}

# REASONING FROM ALL TEACHERS
{combined_reasoning}

# ANSWERS FROM ALL TEACHERS
{combined_answers}

# YOUR TASK
Create the definitive merged answer. For each concept/claim:
- Pick the clearest, most complete explanation from any teacher
- If multiple teachers explain something well, synthesize them
- If one teacher has a unique insight others missed, include it (flagged as unique)
- Maintain a coherent, well-structured flow
- Use the best code examples, math derivations, and practical advice

# OUTPUT FORMAT
## REASONING
[merged reasoning — how you approached this question, what frameworks apply]

## ANSWER
[the comprehensive merged answer with all sections]

Write the merged response now. This goes into a premium training dataset — make it the best possible answer:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.35,
                max_tokens=14000,
            )

            content = response["content"].strip()

            # Split into reasoning and answer
            reasoning = ""
            answer = content

            if "## REASONING" in content and "## ANSWER" in content:
                reasoning_start = content.index("## REASONING")
                answer_start = content.index("## ANSWER")
                reasoning = content[reasoning_start:answer_start].replace("## REASONING", "").strip()
                answer = content[answer_start:].replace("## ANSWER", "").strip()
            elif "## ANSWER" in content:
                answer_start = content.index("## ANSWER")
                reasoning = content[:answer_start].strip()
                answer = content[answer_start:].replace("## ANSWER", "").strip()

            return {"reasoning": reasoning, "answer": answer}

        except Exception:
            # Fallback: concatenate all answers
            return {
                "reasoning": "\n\n".join(r.reasoning for r in responses.values() if r.reasoning),
                "answer": "\n\n".join(r.answer for r in responses.values() if r.answer),
            }

    async def _enrich(
        self,
        question: str,
        responses: dict[str, TeacherResponse],
        topic: str,
    ) -> dict:
        """Extract all examples, code blocks, references from every answer"""
        examples = []
        code_blocks = []
        references = []
        production_notes_parts = []
        best_practices_parts = []
        failure_cases_parts = []
        optimization_parts = []

        for key, r in responses.items():
            answer = r.answer

            # Extract code blocks
            if "```" in answer:
                in_code = False
                current_block = []
                for line in answer.split('\n'):
                    if line.strip().startswith("```"):
                        if in_code:
                            code_blocks.append('\n'.join(current_block))
                            current_block = []
                            in_code = False
                        else:
                            in_code = True
                    elif in_code:
                        current_block.append(line)

            # Extract examples (lines with "example" or "Example")
            for line in answer.split('\n'):
                lower = line.lower().strip()
                if any(kw in lower for kw in ["example:", "for instance:", "e.g.,", "for example"]):
                    if len(line) > 20:
                        examples.append(line.strip())

            # Extract references
            for line in answer.split('\n'):
                lower = line.lower().strip()
                if any(kw in lower for kw in ["reference:", "see:", "paper:", "[1]", "arxiv:", "doi:"]):
                    if len(line) > 10:
                        references.append(line.strip())

            # Look for section headers indicating production content
            sections = answer.split('\n## ')
            for section in sections:
                lower = section.lower()
                if any(kw in lower for kw in ["production", "deployment", "architecture"]):
                    production_notes_parts.append(section[:1000])
                elif any(kw in lower for kw in ["best practice", "recommendation"]):
                    best_practices_parts.append(section[:1000])
                elif any(kw in lower for kw in ["failure", "edge case", "pitfall", "bug"]):
                    failure_cases_parts.append(section[:1000])
                elif any(kw in lower for kw in ["optimiz", "performance", "benchmark"]):
                    optimization_parts.append(section[:1000])

        return {
            "examples": list(dict.fromkeys(examples))[:20],  # deduplicate
            "code_blocks": list(dict.fromkeys(code_blocks)),
            "references": list(dict.fromkeys(references))[:30],
            "production_notes": "\n\n".join(production_notes_parts[:5]),
            "best_practices": "\n\n".join(best_practices_parts[:5]),
            "failure_cases": "\n\n".join(failure_cases_parts[:5]),
            "optimization_notes": "\n\n".join(optimization_parts[:5]),
        }

    def _compute_agreement(self, responses: dict[str, TeacherResponse]) -> float:
        """
        Estimate inter-teacher agreement using a simple heuristic:
        - Compute word overlap between each pair of answers
        - Higher overlap = higher agreement
        - Returns 0.0-1.0
        """
        if len(responses) < 2:
            return 0.0

        answers = [r.answer.lower() for r in responses.values()]

        def word_set(text: str) -> set:
            words = text.split()
            # Use every 5th word for efficiency on long answers
            return set(words[::5])

        scores = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                ws1 = word_set(answers[i])
                ws2 = word_set(answers[j])
                if not ws1 or not ws2:
                    scores.append(0.0)
                    continue
                intersection = len(ws1 & ws2)
                union = len(ws1 | ws2)
                scores.append(intersection / union if union > 0 else 0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _build_contributions(
        self,
        responses: dict[str, TeacherResponse],
    ) -> dict[str, TeacherContribution]:
        """Build per-teacher contribution tracking"""
        contributions = {}
        for key, r in responses.items():
            tc = TeacherContribution(teacher_key=key)
            tc.code_blocks = r.answer.count("```") // 2

            # Detect sections contributed
            for line in r.answer.split('\n'):
                if line.startswith("## "):
                    tc.sections_contributed.append(line.replace("## ", "").strip())

            contributions[key] = tc
        return contributions

    def get_stats(self) -> dict:
        """Get merger statistics"""
        return {
            "consensus_threshold": self.consensus_threshold,
            "min_teachers_required": self.min_teachers_required,
        }
