"""
PhD-Level Research Agent (LLM-1) — Frontier-Scale Knowledge Extractor.

Powered by Qwen 397B. Sees FULL history. Never repeats. Challenges teachers.
Probes weak spots. Exhausts all knowledge.

Anti-Repetition Design (upgraded):
- Full question history per topic (not just last 2)
- Anti-repetition rules embedded in every prompt
- Sub-topic tree tracking for systematic coverage
- Critical analysis of every answer — detect weaknesses
- Challenge generation — probe gaps and shallow reasoning
- Multi-signal exhaustion detection
- Depth escalation: surface → implementation → math → proofs → edge cases → production
"""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CriticalAnalysis:
    """Result of critically analyzing a teacher's answer"""
    has_gaps: bool = False
    gaps: list[str] = field(default_factory=list)
    is_shallow: bool = False
    shallow_areas: list[str] = field(default_factory=list)
    has_unverified_claims: bool = False
    unverified_claims: list[str] = field(default_factory=list)
    missing_code: bool = False
    missing_math: bool = False
    missing_edge_cases: bool = False
    missing_production_notes: bool = False
    overall_quality: float = 0.0          # 0.0 - 1.0
    recommended_followups: list[str] = field(default_factory=list)


@dataclass
class ExhaustionVerdict:
    """Multi-signal exhaustion detection result"""
    is_exhausted: bool = False
    confidence: float = 0.0               # 0.0 - 1.0
    signals: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    recommended_action: str = ""          # "MOVE_TO_NEXT_TOPIC" | "GO_DEEPER" | "SWITCH_ANGLE"


class PhDResearchAgent:
    """
    PhD-Level Research Agent — behaves like a relentless academic researcher.

    Core Behaviors:
    1. Challenge: Never accept surface-level answers — demand proofs, code, edge cases
    2. Probe: Identify weak spots in each answer and generate targeted follow-ups
    3. Exhaust: Systematically cover every subtopic, angle, and depth level
    4. Never Repeat: Full question history prevents any repetition
    5. Escalate Depth: beginner → implementation → math → proofs → production → research

    Depth Levels (auto-escalated):
        1. Conceptual understanding (what + why)
        2. Implementation details (how + code)
        3. Mathematical foundations (proofs + derivations)
        4. Edge cases & failure modes (what breaks)
        5. Production architecture (scale + deploy)
        6. Research frontiers (open problems + recent papers)
    """

    # Depth escalation triggers
    DEPTH_ESCALATION_TRIGGERS = {
        1: {"min_questions": 5, "min_quality": 0.75},
        2: {"min_questions": 10, "min_quality": 0.78},
        3: {"min_questions": 18, "min_quality": 0.80},
        4: {"min_questions": 28, "min_quality": 0.82},
        5: {"min_questions": 40, "min_quality": 0.85},
        6: {"min_questions": 55, "min_quality": 0.87},
    }

    def __init__(self, llm_client, config):
        self.llm = llm_client
        self.config = config

        # Topic tracking
        self.topic_questions: dict[str, list[str]] = {}          # topic → [questions]
        self.topic_subtopics: dict[str, set[str]] = {}           # topic → {subtopics covered}
        self.topic_qa_pairs: dict[str, list[dict]] = {}          # topic → [{q, a, score}]
        self.topic_depth_levels: dict[str, int] = {}             # topic → current depth (1-6)
        self.topic_question_counts: dict[str, int] = {}          # topic → total questions asked
        self.topic_exhaustion_signals: dict[str, dict] = {}      # topic → exhaustion signals

        # Global stats
        self.total_questions_generated = 0

    # ── Public API ───────────────────────────────────────────

    async def generate_question(
        self,
        topic: str,
        difficulty: str,
        previous_qa_pairs: list[dict],
        context: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Generate the next question with full history awareness and anti-repetition.

        Returns:
            {
                "question": str | None,
                "difficulty": str,
                "reasoning": str,
                "question_type": str,
                "exhausted": bool,
                "force_deeper": bool,
                "depth_level": int,
                "target_subtopic": str,
            }
        """
        # Initialize topic tracking
        if topic not in self.topic_questions:
            self.topic_questions[topic] = []
            self.topic_subtopics[topic] = set()
            self.topic_qa_pairs[topic] = []
            self.topic_depth_levels[topic] = 1
            self.topic_question_counts[topic] = 0

        self.topic_question_counts[topic] = len(self.topic_questions[topic])

        # Check if we should escalate depth
        current_depth = self.topic_depth_levels[topic]
        self._check_depth_escalation(topic)

        # Check critical signals from previous answer
        force_deeper = False
        if previous_qa_pairs:
            last_qa = previous_qa_pairs[-1]
            last_reasoning = last_qa.get("reasoning", "")
            last_answer = last_qa.get("answer", "")

            # Quick heuristic checks (no LLM call needed)
            if len(last_reasoning) < 200:
                force_deeper = True
            if "direct answer" in last_reasoning.lower():
                force_deeper = True
            if len(last_answer) < 500:
                force_deeper = True
            if "```" not in last_answer and current_depth >= 2:
                force_deeper = True  # Should have code at implementation+ depth

        # Build the anti-repetition prompt with full history
        prompt = self._build_question_prompt(
            topic, difficulty, previous_qa_pairs, context, force_deeper
        )

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
            )

            content = response["content"].strip()

            # Check for exhaustion signal
            if "EXHAUSTED:" in content or "TOPIC_EXHAUSTED" in content:
                self.topic_exhaustion_signals[topic] = {
                    "exhausted_at": datetime.utcnow().isoformat(),
                    "total_questions": self.topic_question_counts[topic],
                    "depth_reached": self.topic_depth_levels[topic],
                }
                return {
                    "question": None,
                    "difficulty": difficulty,
                    "reasoning": content,
                    "exhausted": True,
                    "force_deeper": force_deeper,
                    "depth_level": self.topic_depth_levels[topic],
                }

            # Parse structured output
            question = ""
            reasoning = ""
            next_difficulty = difficulty
            q_type = ""
            target_subtopic = ""

            for line in content.split('\n'):
                line_stripped = line.strip()
                if line_stripped.startswith("QUESTION:"):
                    question = line_stripped.replace("QUESTION:", "").strip()
                elif line_stripped.startswith("REASONING:"):
                    reasoning = line_stripped.replace("REASONING:", "").strip()
                elif line_stripped.startswith("DIFFICULTY:"):
                    next_difficulty = line_stripped.replace("DIFFICULTY:", "").strip()
                elif line_stripped.startswith("TYPE:"):
                    q_type = line_stripped.replace("TYPE:", "").strip()
                elif line_stripped.startswith("SUBTARGET:"):
                    target_subtopic = line_stripped.replace("SUBTARGET:", "").strip()

            if not question:
                # Try to extract from unstructured output
                question = content.split('\n')[0] if content else ""
                if len(question) > 600:
                    question = question[:600]

            # Track for anti-repetition
            self.topic_questions[topic].append(question)
            self.total_questions_generated += 1

            if target_subtopic:
                self.topic_subtopics[topic].add(target_subtopic)

            return {
                "question": question[:2000],
                "difficulty": next_difficulty,
                "reasoning": reasoning or "Strategic question targeting uncovered ground",
                "question_type": q_type or "general",
                "exhausted": False,
                "force_deeper": force_deeper,
                "depth_level": self.topic_depth_levels[topic],
                "target_subtopic": target_subtopic,
            }

        except Exception:
            return None

    async def analyze_answer_critically(
        self, question: str, answer: str, topic: str
    ) -> CriticalAnalysis:
        """
        Perform a PhD-level critical analysis of the teacher's answer.

        Identifies:
        - Knowledge gaps (what was NOT covered)
        - Shallow reasoning (hand-waving, missing steps)
        - Unverified claims (assertions without proof)
        - Missing code or math
        - Missing edge cases
        - Missing production considerations
        """
        prompt = f"""# ROLE: PhD Critical Reviewer

You are a relentless academic reviewer. Your job is to find EVERY weakness in this answer.

# TOPIC: {topic}

# QUESTION
{question}

# ANSWER TO REVIEW
{answer[:4000]}

# CRITICAL ANALYSIS FRAMEWORK
Evaluate the answer across these dimensions. Be BRUTALLY honest:

## 1. COMPLETENESS GAPS
What aspects of the question were NOT addressed? What should have been included?

## 2. DEPTH ASSESSMENT
Is this surface-level or genuinely deep? Where does it hand-wave instead of explain?
Does it walk through reasoning step-by-step or jump to conclusions?

## 3. UNVERIFIED CLAIMS
Which claims are stated as fact without proof, derivation, or citation?
Which statements would fail peer review?

## 4. CODE & IMPLEMENTATION
Is there production-ready code? Error handling? Edge cases in code?
If code is present, is it complete and runnable?

## 5. MATHEMATICAL RIGOR
Are formulas derived or just stated? Are variables defined? Are there worked examples?
Is complexity analysis provided where relevant?

## 6. EDGE CASES & FAILURE MODES
What edge cases are ignored? What would break this in production?

## 7. PRODUCTION READINESS
Could you deploy this to production? What's missing for real-world use?

# OUTPUT FORMAT
GAPS: [list each gap, one per line starting with "- "]
SHALLOW_AREAS: [list areas with insufficient depth, one per line starting with "- "]
UNVERIFIED_CLAIMS: [list unproven claims, one per line starting with "- "]
MISSING: [comma-separated: CODE, MATH, EDGE_CASES, PRODUCTION, or NONE]
OVERALL_QUALITY: [0.0-1.0 score]
RECOMMENDED_FOLLOWUPS:
- [specific follow-up question 1]
- [specific follow-up question 2]
- [specific follow-up question 3]"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )

            content = response["content"]
            return self._parse_critical_analysis(content)

        except Exception:
            return CriticalAnalysis()

    async def generate_challenge(self, weak_point: str, topic: str) -> str:
        """
        Generate a targeted challenge question for a specific weak point
        in the teacher's answer.
        """
        prompt = f"""# ROLE: Academic Challenger

A teacher's answer had this weakness on topic "{topic}":

WEAK POINT: {weak_point}

Generate ONE sharp, specific challenge question that forces the teacher to:
- Provide rigorous proof/derivation (not hand-waving)
- Show production-ready code with error handling
- Address edge cases they ignored
- Back up claims with evidence

The challenge should be pointed — expose the exact gap. Make it impossible to answer with a generic response.

# OUTPUT
CHALLENGE: [one precise, demanding question]"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=300,
            )

            content = response["content"].strip()
            if "CHALLENGE:" in content:
                return content.split("CHALLENGE:", 1)[1].strip()
            return content[:300]

        except Exception:
            return f"Regarding {weak_point}: Please provide a rigorous treatment with complete derivations and concrete examples."

    async def probe_deeper(
        self,
        topic: str,
        depth_level: int,
        previous_responses: list[dict],
    ) -> Optional[str]:
        """
        Generate a deeper probing question that escalates to the next depth level.

        Depth levels:
        1 → 2: "Now implement this"
        2 → 3: "Derive the math formally"
        3 → 4: "What breaks? Edge cases?"
        4 → 5: "How to deploy at scale?"
        5 → 6: "What are the open research problems?"
        """
        depth_prompts = {
            2: "Move from conceptual to IMPLEMENTATION. Demand production-ready code.",
            3: "Move from implementation to MATHEMATICAL FOUNDATIONS. Demand formal proofs and derivations.",
            4: "Move from math to EDGE CASES & FAILURE MODES. Demand analysis of what breaks and why.",
            5: "Move from edge cases to PRODUCTION ARCHITECTURE. Demand deployment, scaling, monitoring.",
            6: "Move from production to RESEARCH FRONTIERS. Demand open problems, recent papers, future directions.",
        }

        focus = depth_prompts.get(depth_level, depth_prompts[2])

        prompt = f"""# ROLE: Depth Escalation Agent

Topic: {topic}
Current depth level: {depth_level - 1} → Target: {depth_level}

# ESCALATION DIRECTIVE
{focus}

# PREVIOUS ANSWERS CONTEXT
{self._format_previous_responses(previous_responses)}

# YOUR TASK
Generate ONE question that forces the teacher to go MUCH deeper.
The question must demand content at depth level {depth_level}.
It must be impossible to answer with a shallow response.

# OUTPUT
QUESTION: [one specific, deep, demanding question]
TYPE: depth_escalation_{depth_level}"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=500,
            )

            content = response["content"].strip()
            for line in content.split('\n'):
                if line.strip().startswith("QUESTION:"):
                    return line.strip().replace("QUESTION:", "").strip()[:600]
            return content[:600]

        except Exception:
            return None

    async def detect_exhaustion(
        self, topic: str, all_qa_history: list[dict]
    ) -> ExhaustionVerdict:
        """
        Multi-signal exhaustion detection.

        Signals:
        1. Repetition: Last 5 questions are semantically similar to previous ones
        2. Saturation: 50+ questions asked on this topic
        3. Coverage: All known subtopics explored
        4. Quality ceiling: Recent answers consistently score well (no gaps found)
        5. Diminishing returns: New questions yield <10% new information
        6. Depth ceiling: All 6 depth levels exhausted
        """
        signals = {}
        reasons = []

        q_count = self.topic_question_counts.get(topic, 0)

        # Signal 1: Minimum questions not reached
        min_q = self.config.exhaustion_min_questions
        if q_count < min_q:
            return ExhaustionVerdict(
                is_exhausted=False,
                confidence=0.0,
                signals={"min_questions_not_met": True},
                reasons=[f"Only {q_count}/{min_q} minimum questions asked"],
                recommended_action="CONTINUE",
            )

        # Signal 2: Question count saturation
        max_q = getattr(self.config, 'max_questions_per_topic', 200)
        if q_count >= max_q:
            signals["question_saturation"] = True
            reasons.append(f"Exceeded max questions ({q_count}/{max_q})")

        # Signal 3: Depth ceiling reached
        current_depth = self.topic_depth_levels.get(topic, 1)
        if current_depth >= 6:
            signals["depth_ceiling"] = True
            reasons.append(f"All 6 depth levels exhausted")

        # Signal 4: Quality ceiling (all recent answers score high)
        if all_qa_history:
            recent = all_qa_history[-10:]
            scores = [qa.get("quality_score", 0) for qa in recent if qa.get("quality_score")]
            if scores and all(s >= 0.85 for s in scores):
                signals["quality_ceiling"] = True
                reasons.append("Recent 10 answers all score ≥0.85")

        # Signal 5: Repetition detected via critical analysis of last answers
        if all_qa_history:
            last_qs = [qa.get("question", "") for qa in all_qa_history[-5:]]
            unique_words = set()
            for q in last_qs:
                unique_words.update(q.lower().split())
            if len(unique_words) < 30 and len(all_qa_history) > 20:
                signals["repetition_detected"] = True
                reasons.append("Recent questions show high word overlap — repeating territory")

        # Signal 6: Diminishing returns
        if q_count > 30:
            signals["diminishing_returns"] = True
            reasons.append(f"High question count ({q_count}) suggests diminishing returns")

        # Decision logic
        num_signals = sum(signals.values())
        is_exhausted = num_signals >= 2  # At least 2 signals needed

        confidence = min(num_signals / 3.0, 1.0)  # 3+ signals = high confidence

        # Recommended action
        if is_exhausted and confidence >= 0.7:
            action = "MOVE_TO_NEXT_TOPIC"
        elif signals.get("depth_ceiling") or signals.get("quality_ceiling"):
            action = "MOVE_TO_NEXT_TOPIC"
        elif signals.get("repetition_detected"):
            action = "SWITCH_ANGLE"
        else:
            action = "GO_DEEPER"

        return ExhaustionVerdict(
            is_exhausted=is_exhausted,
            confidence=confidence,
            signals=signals,
            reasons=reasons,
            recommended_action=action,
        )

    # ── Internal Methods ─────────────────────────────────────

    def _check_depth_escalation(self, topic: str):
        """Check if the topic should escalate to the next depth level"""
        current = self.topic_depth_levels.get(topic, 1)
        q_count = self.topic_question_counts.get(topic, 0)

        if current >= 6:
            return

        trigger = self.DEPTH_ESCALATION_TRIGGERS.get(current + 1, {})
        if q_count >= trigger.get("min_questions", 999):
            self.topic_depth_levels[topic] = current + 1

    def _build_question_prompt(
        self,
        topic: str,
        difficulty: str,
        previous_qa_pairs: list[dict],
        _context: Optional[str],
        force_deeper: bool = False,
    ) -> str:
        """Build the comprehensive anti-repetition question generation prompt"""

        asked_count = self.topic_question_counts.get(topic, 0)
        current_depth = self.topic_depth_levels.get(topic, 1)

        # Build full question history
        all_questions = self.topic_questions.get(topic, [])
        questions_list = ""
        if all_questions:
            recent = all_questions[-20:]  # Last 20 questions
            questions_list = "## PREVIOUSLY ASKED QUESTIONS (NEVER REPEAT THESE):\n"
            for i, q in enumerate(recent, 1):
                questions_list += f"  {i}. {q[:200]}\n"
        elif previous_qa_pairs:
            for i, qa in enumerate(previous_qa_pairs[-5:], 1):
                questions_list += f"  {i}. {qa.get('question', '')[:200]}\n"

        # Subtopics covered
        subtopics = self.topic_subtopics.get(topic, set())
        subtopics_text = ""
        if subtopics:
            subtopics_text = "## SUBTOPICS ALREADY COVERED:\n" + "\n".join(
                f"  - {st}" for st in sorted(subtopics)
            ) + "\n\n## UNEXPLORED AREAS TO TARGET:\nThink about what HAS NOT been asked.\n"

        # Depth level guidance
        depth_names = {
            1: "CONCEPTUAL — What, why, when, core concepts",
            2: "IMPLEMENTATION — Code, architecture, patterns, libraries",
            3: "MATHEMATICAL — Proofs, derivations, complexity, formal models",
            4: "EDGE CASES — Failures, bugs, corner cases, debugging",
            5: "PRODUCTION — Deployment, scaling, monitoring, disaster recovery",
            6: "RESEARCH — Open problems, recent papers, future directions, debates",
        }
        depth_guidance = depth_names.get(current_depth, depth_names[1])

        # Difficulty mapping
        diff_map = {
            "beginner": ("basic", "foundational understanding"),
            "intermediate": ("moderate", "implementation and patterns"),
            "expert": ("advanced", "deep technical mastery"),
            "advanced": ("expert-level", "cutting-edge and research"),
            "research": ("PhD-level", "frontier research problems"),
        }
        diff_guidance = diff_map.get(difficulty, ("moderate", "balanced"))

        force_msg = ""
        if force_deeper:
            force_msg = "\n⚠️  Previous answer was too shallow or lacked reasoning. DEMAND step-by-step deep reasoning, code, and proofs.\n"

        prompt = f"""# ROLE
You are a PhD Research Agent conducting a comprehensive knowledge extraction interview.
Your purpose: extract ALL knowledge from the teacher on this topic.
You ONLY ask questions — NEVER answer them yourself.

# TOPIC: {topic}
# DIFFICULTY: {difficulty} ({diff_guidance[0]} level — {diff_guidance[1]})
# CURRENT DEPTH LEVEL: {current_depth}/6 — {depth_guidance}
# TOTAL QUESTIONS ON THIS TOPIC: {asked_count}

{force_msg}

# 🔴 ANTI-REPETITION RULES (VIOLATION = SYSTEM FAILURE)
1. NEVER ask a question that is EVEN SLIGHTLY similar to any question in the history above
2. NEVER rephrase an existing question — ask about something COMPLETELY DIFFERENT
3. Each question MUST target a genuinely NEW angle, concept, or subtopic
4. Diversity over depth cycling — cover all subtopics before going deeper on any one
5. If a question feels familiar → CHANGE IT COMPLETELY
6. Target unexplored territory at the current depth level

# 🔴 ANTI-FABRICATION RULES (VIOLATION = POISONS THE DATASET)
1. Ask ONLY about real, established, citable concepts, papers, tools and techniques
2. NEVER invent a concept, acronym, framework or primitive (no "X-Coordinated Y
   Substrate" style made-up names) and never ask the teacher to "derive" something
   that does not exist
3. Every named technique you reference must be something a practitioner would find
   in documentation, a textbook, or a published paper
4. If you are not certain a term is real, ask about the underlying real topic instead

# 🔴 LENGTH RULE
The question must be self-contained and readable in one breath: 1-3 sentences,
maximum 120 words, ending with a question mark. Do not paste definitions,
citations or a list of every related paper into the question itself.

# ANTI-PATTERN EXAMPLES (DO NOT DO THIS)
- ❌ Already asked about "database indexes" → DO NOT ask about "index types" or "index optimization"
- ✅ Instead, ask about "query planning", "transaction isolation", "MVCC", "WAL", "buffer pool management"
- ❌ Already asked about "Python decorators" → DO NOT ask about "decorator patterns" or "decorator usage"
- ✅ Instead, ask about "metaclasses", "descriptors", "context managers", "coroutines internals"

{questions_list if questions_list else 'No questions yet. Start with a strong foundational question that opens up the topic.'}

{subtopics_text}

# RECENT ANSWER CONTEXT (for follow-up relevance and gap detection)
"""
        if previous_qa_pairs:
            last = previous_qa_pairs[-1]
            prompt += f"Last Question: {last.get('question', '')[:300]}\n"
            prompt += f"Last Answer (excerpt): {last.get('answer', '')[:500]}...\n"

            # Add last 3 Q&A summaries for context
            if len(previous_qa_pairs) > 1:
                prompt += "\n# RECENT Q&A HISTORY:\n"
                for qa in previous_qa_pairs[-3:]:
                    prompt += f"- Q: {qa.get('question', '')[:150]}\n"

        prompt += f"""
# DEPTH LEVEL {current_depth} QUESTION TYPES
At this depth level, your questions should focus on:
{depth_guidance}

# QUESTION DIVERSITY CHECKLIST (verify BEFORE output)
□ Is this COMPLETELY different from every question in the history?
□ Am I targeting a subtopic never explored before?
□ Would a PhD advisor consider this a genuinely new direction?
□ Does this demand content at depth level {current_depth}?
□ If this looks similar to any past question → CHANGE IT NOW

# OUTPUT FORMAT (strict)
QUESTION: [one specific, challenging question targeting UNEXPLORED territory]
REASONING: [why this question adds NEW knowledge — reference what's been covered and why this is different]
TYPE: [conceptual | implementation | mathematical | edge_case | production | research | challenge | probe]
DIFFICULTY: [{difficulty}]
SUBTARGET: [the specific subtopic being targeted — must be new]

Generate a COMPLETELY NEW question at depth level {current_depth} NOW:"""

        return prompt

    def _parse_critical_analysis(self, content: str) -> CriticalAnalysis:
        """Parse the critical analysis LLM output into structured form"""
        analysis = CriticalAnalysis()

        current_section = None
        for line in content.split('\n'):
            stripped = line.strip()

            if stripped.startswith("GAPS:"):
                current_section = "gaps"
                continue
            elif stripped.startswith("SHALLOW_AREAS:"):
                current_section = "shallow"
                continue
            elif stripped.startswith("UNVERIFIED_CLAIMS:"):
                current_section = "claims"
                continue
            elif stripped.startswith("MISSING:"):
                missing_str = stripped.replace("MISSING:", "").strip()
                analysis.missing_code = "CODE" in missing_str
                analysis.missing_math = "MATH" in missing_str
                analysis.missing_edge_cases = "EDGE_CASES" in missing_str
                analysis.missing_production_notes = "PRODUCTION" in missing_str
                current_section = None
                continue
            elif stripped.startswith("OVERALL_QUALITY:"):
                try:
                    analysis.overall_quality = float(stripped.replace("OVERALL_QUALITY:", "").strip())
                except ValueError:
                    analysis.overall_quality = 0.5
                current_section = None
                continue
            elif stripped.startswith("RECOMMENDED_FOLLOWUPS:"):
                current_section = "followups"
                continue

            if stripped.startswith("- ") and current_section:
                item = stripped[2:].strip()
                if current_section == "gaps":
                    analysis.gaps.append(item)
                elif current_section == "shallow":
                    analysis.shallow_areas.append(item)
                elif current_section == "claims":
                    analysis.unverified_claims.append(item)
                elif current_section == "followups":
                    analysis.recommended_followups.append(item)

        analysis.has_gaps = len(analysis.gaps) > 0
        analysis.is_shallow = len(analysis.shallow_areas) > 0
        analysis.has_unverified_claims = len(analysis.unverified_claims) > 0

        return analysis

    def _format_previous_responses(self, responses: list[dict]) -> str:
        """Format previous responses for context in prompts"""
        if not responses:
            return "No previous responses."

        parts = []
        for i, r in enumerate(responses[-5:], 1):
            q = r.get("question", "")[:200]
            a = r.get("answer", "")[:300]
            parts.append(f"Q{i}: {q}\nA{i}: {a}...")

        return "\n\n".join(parts)

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get comprehensive research agent statistics"""
        return {
            "total_topics": len(self.topic_questions),
            "total_questions": sum(len(qs) for qs in self.topic_questions.values()),
            "total_requests": self.llm.request_count,
            "topics_exhausted": len(self.topic_exhaustion_signals),
            "depth_levels": dict(self.topic_depth_levels),
            "subtopics_per_topic": {
                t: len(st) for t, st in self.topic_subtopics.items()
            },
            "questions_per_topic": dict(self.topic_question_counts),
        }

    def get_topic_progress(self, topic: str) -> dict:
        """Get progress for a specific topic"""
        return {
            "topic": topic,
            "questions_asked": self.topic_question_counts.get(topic, 0),
            "depth_level": self.topic_depth_levels.get(topic, 1),
            "subtopics_covered": sorted(self.topic_subtopics.get(topic, set())),
            "subtopics_count": len(self.topic_subtopics.get(topic, set())),
            "is_exhausted": topic in self.topic_exhaustion_signals,
        }


# Backwards-compatible alias — question_worker.py imports `ResearchAgent`.
ResearchAgent = PhDResearchAgent
