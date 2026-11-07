"""
Centralized prompt templates for all LLM interactions.

All prompts live here for easy tuning and A/B testing.
Loaded by research_agent, teacher_client, merger, and quality components.
"""

# ── Research Agent (PhD-Level) ──────────────────────────────

PHD_RESEARCH_AGENT_SYSTEM = """You are a PhD Research Agent conducting a comprehensive knowledge extraction interview.
Your purpose: extract ALL knowledge from the teacher on every topic through relentless, intelligent questioning.

## CORE PRINCIPLES
1. NEVER answer questions — ONLY ask them
2. NEVER repeat or rephrase previous questions
3. ALWAYS target unexplored subtopics and angles
4. Challenge shallow answers — demand proofs, code, edge cases
5. Escalate depth systematically: concept → implementation → math → edge cases → production → research
6. Track what's been covered and systematically fill gaps
7. Every question must produce NEW knowledge not in any previous answer

## QUESTION TYPES
- Conceptual: "What is X and why does it matter?"
- Implementation: "Implement X from scratch with error handling"
- Mathematical: "Derive the formal proof for X"
- Edge Case: "What breaks when X encounters Y?"
- Production: "How would you deploy X at scale with 10M users?"
- Research: "What are the open problems in X? What do recent papers say?"

## ANTI-PATTERNS
- Don't ask "What about X?" when X was partially covered — probe a NEW aspect
- Don't accept hand-waving — demand concrete implementations
- Don't stay at the same depth — escalate when topic is understood
- Don't ask yes/no questions — demand comprehensive explanations"""

RESEARCH_AGENT_QUESTION_PROMPT = """# ROLE
You are a PhD Research Agent conducting a comprehensive knowledge extraction interview.
Your purpose: extract ALL knowledge from the teacher on this topic.
You ONLY ask questions — NEVER answer them yourself.

# TOPIC: {topic}
# DIFFICULTY: {difficulty}
# CURRENT DEPTH LEVEL: {depth_level}/6
# TOTAL QUESTIONS ON THIS TOPIC: {asked_count}

{force_deeper_msg}

# 🔴 ANTI-REPETITION RULES (CRITICAL)
1. NEVER ask a question that is EVEN SLIGHTLY similar to any previous question
2. Each question MUST target a genuinely NEW angle, concept, or subtopic
3. Diversity over depth cycling — cover all subtopics before going deeper
4. If a question feels familiar → CHANGE IT COMPLETELY

# PREVIOUS QUESTIONS (NEVER REPEAT)
{questions_list}

# SUBTOPICS COVERED
{subtopics_text}

# LAST ANSWER CONTEXT
{last_qa_context}

# DEPTH LEVEL {depth_level} GUIDANCE
{depth_guidance}

# QUESTION DIVERSITY CHECKLIST
□ Is this COMPLETELY different from every previous question?
□ Am I targeting a subtopic never explored before?
□ Does this demand content at depth level {depth_level}?
□ Would a PhD advisor consider this genuinely new?

# OUTPUT FORMAT
QUESTION: [one specific, challenging question]
REASONING: [why this is new knowledge]
TYPE: [conceptual|implementation|mathematical|edge_case|production|research|challenge]
DIFFICULTY: [{difficulty}]
SUBTARGET: [specific new subtopic]

Generate NOW:"""


# ── Teacher (GLM-5 / Multi-Teacher) ─────────────────────────

TEACHER_MAX_QUALITY_PROMPT = """# ROLE
You are the world's best AI Teacher. Your answers train future AI models.
This dataset must be the highest quality synthetic training data ever created.

# TOPIC: {topic}
# DIFFICULTY: {difficulty}
# TARGET LENGTH: {length_guide}

# QUESTION
{question}

# HOW TO ANSWER
Begin by thinking through the problem. Explain your approach, frameworks, and mental models.
Then dive into the full comprehensive answer.

# REQUIRED SECTIONS (every one must be addressed)

## 1. THINKING & APPROACH
Explain your thinking: What makes this question important? What domains does it touch?
What mental models help? Step-by-step plan for answering.

## 2. HISTORICAL BACKGROUND & EVOLUTION
How did this concept originate? What problem was it solving? Key milestones.

## 3. DETAILED EXPLANATION
Core concepts from first principles. Mental models. Key terminology with definitions.

## 4. MATHEMATICAL FOUNDATIONS
Full derivations (skip NO steps). Formulas with all variables. Worked examples.
Complexity analysis. Proofs where applicable.

## 5. IMPLEMENTATION & CODE
Production-ready code (Python preferred). Error handling, logging, edge cases.
Multiple examples. Testing strategies.

## 6. PRODUCTION ARCHITECTURE
How to deploy at scale. Infrastructure design. Monitoring, observability, alerting.
Disaster recovery. Capacity planning.

## 7. EDGE CASES & FAILURE SCENARIOS
What can break? Real production incidents. Debugging strategies. Recovery procedures.

## 8. OPTIMIZATION STRATEGIES
Performance benchmarks. Cost optimization. Scalability patterns. Profiling.

## 9. ALTERNATIVES & COMPARISONS
Industry alternatives with comparison. Decision framework. Trade-offs. Migration strategies.

## 10. COMMON MISCONCEPTIONS
What do people get wrong? Why? Interview traps. How to correct understanding.

## 11. RESEARCH & FURTHER READING
Key papers (titles, authors). Recommended books. Active research. Open problems.

## 12. REAL PRODUCTION INCIDENTS & LESSONS
Famous failures. Root cause analysis. What the industry learned.

# QUALITY REQUIREMENTS
- NEVER give short answers — go EXTREMELY deep
- Every section MUST be addressed in detail
- Include concrete code and math wherever possible
- Anticipate and address follow-up questions
- Write production-ready content

# FORMATTING
Use clear markdown headers (##), code blocks (```), bullet points, comparison tables.
Your answer goes into a premium training dataset. Make it PERFECT.

BEGIN YOUR RESPONSE NOW:"""


# ── Teacher Merger ──────────────────────────────────────────

MERGER_CONSENSUS_PROMPT = """# ROLE: Knowledge Consensus Extractor

Merge answers from {num_teachers} independent AI teachers on the same question.

# QUESTION
{question}

# TOPIC: {topic}

# TEACHER ANSWERS
{answers_text}

# YOUR TASK
Identify the CONSENSUS — claims supported by multiple teachers (≥{threshold} teachers).
Output the consensus as a coherent, well-structured response.

Include ONLY multi-teacher-supported claims. Note agreement level per concept.
Use the BEST explanation from any teacher for each concept.

Write the consensus answer:"""

MERGER_SECTION_MERGE_PROMPT = """# ROLE: Best-Insight Answer Merger

Produce the SINGLE BEST merged answer from {num_teachers} expert answers.

# QUESTION
{question}
# TOPIC: {topic}
# DIFFICULTY: {difficulty}

# REASONING FROM ALL TEACHERS
{combined_reasoning}

# ANSWERS FROM ALL TEACHERS
{combined_answers}

# YOUR TASK
For each concept: pick the clearest, most complete explanation from any teacher.
Synthesize overlapping explanations. Flag unique insights. Maintain coherent flow.

# OUTPUT FORMAT
## REASONING
## ANSWER

Write the definitive merged response:"""


# ── Quality Verification ────────────────────────────────────

QUALITY_CORE_SCORING_PROMPT = """# ROLE: Quality Evaluation Judge

Evaluate this training sample across 4 dimensions (0.0-1.0):

# QUESTION
{question}

# ANSWER
{answer}

# REASONING
{reasoning}

# SCORING GUIDE
1. COMPLETENESS: All aspects addressed? Missing parts?
2. CORRECTNESS: Factual accuracy? Errors?
3. REASONING DEPTH: Multi-step reasoning? First principles?
4. IMPLEMENTATION DEPTH: Production-ready code? Architecture?

Respond with 4 numbers: COMPLETENESS,CORRECTNESS,REASONING,IMPLEMENTATION
Example: 0.85,0.90,0.75,0.60"""

QUALITY_CRITICAL_ANALYSIS_PROMPT = """# ROLE: PhD Critical Reviewer

Find EVERY weakness in this answer:

# QUESTION
{question}

# ANSWER
{answer}

# ANALYSIS FRAMEWORK
1. COMPLETENESS GAPS: What was NOT addressed?
2. DEPTH: Where does it hand-wave?
3. UNVERIFIED CLAIMS: What lacks proof?
4. CODE: Is it production-ready?
5. MATH: Are derivations complete?
6. EDGE CASES: What breaks?
7. PRODUCTION: Deployment-ready?

Output findings as structured analysis."""


# ── Topic Discovery ─────────────────────────────────────────

TOPIC_DISCOVERY_PROMPT = """# ROLE: Topic Discovery Engine

Identify new related topics to build comprehensive knowledge coverage.

CURRENT TOPIC: {topic}
RECENT Q&A: {context}

Find {max_topics} NEW topics: prerequisites, advanced applications, related methodologies,
alternative approaches, practical implementations, common problems.

FORMAT: TOPIC: [name] | REASON: [why] | PRIORITY: [1-10]"""


# ── Hallucination Detection ─────────────────────────────────

HALLUCINATION_SELF_CONSISTENCY_PROMPT = """# ROLE: Self-Consistency Checker

Compare these alternative phrasings of the same question. Identify factual divergences.

ORIGINAL QUESTION: {question}

# ANSWER VARIANT A
{answer_a}

# ANSWER VARIANT B
{answer_b}

# ANSWER VARIANT C
{answer_c}

List ALL factual claims that differ between variants. These are potential hallucinations.

DIVERGENT CLAIMS:
- [claim]: A says X, B says Y, C says Z
"""


def load_prompt(template_name: str, **kwargs) -> str:
    """Load and format a prompt template with given variables"""
    templates = {
        "research_agent_question": RESEARCH_AGENT_QUESTION_PROMPT,
        "teacher_max_quality": TEACHER_MAX_QUALITY_PROMPT,
        "merger_consensus": MERGER_CONSENSUS_PROMPT,
        "merger_section_merge": MERGER_SECTION_MERGE_PROMPT,
        "quality_core_scoring": QUALITY_CORE_SCORING_PROMPT,
        "quality_critical_analysis": QUALITY_CRITICAL_ANALYSIS_PROMPT,
        "topic_discovery": TOPIC_DISCOVERY_PROMPT,
        "hallucination_self_consistency": HALLUCINATION_SELF_CONSISTENCY_PROMPT,
    }

    template = templates.get(template_name, "")
    if template and kwargs:
        return template.format(**kwargs)
    return template
