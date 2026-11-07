"""
Teacher prompt + response handling.

Reasoning and answer are captured as TWO separate artifacts:

  reasoning  ← the model's own thinking channel (reasoning / reasoning_content)
  answer     ← the clean, final answer in the content channel

The prompt tells the teacher to think first (privately) and then emit only the
polished answer. That keeps deliberation out of the training target while still
preserving the full chain of thought as its own field.

Earlier versions asked for a "## 1. THINKING & APPROACH" section *inside* the
answer, which meant the dataset's  thinking block held the merger's meta
commentary rather than real reasoning.
"""

from typing import Dict, Optional

from distill.llm.client import LLMClient


def build_teacher_prompt(
    question: str,
    topic: str,
    difficulty: str,
    context: Optional[str] = None,
) -> str:
    """Module-level prompt builder — same output as TeacherClient._build_teacher_prompt."""
    return TeacherClient._build_teacher_prompt(question, topic, difficulty, context)


class TeacherClient:
    """Teacher model — maximum quality answers with a separate reasoning channel."""

    def __init__(self, llm_client: LLMClient, config):
        self.llm = llm_client
        self.config = config

    async def get_answer(
        self,
        question: str,
        topic: str,
        difficulty: str,
        context: Optional[str] = None,
    ) -> Optional[Dict]:
        prompt = self._build_teacher_prompt(question, topic, difficulty, context)

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.teacher_temperature,
            )

            content = (response.get("content") or "").strip()
            reasoning = (response.get("reasoning") or "").strip()

            # If the model put everything in content (no separate channel),
            # fall back to splitting the text.
            if not reasoning and content:
                reasoning, content = self._split_inline_reasoning(content)

            return {
                "reasoning": reasoning,
                "answer": content,
                "tokens": response["usage"].total_tokens,
                "model": response["model"],
            }
        except Exception:
            return None

    # ── Inline fallback (only when the endpoint exposes no reasoning channel) ──

    def _split_inline_reasoning(self, content: str) -> tuple[str, str]:
        """Split a single-channel response into (reasoning, answer)."""
        # Explicit "## REASONING ... ## ANSWER" layout
        if "## REASONING" in content and "## ANSWER" in content:
            r_start = content.index("## REASONING")
            a_start = content.index("## ANSWER")
            if r_start < a_start:
                reasoning = content[r_start + len("## REASONING"):a_start].strip()
                answer = content[a_start + len("## ANSWER"):].strip()
                return reasoning, answer

        # First markdown header starts the answer, anything before is reasoning
        if "## " in content:
            idx = content.index("## ")
            head = content[:idx].strip()
            if len(head) >= 80:
                return head, content[idx:].strip()

        # Long responses: leading paragraphs are the reasoning
        paragraphs = content.split("\n\n")
        if len(paragraphs) > 4:
            split = max(2, len(paragraphs) // 5)
            return "\n\n".join(paragraphs[:split]).strip(), "\n\n".join(paragraphs[split:]).strip()

        return "", content

    @staticmethod
    def _build_teacher_prompt(
        question: str,
        topic: str,
        difficulty: str,
        context: Optional[str] = None,
    ) -> str:
        """Think-first teacher prompt.

        The model is instructed to deliberate privately, then emit only the final
        answer. Reasoning lands in the reasoning channel, content lands in the
        content channel — no meta-commentary in the training target.
        """
        return f"""# ROLE
You are an expert teacher. This answer becomes training data for a smaller model,
so it must be accurate, complete, and clean.

# TOPIC: {topic}
# DIFFICULTY: {difficulty}
# CONTEXT: {context or 'None'}

# QUESTION
{question}

# STEP 1 — THINK FIRST (private)
Before writing anything, think the problem through completely: what is really being
asked, which concepts and mental models apply, what the correct derivation is,
which edge cases matter, how you will structure the answer. Do all of this in your
own private thinking. Do NOT repeat this deliberation in your reply.

# STEP 2 — REPLY WITH ONLY THE FINAL ANSWER
Write the answer a strong engineer or student would actually want to read. Include,
in whatever depth the question genuinely needs:
- the core explanation, from first principles where it helps
- a worked example or two
- code, if the topic is implementation-related (runnable, with error handling)
- the edge cases and failure modes that matter in practice
- how it is used in the real world, at scale
- common misconceptions and how to avoid them
- references worth reading, when they are real and relevant

Do NOT pad. Do not add filler sections, restate the question, or narrate your
approach. Cover what the question needs and stop. Never invent citations,
benchmarks, or incident stories — omit them instead.

# FORMATTING
Use markdown: `##` headers for sections, fenced code blocks, tables where they help.
Write for a reader who wants the real answer, not a wall of headers.

BEGIN YOUR ANSWER NOW:"""

    def get_stats(self) -> Dict:
        return {
            "total_answers_provided": self.llm.request_count,
            "total_tokens": self.llm.total_usage.total_tokens,
            "total_input_tokens": self.llm.total_usage.input_tokens,
            "total_output_tokens": self.llm.total_usage.output_tokens,
        }
