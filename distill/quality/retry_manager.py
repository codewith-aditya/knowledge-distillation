"""
Retry Manager — Automatic retry orchestration for low-quality answers.

When quality < threshold:
1. Re-query all teachers with escalated prompt ("please be more detailed")
2. Track retry count per question
3. After max retries → flag LOW_QUALITY instead of discarding
4. Increment temperature each retry for diversity
5. Back off between retries

Never discards data — saves everything, flags low-quality samples.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RetryStrategy(Enum):
    """Available retry strategies"""
    REASK_SAME = "reask_same"             # Ask again with same prompt (temperature change only)
    ESCALATE_PROMPT = "escalate_prompt"   # Add "be more detailed" instructions
    SPLIT_QUESTION = "split_question"     # Break into sub-questions
    DIFFERENT_ANGLE = "different_angle"   # Rephrase question from different angle
    FORCE_STRUCTURE = "force_structure"   # Demand specific sections


@dataclass
class RetryRecord:
    """Record of a single retry attempt"""
    attempt: int
    strategy: RetryStrategy
    timestamp: str
    previous_score: float
    new_score: float = 0.0
    improvement: float = 0.0
    accepted: bool = False
    error: str = ""


@dataclass
class RetryResult:
    """Final result after all retries"""
    question: str
    topic: str
    total_attempts: int
    accepted: bool
    best_score: float
    best_answer: str = ""
    best_reasoning: str = ""
    retry_history: list[RetryRecord] = field(default_factory=list)
    final_disposition: str = ""  # "ACCEPTED" | "LOW_QUALITY" | "MAX_RETRIES_EXCEEDED"
    low_quality_flag: bool = False


class RetryManager:
    """
    Orchestrates automatic retry when quality is below threshold.

    Strategy rotation:
    - Attempt 1: ESCALATE_PROMPT (ask for more detail)
    - Attempt 2: FORCE_STRUCTURE (demand specific sections)
    - Attempt 3: DIFFERENT_ANGLE (rephrase from new angle)
    - Attempt 4: REASK_SAME with higher temperature
    - Attempt 5: SPLIT_QUESTION (break into sub-questions)
    """

    STRATEGY_SEQUENCE = [
        RetryStrategy.ESCALATE_PROMPT,
        RetryStrategy.FORCE_STRUCTURE,
        RetryStrategy.DIFFERENT_ANGLE,
        RetryStrategy.REASK_SAME,
        RetryStrategy.SPLIT_QUESTION,
    ]

    def __init__(
        self,
        max_retries: int = 5,
        temperature_increment: float = 0.1,
        base_temperature: float = 0.3,
        min_quality_threshold: float = 0.80,
    ):
        self.max_retries = max_retries
        self.temperature_increment = temperature_increment
        self.base_temperature = base_temperature
        self.min_quality_threshold = min_quality_threshold

        # Statistics
        self.total_retries_triggered = 0
        self.total_retries_performed = 0
        self.total_accepted_after_retry = 0
        self.total_low_quality_saved = 0
        self.retry_success_by_strategy: dict[str, int] = {s.value: 0 for s in RetryStrategy}

    def get_strategy(self, attempt: int) -> RetryStrategy:
        """Get retry strategy for a given attempt number (1-based)"""
        idx = min(attempt - 1, len(self.STRATEGY_SEQUENCE) - 1)
        return self.STRATEGY_SEQUENCE[idx]

    def get_temperature(self, attempt: int) -> float:
        """Get temperature for a given attempt (increases with each retry)"""
        return min(
            self.base_temperature + (attempt - 1) * self.temperature_increment,
            1.0,
        )

    def escalate_prompt(self, original_prompt: str, attempt: int) -> str:
        """
        Escalate the prompt for retry.

        Additions depend on strategy:
        - ESCALATE_PROMPT: Add "please be more detailed" with specific requirements
        - FORCE_STRUCTURE: Demand specific sections
        - DIFFERENT_ANGLE: Rephrase the question
        - REASK_SAME: Same prompt, temperature handles variation
        - SPLIT_QUESTION: Break into focused sub-questions
        """
        strategy = self.get_strategy(attempt)

        if strategy == RetryStrategy.ESCALATE_PROMPT:
            escalation = f"""

# ⚠️ QUALITY ESCALATION (Retry #{attempt})
Your previous answer was below the quality threshold. This time:
- Go DEEPER — every concept needs thorough explanation
- Include MORE code examples with error handling
- Add mathematical derivations where applicable
- Discuss edge cases and failure modes
- Your answer will be used to train future AI models — make it the BEST possible

{original_prompt}"""

        elif strategy == RetryStrategy.FORCE_STRUCTURE:
            escalation = f"""

# ⚠️ STRUCTURED RESPONSE REQUIRED (Retry #{attempt})
Your previous answer lacked structure. You MUST include ALL of these sections:

## 1. Core Explanation (from first principles)
## 2. Mathematical Foundation (with derivations)
## 3. Implementation (production-ready code)
## 4. Edge Cases & Failure Modes
## 5. Production Considerations
## 6. Common Misconceptions
## 7. References & Further Reading

{original_prompt}"""

        elif strategy == RetryStrategy.DIFFERENT_ANGLE:
            # Extract the question and reframe it
            escalation = f"""

# ⚠️ ALTERNATIVE ANGLE REQUIRED (Retry #{attempt})
Approach this from a COMPLETELY DIFFERENT perspective than your previous attempt.
Focus on aspects you may have overlooked — different frameworks, alternative implementations,
contrasting viewpoints, or deeper theoretical foundations.

{original_prompt}"""

        elif strategy == RetryStrategy.SPLIT_QUESTION:
            escalation = f"""

# ⚠️ FOCUSED SUB-QUESTION (Retry #{attempt})
Your previous answer was too broad. Focus SPECIFICALLY on the most important sub-topic.
Go extremely deep on this one aspect rather than covering everything shallowly.

{original_prompt}"""

        else:  # REASK_SAME
            escalation = f"""

# ⚠️ RETRY WITH IMPROVED QUALITY (Retry #{attempt})
Please provide a more thorough, detailed, and well-structured answer this time.
Think step-by-step. Include examples, code, and math where relevant.

{original_prompt}"""

        return escalation

    def should_retry(self, score: float, attempt: int) -> bool:
        """Determine if another retry should be attempted"""
        if score >= self.min_quality_threshold:
            return False
        if attempt >= self.max_retries:
            return False
        return True

    def create_result(
        self,
        question: str,
        topic: str,
        attempts: list[RetryRecord],
        best_answer: str = "",
        best_reasoning: str = "",
    ) -> RetryResult:
        """Create final result from retry history"""
        if not attempts:
            return RetryResult(
                question=question,
                topic=topic,
                total_attempts=0,
                accepted=False,
                best_score=0.0,
                final_disposition="NO_ATTEMPTS",
            )

        best = max(attempts, key=lambda a: a.new_score)
        accepted = best.accepted

        if accepted:
            disposition = "ACCEPTED"
            low_quality = False
            self.total_accepted_after_retry += 1
        elif len(attempts) >= self.max_retries:
            disposition = "MAX_RETRIES_EXCEEDED"
            low_quality = True
            self.total_low_quality_saved += 1
        else:
            disposition = "LOW_QUALITY"
            low_quality = True
            self.total_low_quality_saved += 1

        return RetryResult(
            question=question,
            topic=topic,
            total_attempts=len(attempts),
            accepted=accepted,
            best_score=best.new_score,
            best_answer=best_answer or "",
            best_reasoning=best_reasoning or "",
            retry_history=attempts,
            final_disposition=disposition,
            low_quality_flag=low_quality,
        )

    def record_retry_success(self, strategy: RetryStrategy):
        """Record which strategy led to success"""
        self.retry_success_by_strategy[strategy.value] += 1

    def get_most_effective_strategy(self) -> str:
        """Get the retry strategy with the most successes"""
        if not self.retry_success_by_strategy:
            return "unknown"
        return max(self.retry_success_by_strategy, key=self.retry_success_by_strategy.get)

    def get_stats(self) -> dict:
        """Get retry manager statistics"""
        return {
            "total_retries_triggered": self.total_retries_triggered,
            "total_retries_performed": self.total_retries_performed,
            "total_accepted_after_retry": self.total_accepted_after_retry,
            "total_low_quality_saved": self.total_low_quality_saved,
            "retry_success_rate": round(
                self.total_accepted_after_retry / self.total_retries_triggered * 100, 2
            ) if self.total_retries_triggered > 0 else 0.0,
            "best_strategy": self.get_most_effective_strategy(),
            "strategy_successes": self.retry_success_by_strategy,
        }
