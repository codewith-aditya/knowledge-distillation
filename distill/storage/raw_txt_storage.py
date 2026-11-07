"""
RAW TXT storage with immediate disk sync (flush + fsync)
Premium format for frontier-scale training data.

Upgraded format includes:
- Multi-teacher metadata (teachers contributed, agreement scores)
- Quality dimension scores (7-dimension breakdown)
- Examples, code blocks, production notes
- Retry count and LOW_QUALITY flags
- Knowledge graph metadata

Optimized for crash safety and 24/7 autonomous operation.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from threading import Lock


class RawTxtStorage:
    """
    Premium RAW TXT dataset storage with immediate disk sync.

    Features:
    - Immediate flush + fsync after every sample
    - Automatic file rotation at configurable size
    - Crash-safe (only current sample may be lost)
    - Sequential file numbering (raw_001.txt, raw_002.txt, ...)
    - Rich multi-teacher sample format
    """

    def __init__(self, base_path: str, rotation_size_gb: float = 5.0):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.rotation_size_bytes = int(rotation_size_gb * 1024 * 1024 * 1024)
        self.current_file: Optional[Path] = None
        self.current_handle = None
        self.file_counter = 1
        self.write_lock = Lock()

        self.total_samples_written = 0
        self.total_bytes_written = 0

        self._initialize_current_file()

    def _initialize_current_file(self):
        """Find the latest file or create first one"""
        existing_files = sorted(self.base_path.glob("raw_*.txt"))

        if existing_files:
            last_file = existing_files[-1]
            file_num = int(last_file.stem.split('_')[1])

            if last_file.stat().st_size >= self.rotation_size_bytes:
                self.file_counter = file_num + 1
                self._create_new_file()
            else:
                self.file_counter = file_num
                self.current_file = last_file
                self.current_handle = open(self.current_file, 'a', encoding='utf-8', buffering=1)
        else:
            self._create_new_file()

    def _create_new_file(self):
        """Create a new RAW TXT file"""
        if self.current_handle:
            self.current_handle.close()

        filename = f"raw_{self.file_counter:03d}.txt"
        self.current_file = self.base_path / filename
        self.current_handle = open(self.current_file, 'a', encoding='utf-8', buffering=1)

    def _check_rotation(self):
        """Check if current file needs rotation"""
        if self.current_file and self.current_file.stat().st_size >= self.rotation_size_bytes:
            self.file_counter += 1
            self._create_new_file()

    # ── Premium multi-teacher write ──────────────────────────

    def write_merged_sample(
        self,
        topic: str,
        difficulty: str,
        question: str,
        reasoning: str,
        answer: str,
        model: str = "multi-teacher-merged",
        tokens: int = 0,
        teachers_contributed: list = None,
        teachers_failed: list = None,
        agreement_score: float = 0.0,
        quality_scores: dict = None,
        examples: list = None,
        code_blocks: list = None,
        production_notes: str = "",
        best_practices: str = "",
        failure_cases: str = "",
        optimization_notes: str = "",
        references: list = None,
        contradictions: list = None,
        coverage_score: float = 0.0,
        retry_count: int = 0,
        low_quality_flag: bool = False,
    ) -> bool:
        """
        Write a premium multi-teacher sample with all enrichments.

        Format:
        ======================================================================
        TOPIC: {topic}
        DIFFICULTY: {difficulty}
        COVERAGE_SCORE: {coverage}
        TEACHERS: {teachers_contributed}
        AGREEMENT_SCORE: {agreement}

        QUESTION:
        {question}

        <think>
        {reasoning}
        </think>

        ANSWER:
        {answer}

        EXAMPLES:
        [all unique examples]

        CODE:
        [all code blocks]

        PRODUCTION_NOTES:
        {production_notes}

        BEST_PRACTICES:
        {best_practices}

        FAILURE_CASES:
        {failure_cases}

        OPTIMIZATION_NOTES:
        {optimization_notes}

        REFERENCES:
        [references]

        CONTRADICTIONS_DETECTED:
        [contradictions]

        METADATA:
        MODEL: {model}
        TEACHERS_CONTRIBUTED: {teachers}
        TEACHERS_FAILED: {failed}
        AGREEMENT_SCORE: {score}
        QUALITY_SCORE: {overall}
        COMPLETENESS: {score}
        CORRECTNESS: {score}
        REASONING_DEPTH: {score}
        IMPLEMENTATION_DEPTH: {score}
        NOVELTY: {score}
        FACTUAL_CONSISTENCY: {score}
        HALLUCINATION_PROB: {score}
        TIMESTAMP: {iso}
        TOKENS: {tokens}
        RETRY_COUNT: {retries}
        LOW_QUALITY: {flag}
        ======================================================================
        """
        with self.write_lock:
            try:
                timestamp = datetime.utcnow().isoformat() + 'Z'
                qs = quality_scores or {}

                # Header
                sample_text = f"""{'=' * 70}
TOPIC: {topic}
DIFFICULTY: {difficulty}
COVERAGE_SCORE: {coverage_score:.3f}
TEACHERS: {', '.join(teachers_contributed or [])}
AGREEMENT_SCORE: {agreement_score:.3f}

QUESTION:
{question}

<think>
{reasoning}
</think>

ANSWER:
{answer}
"""

                # Enrichments
                if examples:
                    sample_text += "\nEXAMPLES:\n"
                    for ex in examples[:15]:
                        sample_text += f"- {ex}\n"

                if code_blocks:
                    sample_text += "\nCODE:\n"
                    for cb in code_blocks[:8]:
                        sample_text += f"```\n{cb[:3000]}\n```\n\n"

                if production_notes:
                    sample_text += f"\nPRODUCTION_NOTES:\n{production_notes[:3000]}\n"

                if best_practices:
                    sample_text += f"\nBEST_PRACTICES:\n{best_practices[:3000]}\n"

                if failure_cases:
                    sample_text += f"\nFAILURE_CASES:\n{failure_cases[:3000]}\n"

                if optimization_notes:
                    sample_text += f"\nOPTIMIZATION_NOTES:\n{optimization_notes[:3000]}\n"

                if references:
                    sample_text += "\nREFERENCES:\n"
                    for ref in references[:20]:
                        sample_text += f"- {ref}\n"

                if contradictions:
                    sample_text += "\nCONTRADICTIONS_DETECTED:\n"
                    for c in contradictions:
                        if hasattr(c, 'claim_a'):
                            sample_text += f"- [{c.teacher_a}] {c.claim_a[:200]} vs [{c.teacher_b}] {c.claim_b[:200]}\n"
                        elif isinstance(c, dict):
                            sample_text += f"- {c.get('teacher_a', '?')}: {c.get('claim_a', '')[:200]} vs {c.get('teacher_b', '?')}: {c.get('claim_b', '')[:200]}\n"

                # Metadata footer
                sample_text += f"""
METADATA:
MODEL: {model}
TEACHERS_CONTRIBUTED: {', '.join(teachers_contributed or [])}
TEACHERS_FAILED: {', '.join(teachers_failed or []) or 'none'}
AGREEMENT_SCORE: {agreement_score:.3f}
QUALITY_SCORE: {qs.get('overall', 0):.3f}
COMPLETENESS: {qs.get('completeness', 0):.3f}
CORRECTNESS: {qs.get('correctness', 0):.3f}
REASONING_DEPTH: {qs.get('reasoning_depth', 0):.3f}
IMPLEMENTATION_DEPTH: {qs.get('implementation_depth', 0):.3f}
NOVELTY: {qs.get('novelty', 0):.3f}
FACTUAL_CONSISTENCY: {qs.get('factual_consistency', 0):.3f}
HALLUCINATION_PROB: {qs.get('hallucination_probability', 0):.4f}
TIMESTAMP: {timestamp}
TOKENS: {tokens}
RETRY_COUNT: {retry_count}
LOW_QUALITY: {str(low_quality_flag).lower()}
{'=' * 70}

"""

                self.current_handle.write(sample_text)
                self.current_handle.flush()
                os.fsync(self.current_handle.fileno())

                sample_bytes = len(sample_text.encode('utf-8'))
                self.total_samples_written += 1
                self.total_bytes_written += sample_bytes

                self._check_rotation()
                return True

            except Exception:
                return False

    # ── Legacy single-teacher write (backward compat) ────────

    def write_sample(
        self,
        topic: str,
        difficulty: str,
        question: str,
        reasoning: str,
        answer: str,
        model: str,
        tokens: int,
    ) -> bool:
        """Write a single Q&A sample (legacy format, backward compatible)"""
        with self.write_lock:
            try:
                timestamp = datetime.utcnow().isoformat() + 'Z'

                sample_text = f"""{'=' * 50}
TOPIC: {topic}
DIFFICULTY: {difficulty}

QUESTION:
{question}

<think>
{reasoning}
</think>

ANSWER:
{answer}

MODEL: {model}
TIMESTAMP: {timestamp}
TOKENS: {tokens}
{'=' * 50}

"""

                self.current_handle.write(sample_text)
                self.current_handle.flush()
                os.fsync(self.current_handle.fileno())

                sample_bytes = len(sample_text.encode('utf-8'))
                self.total_samples_written += 1
                self.total_bytes_written += sample_bytes

                self._check_rotation()
                return True

            except Exception:
                return False

    # ── Stats & lifecycle ────────────────────────────────────

    def get_stats(self) -> dict:
        """Get storage statistics"""
        return {
            "current_file": str(self.current_file) if self.current_file else None,
            "file_counter": self.file_counter,
            "total_samples_written": self.total_samples_written,
            "total_bytes_written": self.total_bytes_written,
            "total_mb_written": round(self.total_bytes_written / (1024 * 1024), 2),
            "total_gb_written": round(self.total_bytes_written / (1024 * 1024 * 1024), 3),
            "current_file_size_mb": round(self.current_file.stat().st_size / (1024 * 1024), 2) if self.current_file else 0,
        }

    def get_current_file_path(self) -> Optional[str]:
        """Get current file path"""
        return str(self.current_file) if self.current_file else None

    def close(self):
        """Close current file handle"""
        if self.current_handle:
            self.current_handle.flush()
            os.fsync(self.current_handle.fileno())
            self.current_handle.close()
            self.current_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
