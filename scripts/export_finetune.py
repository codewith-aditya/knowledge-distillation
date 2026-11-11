#!/usr/bin/env python3
"""
Convert the pipeline's JSONL output into a clean fine-tuning dataset.

What it removes (the "usage points" noise that has no business in training data):
  model, tokens, teachers, agreement_score, timestamp, and the whole quality
  dict — every field that describes the run rather than the Q&A itself.

What it keeps:
  question, answer, reasoning (chain of thought), topic, difficulty.

Also:
  - strips compounded "Related to: Related to: ..." topic labels
  - drops duplicate questions (normalized hash)
  - drops rows below a quality bar, short answers, or empty answers
  - writes train/val splits plus a summary you can paste into a dataset card

Usage:
    python3 scripts/export_finetune.py \
        --input '/root/datasets/jsonl/*.jsonl' \
        --outdir /root/datasets/finetune \
        --min-quality 0.70

Output (in --outdir):
    train.jsonl          messages-format chat records
    val.jsonl            held-out split
    stats.json           counts, filters applied, topic/score distribution
    DATASET_CARD.md      ready-to-edit HuggingFace card
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# ── Cleaning helpers ────────────────────────────────────────

_PREFIX = re.compile(r"^\s*related\s+to\s*:\s*", re.IGNORECASE)
_FALLBACK_TOPIC = "general software engineering"
_MAX_TOPIC_LEN = 240

# Fields that describe the run, not the data. Never exported.
DROP_FIELDS = (
    "model",
    "tokens",
    "teachers",
    "teachers_contributed",
    "teachers_failed",
    "agreement_score",
    "quality",
    "timestamp",
    "retry_count",
    "low_quality",
)

# Boilerplate that some teacher prompts leave at the end of an answer.
_TRAILING_BOILERPLATE = re.compile(
    r"\n+(?:METADATA|USAGE|TOKENS|QUALITY_SCORE)\s*:.*\Z", re.IGNORECASE | re.DOTALL
)


def clean_topic(topic: str | None) -> str:
    """Strip nested 'Related to:' prefixes; cap the label length."""
    name = (topic or "").strip()
    while True:
        stripped = _PREFIX.sub("", name, count=1)
        if stripped == name:
            break
        name = stripped.strip()
    if len(name) > _MAX_TOPIC_LEN:
        name = name[:_MAX_TOPIC_LEN].rstrip()
    return name or _FALLBACK_TOPIC


def clean_text(value: str | None) -> str:
    """Trim, and cut any metadata footer that leaked into the body."""
    text = (value or "").strip()
    if not text:
        return ""
    text = _TRAILING_BOILERPLATE.sub("", text)
    # Right-strip the "===" separators the raw writer appends.
    text = re.sub(r"\n=+\s*\Z", "", text)
    return text.strip()


def question_hash(question: str) -> str:
    norm = re.sub(r"[^a-z0-9 ]+", " ", question.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ── Record building ─────────────────────────────────────────

def build_record(rec: dict, include_reasoning: bool) -> dict | None:
    """Turn one pipeline record into a training record, or None to drop it."""
    question = clean_text(rec.get("question"))
    answer = clean_text(rec.get("answer"))
    reasoning = clean_text(rec.get("reasoning"))

    if not question or not answer:
        return None

    out = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "topic": clean_topic(rec.get("topic")),
        "difficulty": (rec.get("difficulty") or "").strip() or "unknown",
    }
    if include_reasoning and reasoning:
        out["reasoning"] = reasoning
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a fine-tuning dataset from pipeline JSONL.")
    ap.add_argument("--input", default="/root/datasets/jsonl/*.jsonl",
                    help="glob for input JSONL files")
    ap.add_argument("--outdir", default="/root/datasets/finetune", help="output directory")
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="drop records whose overall quality score is below this (0 = keep all)")
    ap.add_argument("--min-answer-chars", type=int, default=500,
                    help="drop answers shorter than this")
    ap.add_argument("--val-ratio", type=float, default=0.02, help="validation split fraction")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed")
    ap.add_argument("--no-reasoning", action="store_true", help="omit the reasoning field")
    ap.add_argument("--max-per-topic", type=int, default=0,
                    help="cap records per topic (0 = no cap)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        print(f"no input files matched: {args.input}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    records: list[dict] = []
    per_topic: Counter = Counter()
    score_buckets: Counter = Counter()

    stats = {
        "input_files": paths,
        "read": 0,
        "dropped_empty": 0,
        "dropped_short": 0,
        "dropped_quality": 0,
        "dropped_duplicate": 0,
        "dropped_topic_cap": 0,
        "kept": 0,
        "topics_cleaned": 0,
    }

    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats["read"] += 1

                quality = rec.get("quality") or {}
                overall = quality.get("overall")
                if isinstance(overall, (int, float)):
                    score_buckets[f"{min(int(overall * 10) * 10, 100)}-{min(int(overall * 10) * 10 + 9, 100)}"] += 1

                raw_topic = rec.get("topic") or ""
                cleaned_topic = clean_topic(raw_topic)
                if cleaned_topic != raw_topic.strip():
                    stats["topics_cleaned"] += 1

                if args.min_quality > 0:
                    if not isinstance(overall, (int, float)) or overall < args.min_quality:
                        stats["dropped_quality"] += 1
                        continue

                answer = clean_text(rec.get("answer"))
                if not answer:
                    stats["dropped_empty"] += 1
                    continue
                if len(answer) < args.min_answer_chars:
                    stats["dropped_short"] += 1
                    continue

                question = clean_text(rec.get("question"))
                if not question:
                    stats["dropped_empty"] += 1
                    continue

                h = question_hash(question)
                if h in seen:
                    stats["dropped_duplicate"] += 1
                    continue

                if args.max_per_topic:
                    if per_topic[cleaned_topic] >= args.max_per_topic:
                        stats["dropped_topic_cap"] += 1
                        continue

                seen.add(h)
                per_topic[cleaned_topic] += 1

                record = build_record(rec, include_reasoning=not args.no_reasoning)
                if record is None:
                    stats["dropped_empty"] += 1
                    continue
                records.append(record)

    if not records:
        print("nothing survived the filters", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(records)
    val_n = max(1, int(len(records) * args.val_ratio)) if args.val_ratio > 0 else 0
    val, train = records[:val_n], records[val_n:]

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(outdir / "train.jsonl", train)
    write_jsonl(outdir / "val.jsonl", val)

    stats["kept"] = len(records)
    stats["train"] = len(train)
    stats["val"] = len(val)
    stats["unique_topics"] = len(per_topic)
    stats["avg_answer_chars"] = int(sum(len(r["messages"][1]["content"]) for r in records) / len(records))
    stats["avg_question_chars"] = int(sum(len(r["messages"][0]["content"]) for r in records) / len(records))
    stats["with_reasoning"] = sum(1 for r in records if r.get("reasoning"))
    stats["top_topics"] = per_topic.most_common(15)
    stats["quality_buckets"] = dict(sorted(score_buckets.items()))
    stats["difficulty"] = Counter(r["difficulty"] for r in records).most_common()

    with open(outdir / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    card = f"""# Reasoning Q&A Dataset

Generated with an autonomous distillation pipeline: one model asks progressively
harder questions, a second model answers them, and a judge scores every sample
before it is kept.

## Composition

| | |
|---|---|
| Records | {stats['kept']:,} (train {stats['train']:,} / val {stats['val']:,}) |
| Topics | {stats['unique_topics']} |
| Avg question | {stats['avg_question_chars']} chars |
| Avg answer | {stats['avg_answer_chars']} chars |
| With chain-of-thought | {stats['with_reasoning']:,} records |

## Format

`messages` — a standard chat conversation, plus the model's reasoning kept as a
separate field so it can be used as a distillation target or dropped:

```json
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "reasoning": "...",
  "topic": "...",
  "difficulty": "expert"
}}
```

## Field notes

- `messages[1].content` is the answer only — no reasoning, no metadata.
- `reasoning` is the teacher's own chain of thought, captured from its thinking
  channel. Use it for distillation targets or ignore it.
- `topic` and `difficulty` are for filtering/sampling, not for training.

## Top topics

{chr(10).join(f"- {t} ({n:,})" for t, n in per_topic.most_common(10))}
"""
    (outdir / "DATASET_CARD.md").write_text(card, encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items() if k not in ("input_files", "top_topics")},
                     indent=2, ensure_ascii=False))
    print(f"\nwrote: {outdir}/train.jsonl, {outdir}/val.jsonl, stats.json, DATASET_CARD.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
