# Autonomous Knowledge Distillation

A 24/7 pipeline that generates high-quality reasoning training data by making two
models work against each other: one asks increasingly hard questions, the other
answers them, and a judge decides what is worth keeping.

The output is a dataset of question → chain-of-thought → answer triples with
per-sample quality scores, written crash-safely to disk.

---

## Why it works this way

Most synthetic-data scripts are a single prompt in a `for` loop, and they produce
shallow, repetitive data. This system is built around four ideas that fix that:

**The examiner is not the examinee.** A separate *student* model writes the
questions and grades the answers; the *teacher* model produces them. If one model
does both, grading degrades into self-play and the correctness score becomes
meaningless.

**Reasoning is captured separately from the answer.** The teacher prompt asks the
model to think first, in private, then emit only the polished answer. The thinking
lands in the reasoning channel, the answer in the content channel, and the two are
stored as distinct fields. Designs that ask for a "THINKING" section *inside* the
answer end up training on the merger's meta-commentary instead of real reasoning.

**Nothing is written until it passes.** Samples are scored first; only the quality
worker touches the dataset file, and only on pass. Writing first and scoring
second means every rejected sample silently stays in the dataset forever.

**Topics rotate.** Each topic has a question budget. When it is spent the topic is
retired and a fresh one is pulled from the rotation pool, so the corpus spreads
across the whole seed list instead of collapsing onto whichever topic started
first.

---

## Architecture

```
  seed topics ──┐
                ▼
        ┌───────────────┐
        │ topic_queue   │   per-topic question budget enforced here
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ question      │   student model asks; validated (real concept,
        │ worker        │   ends with '?', not truncated); deduped
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ answer        │   N teachers queried in parallel;
        │ worker        │   merge only when 2+ respond
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ quality       │   scored; ACCEPT → written to RAW + JSONL
        │ worker        │   REJECT → requeued, never written
        └───────┬───────┘
                │
                └──► back to question_queue for the next question on this topic
```

Redis carries work between stages, so any stage can be restarted or scaled
independently without losing in-flight items. Checkpoints persist counters; the
queues themselves live in Redis.

### Pipeline stages

| Stage | What it does |
|---|---|
| **Research agent** | Writes the next question at a target depth level (concept → implementation → math → edge cases → production → research). Sees the full question history for the topic and is instructed never to repeat an angle. |
| **Validator** | Rejects a generated question if it is truncated, does not end with `?`, rambles past ~160 words, or invents a concept ("formally derive **SomeMadeUpFramework**"). Regenerates up to 3 times. |
| **Dedup** | Three layers: SHA-256 of the normalized question, Jaccard + key-phrase overlap against the topic history, and optional LLM comparison. Indexes live in Redis, so they span workers and survive restarts. |
| **Teacher orchestration** | Fans out one request per active teacher concurrently, with per-teacher rate limiting, health tracking and partial-failure tolerance. |
| **Merger** | With 2+ answers: consensus extraction, contradiction detection, best-insight selection, and enrichment (examples, code, references). With 1 answer: skipped — it would only re-emit the same text. |
| **Quality scorer** | Scores completeness, correctness, reasoning depth and implementation depth in one judge call, plus novelty, factual consistency and hallucination probability heuristics. A judge failure falls back to neutral values with a loud warning — never a silent zero, which would reject every sample forever. |

---

## Requirements

- Linux, Python 3.10+
- Redis
- Two OpenAI-compatible `/v1` endpoints — one for the student, one for the teacher
  (vLLM, SGLang, llama.cpp server, or a hosted API)
- ~2 GB disk per 1,000 samples at default settings

Any endpoint that speaks `POST /v1/chat/completions` and `GET /v1/models` works. A
separate `reasoning` / `reasoning_content` field in the response is used when
present; otherwise the client splits reasoning out of the content heuristically.

---

## Quick start

```bash
git clone <your-fork> distillation && cd distillation

cp .env.example .env
$EDITOR .env                     # set STUDENT_* and TEACHER_*

pip install -r requirements.txt
sudo apt install redis-server && sudo systemctl enable --now redis-server

python3 main.py config/production.yaml
```

Press `Ctrl+C` to stop — it checkpoints and exits cleanly.

### Deploy as a service

```bash
sudo bash scripts/deploy_vps.sh /root/distillation
```

Creates the virtualenv, installs dependencies, verifies both endpoints, installs
the systemd unit and starts the service. Idempotent — safe to re-run.

---

## Configuration

Two files, both driven by `.env`:

**`config/production.yaml`** — endpoints, worker counts, quality thresholds, topic
budget. Any field can be overridden per deployment with
`DISTILL_<UPPERCASE_FIELD>`, e.g. `DISTILL_NUM_ANSWER_WORKERS=16`.

**`config/teachers.yaml`** — the teacher registry. Add a teacher, set its `${VAR}`,
flip `enabled: true`, and it joins the fan-out; weights normalize across active
teachers automatically.

Endpoints are written as `${VAR}` placeholders and expanded at load time, so
hostnames stay in `.env` and never in the repository.

### Tuning

| Knob | Effect |
|---|---|
| `num_answer_workers` | The main throughput lever. These are I/O bound on a remote GPU endpoint, so scale them with *that* endpoint's concurrency, not local CPU count. |
| `teacher_concurrency_limit` | Simultaneous teacher requests. Raise together with `num_answer_workers`. |
| `max_questions_per_topic` | Depth per topic before rotating. Lower = broader coverage, higher = deeper. |
| `min_quality_score` | Acceptance bar. Watch the accept rate after changing it. |
| `min_answer_length` | Hard floor applied before scoring; thin answers never cost a judge call. |

---

## Output

Two streams, written together on every accepted sample.

**`/root/datasets/raw_001.txt`** — human-readable, rotated at
`raw_txt_rotation_size_gb`:

```
======================================================================
TOPIC: Database internals SQL optimization and query planning
DIFFICULTY: expert
TEACHERS: teacher-primary
AGREEMENT_SCORE: 0.000

QUESTION:
<the question>

 thinking
<the model's chain of thought>
</think>

ANSWER:
<the answer>

METADATA:
MODEL: ...
QUALITY_SCORE: 0.791
COMPLETENESS: 0.900
CORRECTNESS: 0.900
REASONING_DEPTH: 0.750
IMPLEMENTATION_DEPTH: 0.760
NOVELTY: 0.800
FACTUAL_CONSISTENCY: 0.800
HALLUCINATION_PROB: 0.0500
TIMESTAMP: ...
TOKENS: 3303
RETRY_COUNT: 0
LOW_QUALITY: false
======================================================================
```

**`/root/datasets/jsonl/samples_*.jsonl`** — one JSON object per sample for
programmatic use (fine-tuning loaders, analytics).

Every accepted sample carries its full score breakdown, so you can filter after
the fact (`QUALITY_SCORE >= 0.8`, `HALLUCINATION_PROB < 0.1`) without re-running
the pipeline.

### Inspecting the dataset

`bin/qna.sh` ships with the repo:

```bash
qna -stats      # samples, questions, topics, queues, throughput
qna             # latest sample in full
qna -s          # latest sample, question + answer only
qna -m 5        # quality metadata for sample 5
qna -           # index: number, topic, first line of each question
qna -t          # topic tally
```

---

## Operating notes

- **Judge output is not always clean.** Reasoning models can spend the whole token
  budget in the reasoning channel and return `content: ""`. The scorer parses
  labelled, CSV and prose forms, scans the reasoning tail, and falls back to
  neutral scores with a warning. After changing judge models, check
  `grep -c 'judge output unparsable' <log>`.
- **Watch the accept rate.** A sudden 0% accept rate almost always means the
  judge, not the data. Silent-zero defaults are the classic cause.
- **Retry loops amplify.** A rejected sample is requeued for a fresh answer; if
  the judge rejects everything, the answer queue grows without bound. Watch
  `redis-cli llen distill:queue:answer`.
- **Disk.** Rotation is per file, so total usage is unbounded by design. Monitor
  the dataset directory or archive rotated files.
- **Topics rotate on budget, not on quality.** A topic is retired when its budget
  is spent or the agent reports exhaustion — never because the answers were weak.

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) — it covers the
layout, the invariants worth protecting, and how to test a change without burning
GPU hours.

---

## License

MIT — see [LICENSE](LICENSE).
