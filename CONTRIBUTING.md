# Contributing

Thanks for taking a look. This document covers how the project is laid out, which
behaviours are load-bearing, and how to test a change without burning GPU hours.

---

## Getting set up

```bash
git clone <your-fork> distillation && cd distillation
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
sudo apt install redis-server && sudo systemctl enable --now redis-server
cp .env.example .env    # point STUDENT_* and TEACHER_* at any OpenAI-compatible /v1
```

For development you do not need a large model. Two small local servers are enough
to exercise the whole pipeline:

```bash
# terminal 1
python -m vllm.entrypoints.openai.api_server --model <small-model> --port 8000
# terminal 2
python -m vllm.entrypoints.openai.api_server --model <small-model> --port 8001
```

Set `STUDENT_BASE_URL=http://localhost:8000/v1` and
`TEACHER_BASE_URL=http://localhost:8001/v1`, then:

```bash
python3 main.py config/default.yaml
```

---

## Layout

```
main.py                     entry point: loads config, starts the coordinator
distill/
  config.py                 dataclass config + YAML/.env loading and overrides
  core/
    research_agent.py       question generation, depth escalation, exhaustion
    teacher_client.py       teacher prompt + reasoning/answer separation
    teacher_registry.py     teacher configs, health, weights
    multi_teacher_orchestrator.py   parallel fan-out with rate limiting
    teacher_merger.py       consensus, contradictions, best-insight merge
    topic_discovery.py      expansion when the topic pool drains
    seed_topics.py          the seed topic list
  orchestration/
    coordinator.py          lifecycle, worker launch, checkpoints, dashboard
    queue_manager.py        all Redis queue operations
  quality/
    scorer.py               quality dimensions + JSONL/RAW gating
    duplicate_detector.py   three-layer dedup (in-memory + Redis-backed)
  storage/
    raw_txt_storage.py      the dataset file format
    jsonl_backup_writer.py  JSONL mirror
    checkpoint_manager.py   crash-recovery state
  knowledge_graph/          concept graph, coverage, gap analysis
  llm/client.py             HTTP client for OpenAI-compatible endpoints
workers/                    one file per worker type
bin/qna.sh                  dataset inspection CLI
scripts/deploy_vps.sh       idempotent VPS deployment
```

---

## Invariants worth protecting

These are the behaviours that took real debugging to get right. A change that
breaks one of them will look fine in a short run and fall apart over hours.

**1. Nothing is written before it passes quality.**
`workers/multi_teacher_answer_worker.py` must not write to storage.
`workers/quality_verification_worker.py` is the only writer, and only on pass. If
you reintroduce a write-then-score path, every rejected sample stays in the
dataset forever and the file silently becomes unusable.

**2. A failed judge must never produce a silent zero.**
`distill/quality/scorer.py` falls back to neutral values *and logs a warning* when
the judge returns nothing parsable. Returning the dataclass default of `0.0` for
correctness means a 0% accept rate that looks like bad data but is a broken judge.

**3. The judge must not be the teacher.**
Keep the student model disabled in `config/teachers.yaml`. Same-model grading is
self-play: the correctness dimension stops measuring anything.

**4. Reasoning and answer stay separate fields.**
The teacher prompt asks the model to think privately, then answer. Do not
reintroduce a "THINKING" section inside the answer body — it turns the stored
reasoning into meta-commentary about the answer rather than the thinking behind it.

**5. Dedup state lives in Redis, not just in memory.**
Per-worker in-memory indexes vanish on restart and are invisible to sibling
workers. `DuplicateDetector.check_and_register()` is the cross-process path; keep
it in the question worker's flow.

**6. Per-topic budget is enforced in the question worker.**
Without it the pipeline loops on one topic indefinitely. The budget counter is
`distill:topiccount:<slug>` and retirement repopulates the topic queue from
`distill:topic_pool`.

**7. Endpoints come from `.env`.**
Config files use `${VAR}` placeholders expanded at load. Never commit a hostname.

---

## Testing a change

There is no test suite yet — contributions there are especially welcome. Until
then, the practical loop is:

```bash
# 1. syntax + import check
python -m compileall -q distill workers main.py

# 2. load config and print effective settings
python -c "
from distill.config import Config
c = Config.from_yaml('config/production.yaml')
print(c.research_model, c.teacher_model, c.num_answer_workers)
"

# 3. run the pipeline for ~5 minutes and read the log
python3 main.py config/production.yaml
grep -E 'ACCEPTED|REJECTED|judge output unparsable|Traceback' /root/logs/distill_*.log
```

For scoring or parsing changes, unit-test the pure functions directly — they need
no network:

```python
from distill.quality.scorer import FrontierQualityScorer
assert FrontierQualityScorer._parse_four_scores("0.85,0.90,0.75,0.60")
assert FrontierQualityScorer._parse_four_scores(
    "Completeness: 0.8 Correctness: 0.9 Reasoning depth: 0.7 Implementation depth: 0.5"
)
```

### What to check before opening a PR

- `python -m compileall -q .` is clean
- a 5-minute run produces at least one `ACCEPTED and saved` line
- no new `ERROR` or `Traceback` lines in the log
- `grep -c 'judge output unparsable'` has not jumped
- if you touched storage: `/root/datasets/raw_001.txt` blocks still parse with
  `bash bin/qna.sh -`

Mention which models you tested against in the PR description — behaviour around
reasoning channels differs a lot between model families.

---

## Style

- Match the surrounding code; no reformatting of untouched lines.
- Comments explain *why*, not *what*. The interesting ones in this codebase mark
  a bug that was actually hit — keep that spirit if you add one.
- async/await throughout the worker and client layers; do not introduce blocking
  I/O in a coroutine.
- Keep prompts in `distill/llm/prompts.py` or the owning client so they stay
  greppable and tunable in one place.

---

## Reporting a bug

Include: what you expected, what happened, the relevant log lines, the models in
use, and whether the accept rate changed. A log excerpt around the first
occurrence is worth more than a description of the symptom.
