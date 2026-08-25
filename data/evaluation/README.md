# Evaluation Benchmarks

This directory holds the project's **frozen evaluation benchmarks**. They are the
measurement instrument for the whole project: they exist to answer, honestly and
reproducibly, *"did a change (prompt, RAG, or fine-tune) measurably help?"*

> ⚠️ **Never used for training.** Everything in `data/evaluation/` is isolated from
> `data/training/`. A contamination check (`scripts/check_contamination.py`) must pass
> before any training run. See [`../../DATA_LICENSES.md`](../../DATA_LICENSES.md) and
> `PROJECT_PLAN.md` §16/§18.

## Files

| File | Items | Split | Status |
|---|---:|---|---|
| `benchmark_v1.jsonl` | 25 | none (all treated as `dev`) | **Frozen, historical.** Do not edit. |
| `benchmark_v2.jsonl` | 112 | `dev` / `test` | **Frozen.** The overall do-no-harm / regression anchor. |
| `benchmark_v3.jsonl` | 47 | `dev` / `test` (35/12) | **Targeted sensitivity instrument** (ATT&CK precision, false premises, factual scorer). Not comparable head-to-head with v2. |

All are **100% original**, authored for this project, licensed CC-BY-4.0, with no
Hack The Box / TryHackMe or other third-party content.

### Why v3 exists (and why it does not replace v2)

`benchmark_v2` has an `attack_mapping` category but **zero** items testing the *exact*
ATT&CK ID for Kerberoasting, and its `keyword` scorer would award partial credit to an
answer that says "T1060" as long as it also says "ticket" — so v2 is **blind to the exact
v0.1 failure**. `benchmark_v3` adds that sensitivity: exact-fact items scored by the new
**`factual`** scorer (`required_all` IDs + **`forbidden`** wrong IDs → hard fail), driven
by the verified fact registry (`data/knowledge/security_facts.json`) so training data,
benchmark, and scorer cite the **same** fact and cannot drift. Because v3 uses a new scorer
and a harder distribution, **it is not comparable to v2 head-to-head** — report v2
(do-no-harm) and v3 (targeted) separately, each with its own baseline (see
`configs/eval_success_criteria.md` §7 and `docs/experiments/exp-002.md`). v3 is built
deterministically by `scripts/build_benchmark_v3.py`; v1/v2 remain unchanged.

### Why v2 exists

`benchmark_v1` (25 items) was a correct proof of concept but statistically
underpowered: 14 categories over 25 items meant most categories had n=1–4, so a
post-training delta would be indistinguishable from noise. `benchmark_v2` expands the
instrument to **112 items, ≥6 per category**, deepens the previously thin categories
(Active Directory, incident response, detection engineering, evidence interpretation),
and increases the discriminating traps (**8 hallucination + 8 insufficient-evidence**
items) that separate an evidence-grounded model from one that merely produces plausible
security prose. `benchmark_v1` is kept unchanged for provenance and reproducibility of
the original baseline.

## Schema

Each line is one JSON object validated by
`src/gemma_cyber/evaluation/schema.py::BenchmarkItem`. Key fields:

- `id` — stable unique identifier (v2 ids are disjoint from v1).
- `category`, `domain` (`blue_team` | `offensive_ctf` | `general`), `difficulty`.
- `scorer` — `mcq` | `keyword` | `insufficient_evidence` | `hallucination` | `factual`
  (all deterministic).
- prompt fields: `question`, optional `context` / `evidence` / `choices`.
- scoring fields: `answer` (mcq), `expected_keywords` + `keyword_threshold` (keyword),
  `required_all` / `required_any` / `forbidden` (`factual` — a `forbidden` hit hard-fails
  the item regardless of other content).
- provenance: `source`, `license`, `provenance` (all mandatory).
- **`split`** — `dev` | `test` (see below). Defaults to `dev` for pre-split files.

## Dev / test split policy

`benchmark_v2` is partitioned into two frozen splits:

- **`dev` (~60%, 67 items)** — the *iteration* set. You may look at individual `dev`
  items and their model responses, tune prompts/scorers, and use it during development.
- **`test` (~40%, 45 items)** — the **held-out** set. Its purpose is to give an
  unbiased final measurement that development has **not** been optimized against.

Approximate ratio: **60 dev / 40 test** overall (actual: 67/45 = 40.2% test), assigned
**stratified per category** so each category contributes ~40% of its items to `test`.

### What is permitted vs. prohibited

**Permitted (using `dev`):**
- Inspecting `dev` items and responses; iterating on system prompts and scorers.
- Selecting training data and iterating on it while checking it against *both* splits
  for contamination.
- Reporting `dev` scores as development signal.

**Prohibited (protecting `test`):**
- Do **not** tune prompts, scorers, thresholds, or training data by looking at `test`
  item content or `test` responses.
- Do **not** add, remove, or reword `test` items to change an outcome.
- Do **not** repeatedly re-run against `test` to pick the "best" configuration — that
  re-introduces the overfitting the split exists to prevent. `test` is for the final,
  pre-registered comparison (see `configs/eval_success_criteria.md`).

Run a split explicitly with the harness:

```bash
python scripts/run_baseline.py --benchmark data/evaluation/benchmark_v2.jsonl \
    --split dev  --out experiments/baseline_gemma3-4b_v2/dev
python scripts/run_baseline.py --benchmark data/evaluation/benchmark_v2.jsonl \
    --split test --out experiments/baseline_gemma3-4b_v2/test
```

## Freeze & versioning policy

- The split is **committed in the data itself** (each row carries its `split`), so it is
  deterministic and reproducible — it is never regenerated at runtime.
- Treat each `benchmark_vN.jsonl` as **immutable** once committed and referenced by a
  baseline. Improvements ship as a **new version** (`benchmark_v3.jsonl`), never as
  in-place edits, so historical scorecards stay comparable.
- Tag releases in git so a given baseline can always be reproduced against the exact
  benchmark it was run on.

## Integrity / quality checks

```bash
# Distributions + acceptance-requirement gate (exits non-zero on any violation):
python scripts/inspect_benchmark.py data/evaluation/benchmark_v2.jsonl

# Ensure v2 did not copy v1 (or, later, that training data does not leak the benchmark):
python scripts/check_contamination.py data/evaluation/benchmark_v2.jsonl \
    data/evaluation/benchmark_v1.jsonl
```

These checks are also enforced by the test suite (`tests/test_schema.py`,
`tests/test_contamination.py`).
