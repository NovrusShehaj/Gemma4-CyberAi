# Decision Log

Meaningful decisions and deviations from `PROJECT_PLAN.md`, newest first.

## 2026-08-24 — Benchmark v2 (evaluation instrument) + contamination tooling

Milestone P0 "Benchmark v2" — evaluation infrastructure only. **No training occurred;
no model changed.** `benchmark_v1.jsonl` was left frozen and untouched.

- **New instrument `benchmark_v2.jsonl` (112 original items).** Expanded from v1's 25 to
  ≥6 per category across 16 categories; balanced blue_team (61) / offensive_ctf (35) /
  general (16). Deepened the categories flagged thin in v1 — added an `active_directory`
  category (8) and strengthened incident_response (8), detection_engineering (8), and
  evidence_interpretation (7). Discriminating traps raised to **8 hallucination + 8
  insufficient_evidence**. All items original, CC-BY-4.0, no HTB/THM; ids disjoint from v1.
  - *Taxonomy note:* `category` is a free-form string in the schema, so adding
    `active_directory` is a data change, not a code/taxonomy change. It matches the
    `domain: ad` vocabulary already in PROJECT_PLAN §15.1.
- **Schema: added a validated `split` field (`dev`|`test`).** Defaults to `dev` so the
  pre-split `benchmark_v1` still loads unchanged. v2 is ~60/40 (67 dev / 45 test),
  **stratified per category** and **frozen in the data** (each row carries its split), so
  the split is deterministic and never regenerated at runtime. Held-out `test` policy is
  documented in `data/evaluation/README.md`.
- **Contamination checker built** (`src/gemma_cyber/data/contamination.py` +
  `scripts/check_contamination.py`): normalized exact match + fuzzy word-n-gram (trigram)
  Jaccard, deterministic, stdlib-only, reusable API, CLI exits non-zero on any overlap for
  pre-train gating. It immediately earned its keep: it caught three v2 items I had authored
  too close to their v1 counterparts (Jaccard 0.73–0.81); those were rewritten to be
  original, and v2 is now clean vs v1 (0 exact, 0 fuzzy ≥0.7) with no internal near-dups.
- **Fresh v2 baseline for `gemma3:4b`** (temp0, seed0), committed to
  `experiments/baseline_gemma3-4b_v2/{dev,test}/`:
  - dev: pass_rate **0.836**, mean 0.766 (n=67, 365s).
  - test: pass_rate **0.933**, mean 0.841 (n=45, 182s).
  - **hallucination = 0.000 on both splits** — the v1 finding reproduces at larger n: the
    base model confidently engages fabricated CVEs/tools/artifacts every time.
- **Pre-registered success criteria** (`configs/eval_success_criteria.md`), calibrated to
  this baseline **before** any candidate exists. Key design decision: because `test`
  overall is near-saturated (0.933, only 6.7 pp headroom), the **primary** bar is
  hallucination-resistance improvement (baseline 0.000 → ≥+33 pp on held-out test and
  ≥+50 pp on the full category), with overall test pass_rate used as a do-no-harm guard
  (≥0.913) and a no-category-regression rule. Percentage points are used throughout and
  distinguished from percentages.
- **Runtime note:** the v2 baseline ran far faster than the v1 baseline (5.4 s/item vs
  ~43 s/item) on the same machine — Ollama used GPU/Metal acceleration this session.
  Scores remain deterministic (temp0/seed0); only wall-clock changed.

## 2026-08-23 — Baseline recorded + MCQ scorer bug fixed

- **Baseline (gemma3:4b, benchmark_v1, temp=0, seed=0):** overall pass_rate **0.84**,
  mean_score **0.813** (n=25). Stored in `experiments/baseline_gemma3-4b/`.
- **Scorer bug found & fixed during verification:** the first MCQ extractor matched option
  letters case-insensitively, so lowercase prose (e.g. the article "A" in "A CVSS score…")
  was mis-read as the answer. Made letter matching case-sensitive + leading-answer priority;
  added regression tests. Also broadened insufficient-evidence markers to include
  "impossible to determine" (the model's actual correct phrasing). Re-scored from stored
  responses via `scripts/rescore.py` (no re-inference; outputs are deterministic).
- **Key finding — actionable weakness:** `hallucination` category scored **0.0** — the base
  model engages with a fabricated CVE and a fabricated tool instead of flagging them. This is
  the clearest target for specialization (SFT for refusal/uncertainty behavior + later RAG).
- **Known proxy-scorer softness (left as-is, not tuned):** `evidence_interpretation` (n=1) and
  one `detection_engineering` item scored low on keyword matching though responses looked
  reasonable. Editing keywords after seeing responses would overfit the benchmark; deferred to
  a future LLM-judge scorer (PROJECT_PLAN.md §17).

## 2026-08-23 — Milestone 1: baseline harness implemented

- **Package layout:** used a src-layout package `src/gemma_cyber/` (with `clients/` and
  `evaluation/` subpackages) instead of the plan's loose `src/clients`, `src/evaluation`.
  Reason: cleanly importable/testable and pip-installable. Minor deviation; structure in
  PROJECT_PLAN.md §21 remains conceptually accurate.
- **Scorers are deterministic only (no LLM judge yet).** The plan (§17) mentions an
  LLM-judge scorer; deferred to keep the baseline fully reproducible and avoid
  overengineering in milestone 1. Scorers implemented: `mcq`, `keyword`,
  `insufficient_evidence`, `hallucination`. The harness dispatches by scorer name, so a
  judge scorer can be added later without schema changes.
- **Python 3.12 venv** created with `uv` (system Python is 3.14, too new for the eventual
  ML stack). pyproject pins `>=3.11,<3.13`.
- **Training deps intentionally excluded** from `pyproject.toml`. Training runs in the
  cloud (no local CUDA GPU; PROJECT_PLAN.md §12/§20) and is added in Phase 4.
- **Benchmark v1 = 25 original items** across fundamentals, web, network, log analysis,
  IR, detection engineering, ATT&CK, privesc, Windows/AD, vuln analysis, CTF reasoning,
  plus insufficient-evidence and hallucination traps. No HTB/THM content.

## Open decisions still pending (from PROJECT_PLAN.md §7)

- **Q1** gemma3:4b vs Gemma 4 as base model.
- **Q2** cloud GPU budget/access (gates fine-tuning).
- **Q3** personal/educational vs distribution/commercial (licensing + data obligations).
