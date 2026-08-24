# Decision Log

Meaningful decisions and deviations from `PROJECT_PLAN.md`, newest first.

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
