# Decision Log

Meaningful decisions and deviations from `PROJECT_PLAN.md`, newest first.

## 2026-08-25 — SFT v0.2, benchmark v3, evaluation hardening, exp-002 prep

Evaluation-first hardening + a genuinely diverse dataset, preparing the **first real**
fine-tune. Historical artifacts (v1/v2 benchmarks, sft_v0.1 builder, exp-001) unchanged.

- **Corrected the record: `gemma3-cyber:v0.1` was never fine-tuned.** Its `Modelfile` is
  `FROM gemma3:4b` + a system prompt; exp-001 responses are 66/67 identical to the base and
  its scorecard equals the base. The "Kerberoasting → T1060" failure is **base-model**
  behavior. exp-002 (this milestone) is the first real training run. Not rewriting exp-001;
  documented in `docs/experiments/exp-002.md`.
- **Root cause of the failure (evidence).** (1) `sft_v0.1` has **360 rows but only 91 unique
  answers** — the builder cloned each answer 5–10× via `for var in range(...)` loops. (2)
  The exact ID `T1558.003` appears **0×** in training and **0×** in `benchmark_v2`, so it
  was neither taught nor tested. (3) The `keyword` scorer would give partial credit to a
  "T1060" answer. So the problem is **dataset diversity + evaluation blindness**, not the
  architecture.
- **Fact registry (single source of truth).** `data/knowledge/security_facts.json` +
  `src/gemma_cyber/knowledge/facts.py`: verified ATT&CK IDs/tactics/mechanics with
  `forbidden_ids` (common confusions) and `obsolete_ids` (e.g. T1060 → T1547.001). Reused by
  scorers, the judge rubric, and the dataset builder so a fact is defined **once** and
  cannot drift. ATT&CK IDs verified against MITRE ATT&CK Enterprise at authoring time (must
  be re-confirmed at release — ATT&CK is versioned).
- **`factual` scorer (evaluation-first fix).** New deterministic scorer:
  `forbidden` hit → **hard fail (0.0)** regardless of correct keywords; `required_all` /
  `required_any` must be satisfied to pass. ATT&CK-ID matching uses word boundaries so
  T1558 ≠ T1558.003 and forbidden T1060 ≠ T10600. Added behind the existing dispatch;
  v1/v2 scorers unchanged. Locked by tests, incl. the exact T1060 failure text.
- **Judge rubric → `judge-v2`.** Reweighted to put **technical correctness first** (facts →
  ATT&CK IDs → protocol → evidence grounding → …), with an explicit critical-error override
  (wrong ATT&CK ID / fabricated artifact / false protocol claim → FAIL even if other content
  is correct). Fail-safe parsing unchanged. **Calibration must be re-run** before citing
  judge numbers against the judge-v1 baseline (deterministic scorers remain primary).
- **`sft_v0.2` (277 items, 277 unique answers).** New builder `builder_v2.py` (v0.1 builder
  frozen). No clone loops — a `_Deduper` raises on any identical answer. ATT&CK-precision +
  contrastive + chain-mapping families are **registry-driven** (35 verified techniques, each
  genuinely distinct). Diversity measured by `validate_dataset.py` (new): **v0.1 = 91/360
  unique, 1145 near-dup pairs → v0.2 = 277/277, 0 near-dups.** 15 task types; schema-valid;
  contamination-clean vs v1/v2/v3 at 0.5. **Quality-over-quota:** built genuinely-distinct
  content toward the ~300 range rather than padding to 500–700 with filler; the builder
  scales cleanly (each registry technique = +1–2 items, zero dup risk).
- **`benchmark_v3` (47 items).** Targeted sensitivity instrument (ATT&CK precision, false
  premises, protocol mechanics, hallucination, evidence attribution) with explicit stratified
  dev/test (35/12), built by `scripts/build_benchmark_v3.py`. **v2 stays the frozen
  do-no-harm anchor; v3 is not comparable head-to-head** (new scorer + harder distribution)
  — separate baselines/targets documented in `configs/eval_success_criteria.md` §7.
- **Gemma-3 formatting.** `src/gemma_cyber/data/formatting.py::to_gemma_chat_text` renders
  the canonical Gemma turn format, **folding the `system` role into the first user turn** and
  emitting `model` turns — because Gemma chat templates historically reject a standalone
  `system` role. Tested over every v0.2 item. The Colab notebook now uses it + completion-only
  masking (loss on `model` turns) and points at `sft_v0.2`, seq_len 1024.
- **Training config.** New `configs/training/qlora_gemma3_4b_v0.2.yaml` (→ sft_v0.2, exp-002
  output dir, seq_len 1024 for T4 headroom); v0.1 config preserved. Dry-run passes
  (dataset + config verified; no local GPU, cloud-only per Q2).
- **Tests:** 90 passing (added factual-scorer, fact-registry, formatting, benchmark_v3
  suites). ruff + mypy clean.

## 2026-08-24 — P1 evaluation hardening: LLM-judge scorer + CI

Evaluation-hardening work (no training; the model is unchanged). Deterministic scorers and
Benchmark v1/v2 reproducibility are all preserved.

- **LLM-judge scorer** (`src/gemma_cyber/evaluation/judge.py`) added *behind the existing
  scorer interface*: `JudgeScorer.score()` returns the same `ScoreResult` type, and
  `.evaluate()` returns a full `JudgeVerdict` audit record (judge model, options,
  prompt version, raw output, parsed decision, error). It is a **supplement**, not a
  replacement — the harness records the deterministic score as primary and attaches judge
  fields alongside (`run_benchmark(..., judge=...)`). **Judge failures never inflate
  scores:** empty/malformed/errored output → FAIL with an `error` string (locked by tests).
- **Judge ≠ subject:** default judge is `gemma4:26b` grading `gemma3:4b` (per Q1), to avoid
  self-grading bias. Documented limitation: both are Gemma-family, so correlation is
  reduced, not eliminated.
- **Thinking-model finding (important):** the Gemma 4 family are *reasoning* models — with
  Ollama's default thinking ON they spend the token budget on hidden reasoning and return
  an **empty** `response`. First calibration attempt: 13/22 empty-output errors, 54.5%.
  Fix: added `think` passthrough to `OllamaClient.generate` and the judge calls with
  `think=False`. After the fix: **0 errors, 86.4% agreement.** The same flag will be needed
  if a Gemma 4 model is ever used as an evaluation *subject* (a future explicit experiment).
- **Calibration** (`scripts/judge_calibration.py`, `data/evaluation/judge_calibration.jsonl`,
  `experiments/judge_calibration/`): 22 dev items (held-out `test` never used for judge
  development), spanning categories/difficulties/scorers and deliberately including
  borderline cases. **Judge-vs-gold agreement 19/22 = 86.4% (target ≥80% MET).** Honesty
  caveat: the gold is **engineer-adjudicated** reference labels (not independent human
  labels); each row has an empty `human_verdict` slot for the owner to verify. The 3
  disagreements are all flagged-borderline items. Notably, judge-vs-deterministic agreement
  is only 50% — quantifying how many keyword false negatives the judge recovers.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): runs `pytest` on push/PR via `uv`
  (Python 3.12). Environment-safe — needs no Ollama/models/GPU/LLM; the live inference test
  auto-skips and judge tests mock the provider.
- **Tests:** 65 passing (added `tests/test_judge.py` + a harness judge-integration test).

## 2026-08-24 — Open questions Q1/Q2/Q3 answered (project owner decisions)

The three critical open questions (PROJECT_PLAN.md §7/§27) are now decided by the
project owner. These are **authoritative** for all subsequent development.

### Q1 — Base model strategy

**Decision.**
- **`gemma4:26b` is the preferred local development model when the available hardware can
  run it effectively** (the M3 Max / 128 GB machine can; the 16 GB ThinkPad cannot).
- **`gemma3:4b` remains the fallback** for constrained hardware **and remains the
  established historical/reproducible baseline.** It is not replaced or erased.
- **`gemma4:12b`** may be used as an intermediate local-development/reference model when
  appropriate (e.g. the 128 GB machine for lighter workloads).
- Adopting a Gemma 4 model as a *development* model does **not** change the benchmark's
  canonical baseline. Any base-model change to the baseline must be a **separate, explicit
  experiment** with its own manifest + committed scorecard under `experiments/`, compared
  on the frozen `benchmark_v2` splits — never a silent substitution.

**Rationale.** `gemma3:4b` is small, CPU-serveable, already baselined (v1 pass 0.84, v2
test 0.933 / hallucination 0.000), and gives a reproducible reference point the whole
thesis is measured against. Larger local Gemma 4 models raise the achievable ceiling for
development/reference where hardware permits, but "did specialization help?" is only
answerable against a *fixed* baseline; mixing in a stronger base silently would confound
that. So: Gemma 4 for capability headroom in dev, `gemma3:4b` preserved as the measurement
anchor. Both are "a Gemma" under the Gemma Terms (see Q3).

**Operational rule:** `gemma4:26b preferred when hardware permits → gemma3:4b fallback`,
while `gemma3:4b baseline → reproducible historical comparison` is preserved.

### Q2 — Training / GPU budget

**Decision.**
- **No paid GPU infrastructure at this time.** No RunPod, AWS, Lambda, Paperspace, or
  other paid GPU services.
- Preferred execution environments, in order: **(1) local hardware; (2) free Google Colab;
  (3) free Kaggle.**
- This may only change via a new, explicit owner decision recorded here.

**Consequence for the roadmap.** Local fine-tuning constraints remain in force (the dev
machines have no CUDA GPU; QLoRA/bitsandbytes need CUDA — PROJECT_PLAN.md §20). The first
fine-tuning experiment (future P2) must fit inside a **free Colab/Kaggle T4/A10-class**
session: small QLoRA, modest sequence length, checkpoint to Drive/output, ephemeral. Any
plan that assumes a rented/paid GPU is out of scope until this decision changes. Nothing in
the current milestone spends money or provisions GPUs.

### Q3 — Project intent: distribution

**Decision.** The project **is intended for distribution.** Licensing, provenance, dataset
sourcing, model licensing, and redistribution obligations are therefore **first-class
engineering constraints**, not afterthoughts.

**Implications (existing policy in `DATA_LICENSES.md` and PROJECT_PLAN.md §16 remains
authoritative — this records intent, not new legal conclusions):**
- **Benchmark data:** already original + `CC-BY-4.0` with recorded `source`/`provenance`
  (v1 and v2). Keep every future eval/training row licensed and provenanced; the schema
  validator already rejects unlicensed rows.
- **Training data (future):** must be original or appropriately licensed, provenance
  tracked, and **contamination-checked** against the benchmark before use
  (`scripts/check_contamination.py`). No HTB/THM or other proprietary content — this is
  reinforced by the distribution intent, not relaxed by it.
- **Third-party content:** excluded unless its license explicitly permits redistribution;
  uncertain-status material goes to `data/quarantine/` (git-ignored) and never ships.
- **Model licensing:** any model derived from Gemma **remains "a Gemma"** under the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) + Prohibited Use Policy, which
  flow downstream on distribution. This applies to both `gemma3:4b` and any Gemma 4
  derivative (Q1). Distribution must pass those terms downstream.
- **Copyright / ToS:** treat all externally sourced content as untrusted; no scraping of
  ToS-protected material. A licensing/legal review is advisable before any public release
  (tracked as an open item, not asserted as done).
- This decision does **not** grant permission to include unlicensed or improperly sourced
  cybersecurity material — it raises the bar.

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

## Open decisions — RESOLVED 2026-08-24 (see top of this log for full records)

- **Q1** ✅ `gemma4:26b` preferred local dev model where hardware permits; `gemma3:4b`
  remains the fallback **and** the frozen reproducible baseline (not replaced).
- **Q2** ✅ No paid GPU. Prefer local → free Colab → free Kaggle.
- **Q3** ✅ Intended for distribution — licensing/provenance/model-terms are first-class.
