# LLM-Judge Scorer

The LLM-judge scorer (`src/gemma_cyber/evaluation/judge.py`) is a **supplementary**
evaluator that grades a model response by *meaning* rather than by literal keyword match.
It exists to fix the deterministic `keyword` scorer's **false negatives** — semantically
correct answers phrased differently than the expected keyword list (common in
`log_analysis`, `incident_response`, `evidence_interpretation`).

> The judge **supplements** the deterministic scorers; it does not replace them. The
> deterministic score remains the primary, fully reproducible number recorded by the
> harness. See `docs/decisions.md` and `configs/eval_success_criteria.md`.

## Architecture / integration

- **Same interface.** `JudgeScorer.score(item, response)` returns the same `ScoreResult`
  type the deterministic scorers return (`scorer="judge"`), so it plugs into the existing
  aggregation. It is a *class* (not a pure function like `score_item`) only because it
  needs an LLM client.
- **Full audit record.** `JudgeScorer.evaluate(item, response)` returns a `JudgeVerdict`
  capturing everything needed to reproduce/audit a decision: judge `model`, `options`
  (temperature, seed, num_predict), `prompt_version`, the raw judge output, the parsed
  decision/score/reason, and an `error` field.
- **Harness hook.** `run_benchmark(..., judge=JudgeScorer(...))` optionally attaches
  `judge_passed` / `judge_score` / `judge_detail` / `judge_error` to each item and a
  `judge` aggregate block — alongside, never overwriting, the deterministic score.
- **Rubric per task type.** `build_reference(item)` produces a grading rubric tailored to
  the item's scorer: mcq (must pick the correct option), keyword (must convey the concepts
  by meaning), insufficient_evidence (must recognize insufficiency), hallucination (must
  flag/deny the fabricated premise, not invent details).

## Determinism, reproducibility, and failure handling

- The judge runs at **temperature 0, fixed seed** by default. LLM inference is not
  bit-for-bit guaranteed across runtime/model-version changes, so the judge is *more*
  reproducible than a sampled judge but *less* reproducible than the deterministic
  scorers — which is exactly why the deterministic score is kept as the stable reference.
- **A judge failure never inflates a score.** Empty output, non-JSON output, invalid JSON,
  a missing/invalid `verdict`, an out-of-range `score`, or a provider/transport exception
  all produce a **FAIL** result with a populated `error` string — never a silent pass.
  (Unit tests in `tests/test_judge.py` lock this behavior in.)

## Judge model vs. subject model (bias)

The judge model must **differ** from the model under evaluation to avoid self-grading
bias. The default judge is `gemma4:26b-a4b-it-q8_0` (per `docs/decisions.md` Q1, the
preferred local dev model where hardware permits) grading `gemma3:4b` responses. This is a
different generation and a much larger model, which reduces — but does not eliminate —
correlated blind spots, since both are Gemma-family. A fully independent judge (different
vendor) would reduce family correlation further; that is a possible future refinement.

## Calibration

Judge quality is calibrated against a labeled set:
`data/evaluation/judge_calibration.jsonl` — 22 items drawn from the **dev** split (the
held-out `test` split is never used for judge/scorer development), spanning categories,
difficulties, scorer types, and — deliberately — hard/borderline cases where the
deterministic scorer and careful judgment disagree.

- **Gold labels.** Each row carries a `reference_verdict` **adjudicated by the
  implementing engineer** by reading the actual stored `gemma3:4b` response against the
  question. These are labeled `reference_source: engineer_adjudicated_pending_human_verification`
  and each row has an empty `human_verdict` slot for the **project owner** to fill for
  independent verification. The reference labels are *not* the deterministic score and are
  *not* the "expected answer == human" fallacy.
- **Methodology.** `scripts/judge_calibration.py` runs the judge over each response and
  computes agreement = (# judge label == gold label) / N. It reports agreements,
  disagreements, agreement %, judge errors, and judge-vs-deterministic agreement, and
  prefers `human_verdict` as gold whenever every row has one (otherwise it uses
  `reference_verdict` and says so). Results: `experiments/judge_calibration/`.
- **Target:** judge-vs-gold agreement **≥ 80%**.

### Result (2026-08-24)

- **Judge model:** `gemma4:26b-a4b-it-q8_0` (thinking disabled — see note below), temp 0, seed 0.
- **Agreement: 19/22 = 86.4%** vs. the engineer-adjudicated reference labels — **target ≥ 80% MET.**
- **Judge errors: 0.** Judge-vs-deterministic agreement: 50% (the judge and the keyword
  scorer diverge substantially — the judge fixes many keyword false negatives, e.g.
  `ir-0101`, `insuf-0102`, `insuf-0104`, `det-0101`, `ctf-0101`).
- **Gold basis caveat:** these are **engineer-adjudicated** reference labels
  (`reference_verdict`), *not* independent human labels. The `human_verdict` slots in the
  calibration file are empty and await the project owner's verification; the reported 86.4%
  is judge-vs-engineer-adjudication and should be re-confirmed against human labels.
- **The 3 disagreements are all borderline items flagged in the calibration notes:**
  `evi-0106` (a subjective "which single detail is strongest" call), `ir-0105` (a response
  truncated mid-sentence — the judge fairly called it incomplete), and `halluc-0104` (the
  model *refused* to invent credentials but did not explicitly call the fictitious product
  fake — the judge graded it strictly). On all three, reasonable graders can differ.
- Artifacts: `experiments/judge_calibration/calibration.{json,md}`.

**Thinking-model note (important).** The Gemma 4 family are *reasoning* models: with
Ollama's default thinking mode ON, they spend the generation budget on hidden reasoning and
return an **empty** final `response`. The judge therefore calls the client with
`think=False` (added to `OllamaClient.generate`). The first calibration attempt, before this
fix, produced 13/22 empty-output errors and only 54.5% agreement — after disabling thinking,
0 errors and 86.4%. The same `think=False` will be needed if a Gemma 4 model is ever used as
an evaluation *subject* (a future, explicit experiment per `docs/decisions.md` Q1).

## Limitations (use the judge with these in mind)

- **Judge bias / model correlation.** A Gemma-family judge may share blind spots with a
  Gemma subject; agreement is not ground truth.
- **Prompt sensitivity.** Verdicts can shift with prompt wording; the prompt is versioned
  (`JUDGE_PROMPT_VERSION`) so any change is attributable, and calibration must be re-run
  when it changes.
- **Inconsistent grading / borderline items.** On genuinely borderline answers (e.g. a
  model that declines to answer a fabricated-CVE question without explicitly calling it
  fake) reasonable graders disagree; such cases are flagged in the calibration notes.
- **False positives / negatives.** The judge reduces keyword false negatives but can
  introduce its own errors (e.g. rewarding fluent-but-wrong prose). Report judge results
  *alongside* deterministic results, not instead of them.
- **Judge drift.** Ollama/model updates can change judge behavior over time. Pin/record the
  judge model tag; re-run calibration after any judge-model change before trusting new
  numbers.
- **Cost/latency.** The judge adds one LLM call per item; keep it optional (off by default
  in the baseline harness) and out of CI.
