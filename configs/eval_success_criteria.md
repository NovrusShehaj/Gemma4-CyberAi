# Pre-Registered Evaluation Success Criteria

> **Status:** pre-registered on 2026-08-24, **before** any fine-tuning experiment exists.
> Its purpose is to fix the bar a specialized model must clear *before* we see its
> results, so that noise cannot later be rationalized as success (PROJECT_PLAN.md §18).
> Thresholds here are **frozen**. They may change only by writing a new, dated version of
> this file with an explicit rationale — never silently after seeing an experiment's
> numbers.

This is an **evaluation-infrastructure** artifact. **No training has occurred.** Nothing
here claims the model has improved; it defines how we will decide *whether* it did. The
thresholds below are calibrated to the freshly measured Benchmark v2 baseline (§1), which
existed before any candidate model.

---

## 1. Baseline definition

- **Model:** `gemma3:4b` via Ollama (GGUF, ~Q4), the established project baseline.
- **Benchmark:** `data/evaluation/benchmark_v2.jsonl` (112 original items, frozen).
- **Inference config:** temperature `0`, seed `0`, `num_predict=512`, baseline system
  prompt (`harness.py::BASELINE_SYSTEM_PROMPT`). Deterministic and reproducible.
- **Scoring:** the deterministic scorers in `src/gemma_cyber/evaluation/scorers.py`
  (`mcq`, `keyword`, `insufficient_evidence`, `hallucination`). These are transparent
  proxies, not ground truth; an LLM-judge scorer (P1) may later reduce keyword
  false-negatives, but any comparison must use the **same** scorer version on both models.
- **Splits scored separately:** `dev` (67 items) and `test` (45 items).

**Frozen Benchmark v2 baseline (`gemma3:4b`), measured 2026-08-24:**

| Split | n | pass_rate | mean_score |
|---|---:|---:|---:|
| dev  | 67 | 0.836 | 0.766 |
| test | 45 | **0.933** | 0.841 |

Baseline on the two "uncertainty behavior" trap categories:

| Category | dev n | dev pass | test n | test pass |
|---|---:|---:|---:|---:|
| hallucination | 5 | **0.000** | 3 | **0.000** |
| insufficient_evidence | 5 | 0.600 | 3 | 1.000 |
| **combined trap** | 10 | 0.300 | 6 | 0.500 |

Full per-category baseline is committed under
`experiments/baseline_gemma3-4b_v2/{dev,test}/scorecard.md`.

### Two facts that shape every threshold below

1. **Overall `test` is near-saturated at 0.933.** Only **6.7 pp** of headroom remain, so a
   large overall gain is not achievable and overall pass_rate is a *weak* discriminator.
   The non-trap categories (mostly `mcq`/`keyword`) are already largely solved by the base
   model. Overall is therefore used as a **do-no-harm guard**, not the primary bar.
2. **Hallucination resistance is 0.000 on both splits.** The base model confidently
   engages fabricated CVEs/tools/artifacts every time. This is the project's clearest,
   highest-headroom weakness (K3) and is where specialization (SFT for
   uncertainty/refusal behavior) can actually move the needle. It is the **primary** bar.

---

## 2. Metrics

**Primary metric — hallucination resistance.** Pass_rate on the `hallucination` category,
where the base model scores 0.000. Because each split's trap subset is small (test n=3,
dev n=5), this is measured two ways and **both** must move (see §4): over the full
category (8 items) and over the held-out `test` subset (3 items).

**Guard metric — overall `test` pass_rate.** Reported and constrained not to regress
(§4.2). Any overall gain above baseline is a reported bonus, not the bar.

**Secondary metrics.**
- Combined `hallucination` + `insufficient_evidence` pass_rate ("uncertainty behavior";
  test baseline 0.500, n=6).
- Overall `test` `mean_score` (partial-credit signal; baseline 0.841).
- Per-category `test` pass_rate (regression detection).
- `dev` results are reported for context but are **not** the bar (dev is used during
  iteration and is therefore optimistically biased).

Deltas in pass_rate are always reported in **percentage points (pp)**, never as a
percentage-of-a-percentage. Example: 0.000 → 0.667 is **+66.7 pp**, not "+67%".

---

## 3. Test-set policy

`test` is **held out**. During iteration you may use `dev` freely, but you may not tune
prompts, scorers, thresholds, or training data against `test` content or `test`
responses, and you may not re-run `test` repeatedly to pick a "best" config. `test` is
evaluated for the final, pre-registered comparison only. Full permitted/prohibited list:
`data/evaluation/README.md`.

---

## 4. Success thresholds (frozen)

A candidate model (fine-tuned, prompted, or RAG-augmented) **passes the milestone bar**
only if **all three** hold, evaluated with the identical harness/scorers as the baseline.

### 4.1 Hallucination resistance improves (primary)

- **Held-out:** `hallucination` pass_rate on the **`test`** subset improves by
  **≥ +33 pp** over baseline (baseline 0.000 → candidate ≥ 0.333, i.e. ≥ 1 of 3 test
  traps resisted), **and**
- **Full category:** `hallucination` pass_rate over **all 8** hallucination items
  improves by **≥ +50 pp** (baseline 0.000 → candidate ≥ 0.500).

Both are required so the gain is corroborated beyond the tiny 3-item held-out subset and
is not a single-item fluke. Rationale: baseline is 0.000 everywhere on this category, so
any real acquisition of "flag the fabrication" behavior clears these floors; a model that
merely produces plausible prose will not.

### 4.2 Overall `test` does not regress (do-no-harm guard)

- `test` overall pass_rate ≥ **0.913** (baseline 0.933 − 2 pp tolerance for single-item
  noise; ⌊0.02 × 45⌉ ≈ 1 item).

Overall is near-saturated (ceiling 1.000), so this guards against the common failure mode
where teaching refusal behavior degrades the categories the base already handles
(catastrophic forgetting). An overall gain, if any, is reported as a bonus.

### 4.3 No unacceptable category regression

- No individual `test` category drops by more than **one item's worth** of pass_rate below
  its baseline (`1 / n_category`, the granularity floor for small categories), **and**
- the `insufficient_evidence` category (test baseline 1.000) may **not** regress at all
  (it is a behavior we already have and must keep while improving hallucination).

> Threshold values (+33 pp / +50 pp hallucination; 0.913 overall guard) are **derived from
> this Benchmark v2 baseline** and its per-split sample sizes, and are pre-registered here
> before any candidate exists. They are frozen per the header.

---

## 5. Interpreting outcomes

- **Pass (all three met):** record an honest positive verdict; version/tag the candidate
  (e.g. `gemma3-cyber:v0.1`); keep the full experiment manifest; result must be
  reproducible from it.
- **Fail the hallucination bar (4.1):** the intervention did **not** fix the targeted
  weakness. Record the negative result — a valid outcome, not a process failure
  (PROJECT_PLAN.md §2, G4). Do **not** move the goalposts.
- **Hallucination improves but overall regresses (4.2/4.3 fail):** treat as **not passing**
  — a gain bought by catastrophic forgetting. Analyze which categories dropped and iterate
  (change data **or** config, one variable at a time — §19).
- **Mixed / within noise:** given the small per-category `test` n, always report the delta
  **with** a bootstrap 95% CI or a paired McNemar test, and call a sub-CI delta "no
  measurable effect" rather than a trend. If overall is already at 0.933, do not chase
  overall — the informative movement is in the hallucination category.

---

## 6. Reproducibility requirements

Any comparison cited against this bar must record, in its experiment directory:

- model identifier + exact tag/quant; benchmark version (`benchmark_v2`) + split;
- inference config: temperature, seed, `num_predict`, system prompt;
- scorer version (git commit) — the **same** on baseline and candidate;
- environment info (already captured: `host.platform`, `host.python`, `duration_seconds`);
- raw per-item responses (so re-scoring is possible without re-inference).

The baseline this document is calibrated against lives in
`experiments/baseline_gemma3-4b_v2/` and was produced by:

```bash
python scripts/run_baseline.py --benchmark data/evaluation/benchmark_v2.jsonl \
    --split dev  --out experiments/baseline_gemma3-4b_v2/dev
python scripts/run_baseline.py --benchmark data/evaluation/benchmark_v2.jsonl \
    --split test --out experiments/baseline_gemma3-4b_v2/test
```

---

## 7. Addendum (2026-08-25) — Benchmark v3 targeted instrument

> **The §1–§6 v2 criteria above are FROZEN and unchanged.** This addendum *adds* a second,
> non-substitutable instrument. It does not relax any v2 threshold.

`benchmark_v3.jsonl` (47 items, `factual`/`mcq`/`hallucination`/`keyword`/
`insufficient_evidence`) targets the *exact* v0.1 failure that v2 cannot see: exact ATT&CK
IDs (Kerberoasting = **T1558.003**, not T1060/T1068), false premises, and protocol
mechanics. It uses the new **`factual`** scorer (required IDs + **forbidden** wrong IDs →
hard fail), so a fluent answer containing a wrong ID cannot earn credit.

**Not comparable to v2.** v3 uses a different scorer and a deliberately harder distribution.
Do **not** compare a v3 number to a v2 number. Each is scored against **its own base
`gemma3:4b` baseline**, measured as the control arm of the same experiment (no base-on-v3
number pre-dates this addendum, so it is measured, not assumed).

**v3 pre-registered targets (for exp-002 and later), evaluated on `test`:**
- `attack_mapping` (factual + mcq) pass_rate ≥ **+40 pp** over base-on-v3, **and** the
  flagship item `v3-attack-kerberoasting-t1060-trap` flips **fail → pass**.
- `false_premise` pass_rate ≥ **+33 pp** over base-on-v3.

**Guard:** the v2 do-no-harm criteria (§4.2–§4.3) still apply unchanged and are evaluated on
`benchmark_v2` `test`. A v3 gain that comes with a v2 regression is **not** a pass.

Rationale for setting these before the candidate exists: the base model demonstrably
hallucinates the exact IDs, so any genuine acquisition of exact-fact behavior clears these
floors, while plausible-prose output (which the `factual` scorer hard-fails on a wrong ID)
will not. Thresholds are frozen per the header; change only via a new dated addendum.
