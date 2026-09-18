# Model card — Gemma-Cyber (APP-GATE)

**Status:** this file is the honest capability boundary for anything served
today. It is **not** a claim that a specialized cybersecurity model has passed
evaluation.

Two gates stay distinct:

- **APP-GATE** — the FastAPI service can be exposed on a single host behind TLS
  with Auth0, fail-closed config, and operator smoke evidence.
- **MODEL-GATE** — a registry entry may be stage `production` with
  `passed_eval=true` only after it clears
  [`configs/eval_success_criteria.md`](../configs/eval_success_criteria.md).
  That has **not** happened.

## What is served

| Field | Value |
|---|---|
| Default serve tag | `gemma3:4b` (Ollama) |
| Base weights | Google Gemma 3 4B instruction-tuned (Gemma Terms of Use) |
| Quant / runtime | Ollama GGUF (typically ~Q4) on a private Ollama process |
| Registry stage | `evaluated` (baseline scorecard exists) |
| `passed_eval` | **false** |
| Production alias | **none** — zero registry entries are `production` |

This is an untuned general instruction model used as a **defensive Q&A
assistant**. It has no tools, no agents, no RAG, and no ability to scan or
exploit systems.

## Frozen Ollama baseline (`gemma3:4b`, benchmark_v2)

Measured 2026-08-24, `experiments/baseline_gemma3-4b_v2/`:

| Split | n | pass_rate |
|---|---:|---:|
| test | 45 | 0.933 |
| hallucination traps (test) | 3 | **0.000** (all three asserted confidently) |

Hallucination pass_rate 0.000 on the frozen Ollama v2 traps means the base
model **failed** those items, not that it is hallucination-free.

## Fine-tune candidate (not served, not promoted)

`gemma3-cyber:v0.2` / experiment **exp-002r** (2026-08-27), candidate of record
= end-of-epoch-3 fused MLX 4-bit
`fused_model_sha256=1ea70da8a68526b1abbffb1eff6738319f961550348ee90ecea0bedd497ef702`.

From [`experiments/exp-002r-gemma3-cyber-v0.2/RESULTS.md`](../experiments/exp-002r-gemma3-cyber-v0.2/RESULTS.md):

| Check | Result |
|---|---|
| v2 test overall 0.956 (bar ≥ 0.913) | PASS |
| v2 hallucination n=3 0.333 (Δ vs MLX base = 0) | floor met, no gain |
| v3 `attack_mapping` 0.000 vs base 0.250 | **FAIL** |
| Kerberoasting trap still emits **T1060** (correct: T1558.003, Credential Access TA0006) | **FAIL** |
| Verdict | **DOES NOT PASS** |

Registry: stage `experimental`, `passed_eval=false`. **Do not** set
`passed_eval=true` or stage `production` on this tag.

## APP-GATE model waiver (2026-09-10)

Production may serve the unevaluated-for-promotion **base** model `gemma3:4b`
so the application stack can be operated. Operators and users must treat every
answer as unverified:

- Do not market this deployment as a “specialized cyber model.”
- Do not treat ATT&CK IDs, CVEs, or protocol claims as authoritative.
- Verify before acting. The Kerberoasting→T1060 confusion is a known failure
  of both the base model and v0.2.
- Gemma weights remain subject to the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and Prohibited Use
  Policy, independent of this repository’s Apache-2.0 code license.

This waiver does **not** satisfy MODEL-GATE. MODEL-GATE requires a new
experiment (sft_v0.3 / exp-003 or later) that clears the frozen bars.

## Intended use

Education, defensive security discussion, authorized testing in CTF/lab
environments, and analysis of **owned** systems. Not for live targeting,
exploit execution, or autonomous tooling.

## The `gemma4` terminal agent does not change this card

The local agent (`docs/agent.md`) ships in the same package. Building it is
**not** evidence about any model and does not promote anything:

- the agent's default model remains `gemma3:4b`, the unpromoted base model;
- `gemma3-cyber:v0.2` remains experimental — it failed MODEL-GATE and no agent
  work re-ran that evaluation;
- the agent's system policy tells the model its answers are unverified and
  names the known Kerberoasting-as-T1060 failure;
- measured agent behaviour (`docs/agent.md`) shows `gemma3:4b` calling tools
  unreliably, which is consistent with this card, not an argument against it.
