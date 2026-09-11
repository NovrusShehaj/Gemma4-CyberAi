# Gemma4-CyberAI Training & Production Roadmap

> Confidence labels used throughout: **[Verified]** = confirmed from repo or primary sources; **[Estimate]** = engineering estimate; **[Recommendation]**; **[Verify]** = `Verification required — do not treat this as confirmed.`

## 1. Executive Recommendation

**Keep the current base (`google/gemma-3-4b-it`) as your fast/local tier, re-run and evaluate the lost exp-002 fine-tune first, and add a 14B–32B flagship tier (Qwen3-14B/32B recommended, Apache-2.0) fine-tuned on the same dataset.** Do not switch base models before one valid fine-tune has been measured — today the registry's "production" entry is still the un-fine-tuned base, and `gemma3-cyber:v0.2` artifacts were lost before evaluation. That is the single highest-value, lowest-cost action and it is P0.

Key structural facts driving this plan:

- **Unsloth cannot train on your Mac.** Unsloth training kernels require NVIDIA CUDA; Apple Silicon/MPS training is not supported (in-progress per Unsloth maintainers). **[Verified]** Your realistic options are: (a) Unsloth on a cloud GPU (Colab/RunPod), or (b) MLX (`mlx-lm` LoRA) locally on the M3 Max. Your existing pipeline (transformers + PEFT/TRL + bitsandbytes, Colab T4) is already a valid CUDA path; Unsloth is an optimization of it, not a prerequisite.
- **Your production surfaces are ahead of your model.** FastAPI + Auth0, Docker, staged model registry, eval harness with hallucination/judge scorers, contamination checks, and 156 tests already exist. The open item is model quality and dataset scale (277 SFT examples is far too small).
- **Fine-tuning alone will not reach "production-level."** Production readiness must be demonstrated through the evaluation gates in §14; the plan treats evaluation as the arbiter, not training loss.

Recommended end-state: a multi-model registry — a 4B cyber fine-tune for local/low-latency/on-device use, and a 14B–32B flagship for server use — both trained on the same dataset, compared under the same benchmark, promoted through your existing experimental→evaluated→candidate→production gates.

## 2. Assumptions & Unknowns

**Known from repo (verified):**

- Base model: `google/gemma-3-4b-it`; QLoRA 4-bit NF4, bfloat16 compute; export via llama.cpp → GGUF Q4_K_M.
- Training stack: transformers + PEFT/TRL + bitsandbytes (pinned versions), `scripts/train_qlora.py`, Colab T4 notebook.
- Data: `sft_v0.2.jsonl` (277 examples, 15 task types), contamination checker, ATT&CK fact registry (`security_facts.json`).
- Eval: benchmark_v2 (dev/test split) + benchmark_v3 (47 items), hallucination/insufficient-evidence/factual scorers, LLM-judge with judge≠subject at temp 0, pre-registered criteria.
- Serving: Ollama + FastAPI `/v1` + Auth0 JWT, Docker, CI security scanning, staged registry.

**Unknown / conditional assumptions:**

| Unknown | Assumption made | How the recommendation changes |
|---|---|---|
| Whether exp-002 artifacts are recoverable | Not recoverable; re-train from the saved config + manifest | If recoverable, skip re-training and go straight to eval |
| Long-term intended serving environment (Mac-only vs server) | Hybrid: Mac for dev/local, server/colocated GPU for flagship | Mac-only → cap flagship at 14B; server GPU available → 32B comfortably |
| Judge model (`gemma4:26b` referenced in repo) availability | Judge runs where hardware permits; judge≠subject invariant maintained | If judge must run locally on Mac, use a verified 27B-class model instead; **Verify — do not treat "Gemma 4 26B" as confirmed; verify the exact model id on Ollama/HF** |
| Budget for cloud GPU hours | Modest (Colab-class, occasional A100/L4 hours) | Larger budget → full-FT or 32B QLoRA on cloud becomes the flagship path |
| Licensing intent (open distribution vs internal use) | Internal use | Open distribution restricts model choice to Apache-2.0/MIT bases (§11) |

## 3. Current Model Assessment

**[Verified, from repo]** Gemma4-CyberAI today is `gemma3:4b` (base) in production with a 0.84 baseline pass rate; the cyber fine-tune `v0.2` is "experimental" and unproven.

- **Capacity is the ceiling, not your pipeline.** A 4B model can learn format, task framing, refusal behavior, and a modest amount of security vocabulary; it cannot reliably absorb the breadth of CWE/CVE/cloud/IAM/crypto reasoning at professional depth. Expect dataset scaling to hit diminishing returns on this base quickly (roughly after 1–3k high-quality examples **[Estimate]**).
- **Strengths of the current setup worth preserving:** chat-format SFT masking with the same engine used in production (train/serve consistency), pre-registered success criteria, contamination checking, judge calibration, staged promotion. Most cybersecurity-LLM projects fail on exactly these; yours already has them.
- **Weaknesses:** dataset too small (277 examples); no measured fine-tune; no external benchmark cross-check; no RAG, so the model's CVE/CWE knowledge is frozen at training time; eval set (45–112 items) is too small to detect small effect sizes — wide confidence intervals.

**Verdict:** strategy is sound; the model is undersized for the ambition, and the first fine-tune needs to be completed and measured before any architectural decision.

## 4. Recommended Training Strategy

Order = the sequence in which to apply, given your state.

| Technique | Purpose | Advantages | Disadvantages | Compute | Use for this project? | Order |
|---|---|---|---|---|---|---|
| **SFT (LoRA/QLoRA)** | Teach task format, security reasoning patterns, remediation style, abstention | Cheap, stable, well-understood; your infra exists | Can't add much raw knowledge; small data overfits fast | 4B QLoRA: Mac via MLX or any cloud GPU; 14–32B QLoRA: 24–48 GB GPU **[Estimate]** | **Yes — core method** | 1st |
| **Synthetic data generation** | Scale 277 → thousands of secure/insecure pairs, remediations, explanations | Controls quality/distribution; fills category gaps | Quality is the whole game; needs verification loop and dedup; license care with source code | Generation runs on strongest available model; verification is local | **Yes — before SFT scaling** | 1st (with SFT) |
| **High-quality general replay data** | Prevent catastrophic forgetting | Small amount preserves coding/reasoning | Wrong mix dilutes specialization | Same as SFT | **Yes — mix ~20–30% general/coding **[Estimate, tune on dev set]** | 1st |
| **RAG (retrieval over CVE/CWE/standards)** | Fresh, citable, factual knowledge | Halves the hallucination surface; knowledge refreshable without retraining | Infrastructure cost; retrieval quality becomes a failure mode; prompt-injection risk in retrieved text | CPU-class vector DB fine | **Yes — highest ROI after SFT** | 2nd |
| **Preference optimization (DPO/KTO)** | Better style, calibrated confidence, prefer-abstain-over-invent | No reward model needed; uses your judge/eval to build pairs | Amplifies SFT errors if SFT is weak; pairs are labor-intensive | Cheap (LoRA rank on top of SFT) | **Yes — but only after SFT ≥ gate** | 3rd |
| **Rejection sampling / self-distillation** | Generate high-quality SFT data from a stronger model (or the model itself + verification) | The standard way to scale verified data | Requires a verifier (your scorers + judge) to avoid laundering errors | Strong generator model required | **Yes — use a 32B-class generator + your fact registry as verifier** | 2nd |
| **Continued/domain-adaptive pretraining (CPT)** | Inject raw security corpus knowledge | Adds vocabulary/coverage | Data-hungry (needs 10s–100s of millions of tokens), risky for forgetting, hard to evaluate at your eval-set size | Full-FT-scale compute | **No for now** — revisit only if RAG+SFT plateau and a big clean corpus (e.g., curated CVE/NVD + standards text) exists | 4th (likely never) |
| **Full fine-tuning** | Max capacity adaptation | Best quality ceiling | Expensive, forgets more, needs real GPU + careful LR | 4B: 1×A100-class; 27B: multi-GPU | **No** — QLoRA/LoRA is sufficient at this data scale | — |
| **Tool use / function calling** | Let the model query live CVE feeds, scanners, sandboxes | Grounds answers in facts; kills stale-CVE hallucinations | Engineering cost; your repo explicitly deferred it | API/agent infra | **Yes — after RAG** | 4th |

**Recommendation:** SFT on scaled verified data → measure → RAG → DPO-style preference pass → tools. CPT and full FT are not justified by your data size or eval granularity.

## 5. Dataset Architecture

### 5.1 Categories and proportions (conceptual, not fake precision)

Build on `builder_v2.py` + `schema.py`. Target scale: **3,000–5,000 verified SFT examples** for the 4B tier before touching hyperparameters **[Recommendation]**.

| Category | Rough share | Notes |
|---|---|---|
| Secure vs insecure code pairs + remediation | 25–30% | Your highest-differentiation data; include CWE mapping and *why* |
| Vulnerability explanation / code review narratives | 15–20% | Review-style outputs with severity + evidence citations |
| Secure-by-construction generation (write secure code) | 10–15% | Complements detection; catches "can spot but can't write" gaps |
| Web/app security, authn/authz, API security | 10% | OWASP-aligned; high professional demand |
| Cloud/container/IAM/network defense | 10% | Terraform/K8s/IAM policy examples |
| Detection engineering + incident response | 5–10% | Log analysis, ATT&CK-aligned (you already have the fact registry) |
| Cryptography implementation review | 5% | Misuse patterns (ECB, weak RNG, cert validation) |
| Threat modeling + architecture | 5% | STRIDE-style structured outputs |
| Abstention / insufficient-evidence examples | 5% | **Critical for hallucination goals (§10)** — examples where the correct answer is "cannot determine from the provided code/context" |
| General programming + reasoning replay | 20–30% relative to cyber portion | Prevents overspecialization; take from permissive-license high-quality sets |

### 5.2 Pipeline (extend what exists)

1. **Collection/generation** — synthetic pairs generated by a 32B-class model, constrained to your 15 task types; human/automated verification against `security_facts.json` and a compiler/linter where possible. CTF material only as comprehension aids, small share, no operational exploit content.
2. **Cleaning & normalization** — dedupe by AST/exact hash; near-duplicate detection via MinHash/embedding similarity (drop >0.9 cosine **[Estimate]**); normalize code fences, language tags, chat template via the existing `to_gemma_chat_text()` path per base model.
3. **Quality filtering** — reject examples where: no verifiable ground truth; explanation contradicts the code; remediation introduces a new vuln; truncation at `max_seq_length`.
4. **Licensing** — extend `DATA_LICENSES.md`; only permissive/owned code; record provenance per example in the schema.
5. **PII/secrets removal** — regex + entropy-based secret scanning (reuse the gitleaks rules from CI) and a PII pass before anything enters training.
6. **Exploit-content controls** — keep payloads non-operational (defanged, conceptual); align with the repo's §24 safety gating; document the policy in `DATA_LICENSES.md`.
7. **Splits** — train/dev/test with test never used for iteration; **temporal split where CVE-year is a field** (train on ≤ cutoff year, test on after) to approximate real generalization; benchmark-contamination check via the existing `contamination.py` against every new eval item.
8. **Validation-set hygiene** — dev set used for LR/epoch selection only; test set touched once per release candidate.

## 6. Cybersecurity Curriculum

Staged ordering for *data mixing during training* (mixture curriculum, not hard phases — hard curricula with LoRA on small data rarely help **[Estimate]**; the practical version is "front-load the mixture toward fundamentals in epoch 1, uniform after" if you use a multi-epoch curriculum, otherwise a fixed uniform mix is fine):

1. Programming fundamentals → 2. Secure coding → 3. Vulnerability taxonomy (CWE) → 4. Web security → 5. Authn/authz → 6. API security → 7. Cryptography → 8. OS security → 9. Networking → 10. Cloud/container → 11. Code auditing → 12. Remediation → 13. Threat modeling → 14. Detection & response → 15. Security architecture → 16. Professional reporting.

**Why ordering helps:** early exposure to secure-vs-insecure *contrast pairs* on simple code teaches the discriminative skill (spot the flaw) before the integrative skills (reason across components, judge severity, write remediations). This reduces the failure mode where the model memorizes surface patterns ("if `md5` appears, say weak hash") instead of analyzing data flow — which is exactly what your hallucination/FP scorers should detect. Tail stages (13–16) force compositional outputs that can't be pattern-matched, acting as an anti-memorization pressure.

## 7. Unsloth Training Configuration

**Framework reality check [Verified]:** Unsloth requires NVIDIA CUDA for training; it does not run on Apple Silicon MPS today (Mac support in progress). Your repo currently uses transformers+PEFT+TRL on Colab — a valid path. Use Unsloth as a *speed/memory optimization on cloud GPUs* (it supports Gemma 3 4B/12B/27B and Qwen3 **[Verified via Unsloth model catalog]**), and **MLX (`mlx-lm` LoRA) as the local-training path on the M3 Max**.

### 7.1 Cloud/CUDA path (Unsloth or your existing PEFT stack)

Starting ranges, 4B class (your current base) — each value justified:

```yaml
# configs/training/qlora_gemma3_4b_v0.3.yaml (proposed deltas from v0.2)
max_seq_length: 2048        # v0.2's 1024 truncates code-review examples; check truncation stats first
lora_r: 32                  # 4B, ~1-5k examples: 16-32 is the useful band; higher memorizes, lower underfits
lora_alpha: 64              # keep alpha = 2*r (matches v0.2's 32/16 ratio convention)
lora_dropout: 0.05          # small data → light regularization; >0.1 hurts small-LoRA capacity
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # all-linear standard
packing: false              # keep false: you rely on loss masking per example; packing breaks clean masking semantics
per_device_batch_size: 2
gradient_accumulation_steps: 8   # effective 16 — stabilizes small-batch LoRA
learning_rate: 2e-4         # your v0.2 value is standard for 4-bit LoRA; sweep {1e-4, 2e-4} first
epochs: 2-3                 # watch dev loss; >3 on <5k examples almost always overfits
warmup_ratio: 0.03-0.05
weight_decay: 0.01
max_grad_norm: 1.0
optimizer: adamw_torch_fused (or adamw_8bit for memory)
lr_scheduler: cosine
precision: bf16 compute / 4-bit NF4 weight (double-quant)  # unchanged
gradient_checkpointing: true
checkpoint/eval: every ~200 steps on dev split; keep best-by-dev-loss, not last
```

**Scaling by model size [Estimate]:**

| Setting | ~4B | ~14B | ~27–32B |
|---|---|---|---|
| LoRA rank | 16–32 | 32–64 | 32–64 |
| LR (LoRA) | 1–2e-4 | 1–2e-4 | 5e-5–1e-4 |
| Effective batch | 16 | 16–32 | 32–64 |
| Epochs | 2–3 | 2–3 | 1–3 |
| GPU for QLoRA | ≥16 GB | ≥24 GB | ≥48 GB **[Estimate]** |

**Knob order when experimenting (one variable at a time, per repo §19):** LR → epochs (via dev loss) → rank → target-module set → data mixture. Do not change data and hyperparameters in the same run.

### 7.2 Local path (MLX on M3 Max)

`mlx-lm` LoRA on 4-bit quantized models is the verified-native Apple Silicon training route. **[Verified: MLX supports LoRA/full FT on Apple Silicon via unified memory; medium-high confidence on practical size limits]**

- Practical on 128 GB: QLoRA-style LoRA on 4-bit models up to ~30B-class (incl. MoE like Qwen3-30B-A3B) with reduced trainable layers (default 16) and small batch. 70B 4-bit LoRA is memory-feasible but slow — treat as "possible, not practical." **[Estimate]**
- Keep train/serve parity: train locally with MLX only for *experiments*; still produce GGUF for Ollama serving, or serve MLX-converted weights. Verify MLX→GGUF conversion fidelity for your chosen model before committing to it. **Verify per model — conversion support varies; do not treat this as confirmed for any specific checkpoint.**
- For your 4B tier, MLX LoRA locally is entirely practical and removes the Colab dependency for iteration. **[Recommendation]**

## 8. Experiment Matrix

Aligned with the repo's one-variable-per-experiment rule. Each experiment: pre-registered hypothesis in `docs/experiments/`, evaluated on benchmark_v2 dev split; only the winning lineage advances to the test split.

| Exp | Base | Dataset | Method | Key variable | Purpose / hypothesis |
|---|---|---|---|---|---|
| exp-002r | gemma-3-4b-it | sft_v0.2 (277) | QLoRA (existing config) | — (re-run) | **P0**: reproduce exp-002; produce artifacts + eval that were lost. Gate: beat 0.84 base pass rate |
| exp-003 | gemma-3-4b-it | sft_v0.3a (~1.5k scaled, same mixture recipe) | QLoRA, same HP | Data scale | Does 6× verified data lift dev pass rate ≥5 pts? |
| exp-004 | gemma-3-4b-it | sft_v0.3a | QLoRA, same HP | Replay mix (0% vs 25% general) | Does replay protect HumanEval/MBPP-class coding score without hurting cyber score? |
| exp-005 | gemma-3-4b-it | sft_v0.3a | QLoRA | LR 1e-4 vs 2e-4 | HP sensitivity; pick LR before scaling further |
| exp-006 | Qwen3-14B | sft_v0.3a | QLoRA (cloud) or MLX (local) | Base model | Does 3.5× capacity justify a second tier? Same eval, judge fixed |
| exp-007 | winner of 003–005 | sft_v0.3b (+abstention/negative examples) | SFT → DPO on judge-ranked pairs | Preference pass | Does DPO reduce hallucination/FP rate ≥20% relative? |

Kill criteria: any experiment that fails its pre-registered bar stops its lineage; no "tune until it passes."

## 9. Evaluation & Benchmarking

Reuse and extend the existing harness (`harness.py`, `scorers.py`, `judge.py`). Structure:

### 9.1 Internal suites (owned, contamination-controlled)
- Expand benchmark_v2/v3 to **≥200 test items** with dev/test separation **[Recommendation]** — 45 test items cannot distinguish small improvements (binomial CI at n=45 is ±~15 pts).
- Scoring dimensions (already partially in `scorers.py`): factual knowledge, secure code review, vulnerability detection, **false-positive rate on clean code (critical — include clean-code fixtures)**, severity assessment, explanation quality, remediation correctness, secure-code generation, authn/authz reasoning, cloud/network reasoning, threat modeling, IR reasoning, instruction following, hallucination rate, abstention calibration, refusal behavior.

### 9.2 External benchmarks (cross-check, contamination-aware)
- **CyberMetric** (10k MCQ), **SecQA v1/v2**, **CTIBench** (CTI tasks), **CyberSecEval 2** (insecure-code detection / prompt injection). All verified to exist. Run zero-shot and report alongside internal metrics; use CyberSecEval's insecure-code subset for the FP-rate cross-check. Keep all of these out of training data (enforced by `contamination.py`).

### 9.3 Code-security ground truth
- Intentionally vulnerable isolated fixtures with known CWE ground truth (curated from public permissively-licensed vulnerable-app corpora, or generated+verified), one vulnerability per fixture, clean twins for FP measurement. Never real user code.

### 9.4 Human evaluation
- Small expert panel (even 1–2 reviewers) rating 30–50 stratified outputs per release candidate on: correctness, completeness, actionability, hallucination, calibration. Report inter-rater agreement. Automated scores alone are not sufficient for the "production" claim.

### 9.5 Regression protocol
- Every candidate runs the full suite with the same judge, temp 0, and pinned prompts; results append to a versioned leaderboard; promotion requires §14 gates.

## 10. Hallucination Reduction

Policy (make it explicit in the system prompt and enforce in training data):

> When reliable evidence is unavailable, state uncertainty instead of inventing a vulnerability, CVE, API, library behavior, command, or remediation.

Mechanisms, in order of expected ROI:

1. **Abstention training data (highest ROI)** — the 5% abstention category (§5.1): examples where the correct output is "cannot determine — the code does not show X" or "I don't have verified information on this CVE." Directly trains the insufficient-evidence behavior your scorer already measures.
2. **RAG over verified knowledge** — CVE/CWE/NVD + OWASP + your `security_facts.json`; answers grounded in retrieved text with **structured citations** ("per CWE-787, NVD-CVE-2024-…"). Stale-knowledge hallucinations become retrieval problems, fixable without retraining.
3. **Negative examples** — chosen/rejected pairs where rejected invents a CVE or overclaims severity; consumed via DPO (exp-007).
4. **Fact-verification datasets** — MCQ-style items built from `security_facts.json` so factual recall is trained and tested on the same verified registry.
5. **Calibration & uncertainty** — train the model to express confidence; measure ECE-style calibration on the MCQ suites; DPO pairs rewarding correct "I'm not sure."
6. **Tool-assisted answers** — live CVE lookup via tool call instead of memory (Phase 6).
7. **Regression gates** — hallucination/FP metrics are hard release gates (§14), so a regression blocks promotion regardless of other gains.

## 11. Alternative Base Models

Existence/specs verified via model cards and Unsloth catalog; **2026-era families beyond those below (e.g., "Gemma 4", Qwen3.5+) appeared only in a catalog listing — treat as unconfirmed [Verify] until checked on HF model cards.**

| Model | Parameters | Architecture | License | Cyber potential | Coding | Local inference (M3 Max 128 GB) | Local FT feasibility | Recommended quant | Unsloth support | Confidence |
|---|---:|---|---|---|---|---|---|---|---|---|
| google/gemma-3-4b-it (current) | 4B | dense, multimodal | Gemma license | moderate | good for size | trivial (Q4/Q8) | MLX LoRA easy | Q4_K_M / Q8 | yes | High |
| google/gemma-3-27b-it | 27B | dense | Gemma license | strong | strong | Q4 ≈ 17–20 GB — comfortable | MLX LoRA feasible (4-bit, reduced layers) | Q4_K_M | yes | High (specs); FT limits: Estimate |
| Qwen/Qwen3-14B | 14B | dense | **Apache-2.0** | strong | very strong | Q4 ≈ 9–10 GB | MLX LoRA easy | Q4_K_M / Q6 | yes | High (specs); sizes: Estimate |
| Qwen/Qwen3-32B | 32B | dense | **Apache-2.0** | very strong | very strong | Q4 ≈ 20 GB — comfortable | MLX LoRA feasible | Q4_K_M | yes | High (specs); sizes: Estimate |
| Qwen/Qwen3-30B-A3B (MoE) | 30B total / ~3B active | MoE | Apache-2.0 | strong | strong | fast inference for class | MLX LoRA very practical on 128 GB | Q4 | yes (Coder variant listed) | High (specs); Estimate on FT |
| microsoft/phi-4 | 14B | dense | **MIT** | moderate-strong | strong | Q4 ≈ 9 GB | MLX LoRA easy | Q4_K_M | yes | High; 16K context noted — a limitation for long code review |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 32B | dense | MIT (per base) | strong (reasoning-heavy) | strong | Q4 ≈ 20 GB | MLX LoRA feasible | Q4_K_M | yes | High; verify per-base license on card |
| meta-llama/Llama-3.3-70B-Instruct | 70B | dense | Llama Community | very strong | very strong | Q4 ≈ 40–45 GB — fits, slow | MLX LoRA: memory-feasible, impractically slow | Q4_K_M | yes | High (specs); Estimate |

Notes: memory figures are GGUF-4-bit ballparks **[Estimate]** — verify against each model's official quant sizes before committing. If you intend to redistribute weights, the Gemma and Llama licenses carry conditions; Apache-2.0 (Qwen3) and MIT (Phi-4) are the frictionless choices.

## 12. M3 Max / 128 GB Feasibility

**128 GB unified memory ≠ 128 GB CUDA VRAM.** macOS caps GPU-allocatable memory (default ~75% of RAM, ~96 GB here) and MPS/training throughput on Apple GPUs is far below a data-center GPU. **[Verified]**

### A. Local inference (llama.cpp/Ollama/MLX)

| Precision | 4B | 14B | 27–32B | 70B |
|---|---|---|---|---|
| BF16/FP16 | ✅ | ✅ (~28 GB) | ✅ (~55–60 GB) | ⚠️ ~140 GB — **no** |
| 8-bit | ✅ | ✅ | ✅ | ⚠️ ~75 GB — borderline **[Estimate]** |
| 6-bit | ✅ | ✅ | ✅ | ✅ ~55 GB, slow |
| 5-bit | ✅ | ✅ | ✅ | ✅ |
| 4-bit | ✅ (fast) | ✅ (fast) | ✅ (~20 GB, usable speed) | ✅ fits (~40 GB) but token rate low |

All plus KV-cache and OS headroom: reserve 10–20 GB for a 27–32B model with long context; cap context via KV-cache quantization if needed. **[Estimate]**

### B. Fine-tuning

- **What works locally:** MLX LoRA on 4-bit models up to ~30B-class (reduced trainable layers, small batch). Your 4B and a 14B flagship are both practical locally; 27–32B is feasible with patience. **[Estimate, medium-high confidence]**
- **What does not work locally:** **Unsloth training** (CUDA-only, verified); any full fine-tuning ≥14B (optimizer state memory); large-batch long-context LoRA on 70B.
- **When to use NVIDIA cloud:** Unsloth-speed runs, QLoRA on 27–32B with full target modules, anything with real wall-clock deadlines, and DPO runs on 14B+.
- **When MLX is the right call:** rapid iteration on 4B/14B, privacy-sensitive data that must not leave the machine, zero cloud cost.
- **Framework-vs-hardware limitations to respect:** MPS lacks some fused kernels and bitsandbytes NF4 paths used by your current stack — your pinned CUDA-oriented `requirements-train.txt` will not run on the Mac as-is; the Mac path is MLX, not transformers+bitsandbytes. **[Verified]**

## 13. Production Architecture

Extend the existing stack (FastAPI/Auth0/Docker/Ollama/registry) rather than replacing it:

```
Client → FastAPI /v1 (Auth0 JWT, rate limit, security headers)
       → Policy layer (input/output controls, injection filters)
       → Orchestrator
           ├─ Model router (registry: 4B fast tier / 14-32B flagship)
           ├─ RAG service (vector DB over CVE/CWE/NVD/OWASP + security_facts.json)
           │    └─ nightly refresh job (NVD API) + diff alerts for new CVEs
           ├─ Tool sandbox (optional, Phase 6): ephemeral container for code exec/scanners
           └─ Ollama runtime(s) with pinned GGUF (SHA-256 in manifest)
Observability: structured logs (JSON), request tracing IDs, latency/token metrics,
               per-model error rates → dashboards
Lifecycle: registry gating (experimental→evaluated→candidate→production),
           eval-gated promotion, canary (route 5-10% traffic to candidate),
           one-click rollback to previous registry entry
Integrity: signed model artifacts, SHA-256 in run manifests, verify-on-load
Secrets: existing env/secret management; no secrets in Modelfiles
RAG hygiene: untrusted retrieved text wrapped as data (never instructions),
             strip/escape injection patterns, cite sources in output
```

Key additions vs today: RAG service with refresh pipeline, model router for the two-tier registry, canary routing, and the sandboxed tool path. Everything else exists in the repo already.

## 14. Production Readiness Gates

A candidate is promoted only when **all** gates pass on the frozen test split + external suites + human review. Fine-tuning alone establishes none of these.

| Gate | Threshold (initial; calibrate after exp-002r establishes real numbers) |
|---|---|
| Cyber factual accuracy (MCQ suites) | ≥ base model + 5 pts, and ≥ 80% absolute |
| Internal benchmark pass rate (test) | beats production base with statistical margin (n≥200) |
| False-positive rate on clean code | ≤ 10% **[Estimate — set empirically after exp-002r]** |
| Hallucination score (existing scorer) | ≤ dev-set SFT optimum + no test regression |
| Remediation correctness (human-rated) | ≥ 80% rated correct/actionable |
| General capability regression | coding/reasoning suites within 2 pts of base |
| Abstention behavior | abstains on ≥ 90% of insufficient-evidence fixtures |
| Refusal/safety | passes safety scenario set; no increase in harmful-output rate |
| Latency (4B tier, M3 Max, 4-bit) | p95 first-token and full-response within documented SLO **[set from measured baseline]** |
| Memory | within documented runtime envelope incl. KV cache |
| Stability | 24 h soak in Docker compose, zero crash/OOM, error rate < 1% |
| Reproducibility | run manifest (pip freeze, data SHA-256, GGUF SHA-256, llama.cpp commit) regenerates identical eval numbers |
| Security | CI security scans clean; artifact signature verified at load |

## 15. Phased Implementation Roadmap

### Phase 0 — Baseline
- **Objective:** a trustworthy measured baseline.
- **Actions:** re-run exp-002 from saved config (exp-002r); persist artifacts immediately; run full eval suite (v2 dev + external MCQ); record base-model numbers in the leaderboard.
- **Deliverables:** exp-002r eval report; measured thresholds for §14; updated registry.
- **Validation:** eval numbers reproducible from manifest; base pass rate confirmed (~0.84).
- **Stop/Go:** Go regardless — this is measurement. Go/No-Go on *continue-with-4B* is decided in Phase 3.

### Phase 1 — Data Engineering
- **Objective:** 1.5–3k verified examples (sft_v0.3).
- **Actions:** extend builder_v2; synthetic generation + verification loop; near-dup detection; PII/secrets scan; licensing updates; abstention category; general replay set; temporal split; contamination check vs all benchmarks.
- **Deliverables:** sft_v0.3 train/dev; updated DATA_LICENSES.md; data quality report.
- **Validation:** zero contamination hits; spot-check 5% sample by hand; dedup metrics.
- **Stop/Go:** proceed if ≥1k examples pass verification; else fix generation loop first.

### Phase 2 — Initial SFT
- **Objective:** measured cyber fine-tune on the 4B (exp-003..005 lineage).
- **Actions:** run matrix §8 on MLX (local) and/or Colab; one variable per run; best checkpoint by dev loss → test split once.
- **Deliverables:** trained GGUF + manifest; eval report.
- **Validation:** beats exp-002r baseline on dev, holds on test.
- **Stop/Go:** **No-Go → escalate base model** if 4B plateaus below +3 pts despite data scaling; Go if above.

### Phase 3 — Evaluation
- **Objective:** rigorous, comparable measurement.
- **Actions:** expand benchmark to ≥200 test items; integrate CyberMetric/SecQA/CTIBench runs; calibration + FP-rate analysis; first human eval round.
- **Deliverables:** expanded suite; leaderboard; gate calibration memo.
- **Validation:** CIs narrow enough to detect 5-pt differences; judge calibration re-verified.
- **Stop/Go:** decide 4B-only vs two-tier here (feeds §18).

### Phase 4 — Dataset Repair
- **Objective:** fix measured weaknesses.
- **Actions:** error analysis by task type; targeted data patches (e.g., add FP-reducing clean-code examples, remediation depth); rebuild sft_v0.4; re-run winner config.
- **Deliverables:** sft_v0.4 + delta eval.
- **Validation:** targeted metrics improve without regressions elsewhere.
- **Stop/Go:** iterate max 2 repair cycles; then move on regardless.

### Phase 5 — Advanced Post-Training
- **Objective:** preference optimization for calibration/hallucination.
- **Actions:** build chosen/rejected pairs via judge + scorers from model outputs; DPO/KTO on winner; abstention emphasis (exp-007).
- **Deliverables:** preference-tuned model; hallucination/FP delta report.
- **Validation:** hallucination/FP reduced ≥20% relative **[Estimate]** with no quality regression.
- **Stop/Go:** keep preference-tuned model only if gates improve; else ship SFT winner.

### Phase 6 — RAG/Tools
- **Objective:** grounded, fresh knowledge.
- **Actions:** stand up RAG service (§13) over CVE/CWE/OWASP + fact registry; nightly NVD refresh; prompt-injection defenses for retrieved text; (optional) CVE-lookup tool.
- **Deliverables:** RAG-enabled endpoint; retrieval eval set; citation format in outputs.
- **Validation:** hallucination on stale-CVE test items drops vs non-RAG; citation present in ≥95% of knowledge answers **[Estimate]**.
- **Stop/Go:** ship RAG behind flag; enable by default only after eval.

### Phase 7 — Production Validation
- **Objective:** prove the §14 gates end-to-end.
- **Actions:** full suite on candidate; canary in Docker compose; 24 h soak; security scan; human eval round; run-manifest reproducibility check.
- **Deliverables:** release-candidate report; go/no-go memo.
- **Validation:** all §14 gates green.
- **Stop/Go:** promote to `candidate` → canary; any red gate blocks.

### Phase 8 — Deployment
- **Objective:** stable production operation.
- **Actions:** promote registry entry; enable canary→full rollout; dashboards/alerts live; rollback rehearsal; post-release eval monitoring cadence (monthly external-benchmark re-run, quarterly data refresh).
- **Deliverables:** production model in registry; runbook updates.
- **Validation:** canary metrics ≥ stable; rollback drill succeeded.
- **Stop/Go:** full rollout; begin next improvement cycle.

## 16. Risks & Failure Modes

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overfitting small dataset → brittle pattern-matching model | High | High | Data scaling before HP tuning; FP-on-clean-code metric; replay data |
| Catastrophic forgetting from cyber-heavy mix | Medium | High | Replay mix exp-004; general-capability gate (§14) |
| Benchmark contamination inflating scores | Medium | High | Existing contamination checker enforced on every data PR |
| Eval set too small to detect real change | High (today) | Medium | Expand to ≥200 test items (Phase 3) |
| Judge model bias/wrong "Gemma 4 26B" assumption | Medium | Medium | Verify judge model id; judge≠subject; human eval cross-check |
| Artifact loss (recurred in exp-002) | Medium | High | Immediate artifact persistence + SHA-256 manifest + registry upload as first post-train step |
| License friction on redistribution | Low-Med | Medium | Prefer Apache-2.0/MIT bases for anything distributed |
| Mac-only path blocked by tooling (bitsandbytes/MPS gaps) | Medium | Medium | MLX as first-class local path; CUDA cloud as fallback |
| RAG retrieved-content prompt injection | Medium | High | Treat retrieved text as data; injection filtering; output policy layer |
| "Production-ready" claimed without evidence | Medium | High | §14 gates are blocking; human eval required |

## 17. P0/P1/P2/P3 Priorities

- **P0 — Required before additional training**
  - [ ] Re-run exp-002 (exp-002r) with immediate artifact persistence; evaluate on dev + external MCQ.
  - [ ] Verify judge model identity (`gemma4:26b` reference) or substitute a verified 27B-class judge.
  - [ ] Expand benchmark test split to ≥200 items with contamination check.
  - [ ] Calibrate §14 thresholds from measured baseline.
- **P1 — High impact**
  - [ ] sft_v0.3: scale verified data to 1.5–3k incl. abstention category and general replay.
  - [ ] Stand up MLX LoRA local training path for the 4B tier.
  - [ ] Run experiment matrix §8 (exp-003..005).
  - [ ] Integrate CyberMetric + SecQA as zero-shot cross-checks.
- **P2 — Valuable optimization**
  - [ ] Qwen3-14B flagship tier (exp-006) via MLX locally or Unsloth/QLoRA on cloud.
  - [ ] RAG service over CVE/CWE/OWASP + fact registry with nightly refresh.
  - [ ] DPO pass with judge-built pairs (exp-007).
  - [ ] Canary routing + rollback drill in the existing Docker/registry stack.
- **P3 — Optional / experimental**
  - [ ] Tool calling / sandboxed code execution.
  - [ ] 30B-A3B MoE experiments locally.
  - [ ] Curriculum-ordered data mixing (epoch-1 front-loading).
  - [ ] Continued pretraining — only if a large clean licensed corpus materializes and Phases 0–6 plateau.

## 18. Final Model Recommendation

**Recommendation: (4) Maintain multiple models — two tiers on one dataset and one eval harness.**

1. **Keep** the fine-tuned `gemma-3-4b-it` as the local/latency tier — it trains in minutes locally via MLX, serves fast on the M3 Max, and is already wired into your registry.
2. **Add** **Qwen3-14B (Apache-2.0)** as the flagship tier: 3.5× capacity, permissive license, strong verified coding ability, trivial 4-bit inference (~10 GB), easy MLX LoRA locally. If Phase 3 shows the 14B leaves headroom on your eval, step up to **Qwen3-32B** (still ~20 GB at 4-bit, MLX-LoRA-feasible on 128 GB **[Estimate]**). **[Recommendation based on verified specs]**
3. **Do not switch the whole project to a larger single model now**, and do not decide until exp-002r + exp-003 produce real numbers: your pipeline is excellent but unmeasured, and every base-model choice should be judged by the same benchmark, not intuition.

Conditional: if all serving must remain on the Mac with strict latency, cap at 14B; if a server GPU is available and redistribution matters, prefer Qwen3-32B over Gemma-3-27B for licensing; if only inference (no fine-tune) is needed for exploration, a 70B at 4-bit is runnable but slow.

## 19. Verification Checklist

Items to confirm against primary sources before relying on them:

- [ ] Exact Unsloth Mac/MPS training status at implementation time (unsloth.ai docs / GitHub) — currently "not supported, in progress."
- [ ] Unsloth support for each chosen model/version in the official model catalog (Gemma 3 4B/27B, Qwen3 14B/32B confirmed at time of writing; newer 2026 families **Verify**).
- [ ] MLX→GGUF conversion support and fidelity for each chosen checkpoint (llama.cpp `convert_hf_to_gguf.py` supported-models list).
- [ ] Official GGUF quant file sizes for 14B/27B/32B/70B candidates (HF model pages) — memory figures here are estimates.
- [ ] Qwen3 context length (32K native / 128K YaRN) on the official model card; whether your serving stack supports YaRN.
- [ ] DeepSeek-R1-Distill license text per base model card (MIT noted, verify per checkpoint).
- [ ] Judge model: the exact `gemma4:26b` id referenced in repo docs — confirm existence/specs on Ollama/HF or substitute a verified model.
- [ ] macOS GPU memory limit on your machine (`sudo sysctl iogpu.wired_limit_mb`) before sizing 32B+ training runs.
- [ ] Current NVD/CVE API terms and rate limits before building the refresh job.
- [ ] External benchmark licenses and permitted usage (CyberMetric, SecQA, CTIBench, CyberSecEval) before redistributing results.
