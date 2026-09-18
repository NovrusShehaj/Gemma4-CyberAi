# Plan: Write `Gemma4-CyberAI-Training-Roadmap.md`

**Goal:** Produce the single Markdown deliverable requested in the prompt, at the repo root: `/Users/novrusshehaj/Github/Gemma4-CyberAI/Gemma4-CyberAI-Training-Roadmap.md`, following the exact required structure (19 sections, `# Gemma4-CyberAI Training & Production Roadmap` first line), with confidence labels, no fabricated facts, and `Verification required` markers where 2026-era specifics are unconfirmed.

## Grounded facts to use

**From repo exploration:**
- Current base: `google/gemma-3-4b-it` (QLoRA 4-bit NF4, exported GGUF Q4_K_M via llama.cpp). Registry's "production" model is still the un-fine-tuned base; `gemma3-cyber:v0.2` trained once on Colab but artifacts were lost before evaluation.
- Training stack: transformers + PEFT/TRL + bitsandbytes (pinned), config-driven `scripts/train_qlora.py`, Colab T4 notebook; shared SFT masking module; NOT Unsloth or MLX yet.
- Data: `sft_v0.2.jsonl` (277 examples, 15 task types), contamination checker, `security_facts.json` fact registry. Benchmarks: v2 (dev/test split) + v3 (47), hallucination/judge scorers, pre-registered criteria.
- Production: FastAPI + Auth0, Ollama serving, Docker, CI security scanning, 156 tests, model registry with staged gating (experimental→evaluated→candidate→production). RAG/tools deliberately deferred.

**From verified web sources (with confidence labels):**
- Unsloth training on Apple Silicon/MPS is NOT supported (training needs CUDA); Mac support in progress — high confidence (unsloth GitHub/docs). GGUF exports run fine on Mac.
- Unsloth supports Gemma 3 (4B/12B/27B), Qwen3 family, Llama 3.1/3.3/4, Phi-4, DeepSeek-R1 distills — model catalog.
- Candidate models with licenses/context: Qwen3 (Apache-2.0, 32K/128K YaRN), Gemma 3 27B (Gemma license, 128K), Phi-4 14B (MIT, 16K), Llama 3.3 70B, DeepSeek-R1-Distill-Qwen-32B. 2026-era "Gemma 4" entries seen only in Unsloth catalog → label medium confidence / "Verification required".
- MLX (mlx-lm) does LoRA fine-tuning natively on Apple Silicon via unified memory; 4-bit LoRA of up to ~30B-class models practical on 128 GB — medium-high confidence; 70B 4-bit feasible but slow.
- Benchmarks: CyberMetric, CTIBench, CyberSecEval 1/2, SecQA v1/v2 — verified to exist; use as external suites alongside the repo's internal harness (contamination-aware).

## Document plan (section mapping to the 19 required headings)

1. **Executive Recommendation** — Keep gemma-3-4b-it as fast/local tier; re-run exp-002 first (P0, data already exists); upgrade dev/flagship tier to Qwen3-14B or 32B-class (Apache-2.0) fine-tuned via MLX locally or Unsloth on cloud GPU; hybrid multi-model registry fits existing staged gating.
2. **Assumptions & Unknowns** — hardware (M3 Max 128 GB confirmed as stated), dataset licensing, eval-judge model (repo refs `gemma4:26b` — flag verify), current artifacts lost.
3. **Current Model Assessment** — 4B class is capacity-limited; infra ahead of model quality.
4. **Recommended Training Strategy** — technique table (CPT/SFT/LoRA/QLoRA/full FT/DPO/RAG/tools) with purpose/compute/use-it?/order.
5. **Dataset Architecture** — build on builder_v2 schema; scale 277 → ~3–5k curated; categories, proportions (conceptual), cleaning/dedup/contamination/temporal split, exploit-content controls.
6. **Cybersecurity Curriculum** — 16-stage progression, rationale.
7. **Unsloth Training Configuration** — honest split: Unsloth config for cloud CUDA path; MLX-equivalent local path; recommended ranges per model size with justification; which knobs to tune first (LR, epochs, rank).
8. **Experiment Matrix** — 5–6 one-variable experiments aligned with repo's §19 one-variable rule.
9. **Evaluation & Benchmarking** — reuse internal harness + external suites; metrics incl. FP rate, calibration, refusal; human eval; ground-truth vulnerable-code fixtures.
10. **Hallucination Reduction** — leverage existing fact registry/scorers; abstention policy, RAG, negative examples, regression gates.
11. **Alternative Base Models** — verified comparison table with confidence column.
12. **M3 Max / 128 GB Feasibility** — inference vs fine-tuning matrix; explicit "unified memory ≠ CUDA VRAM"; Unsloth/MPS limitation called out.
13. **Production Architecture** — extend current Ollama+FastAPI; add RAG with CVE refresh, sandboxed code analysis, observability, canary/rollback via existing registry.
14. **Production Readiness Gates** — measurable thresholds tied to `configs/eval_success_criteria.md`.
15. **Phased Roadmap** — Phases 0–8 with stop/go.
16. **Risks & Failure Modes**.
17. **P0/P1/P2/P3 Priorities** — P0: re-run exp-002 + eval; etc.
18. **Final Model Recommendation** — multi-tier: keep 4B for local/latency, add 14–32B flagship.
19. **Verification Checklist** — items the user must confirm on primary sources.

**Step:** Write the file, then confirm the path. No code changes, no commits.