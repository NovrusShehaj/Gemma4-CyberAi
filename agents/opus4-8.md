# exp-002 Colab Run — Analysis & Hardening Report

All changes are implemented and verified. Here is the full report.

## Executive Summary

The first exp-002 Colab run **executed but is inconclusive**. Training completed (105 steps, loss 2.89→~0.51) and a Q4_K_M GGUF was exported at least once — then a Colab runtime reset wiped `/content`, and **no evaluation was ever run**. So the only metric that exists is training loss, which is not evidence of model quality. **Whether the fine-tune improved the model is unproven.**

Crucially, the parts people worry about were not the problem: the dataset is genuinely good (277/277 unique answers, 0 contamination, 82 trap items — all re-verified), the loss curve is normal, and the GGUF duplicate-key warnings are benign. The real failures were reproducibility and reliability: no artifact persistence, unpinned dependencies (the actual cause of the torchao crash), a committed notebook whose Step 5 broke on kernel restart, and a training script that had silently diverged from the validated notebook. I fixed all four and hardened the export/eval workflow, keeping every change minimal and CI-verified.

## Confirmed Problems

1. **No artifact persistence (P0).** Success was a bare `print`; a runtime reset destroyed the adapter, merged model, and GGUF. Evidence: §17.4 terminal transcript (`/content` empty).
2. **Unpinned dependencies (P0).** Step 1's `pip install` was unpinned; the resolved `peft` required `torchao>0.16.0` while Colab shipped `0.10.0`, crashing the merge. Evidence: §17.2 ImportError.
3. **Committed notebook Step 5 depended on hidden kernel state (P0).** The self-contained fix was made live in Colab but never committed — the repo notebook still had the version that `NameError`ed after a restart. Evidence: cell-12 as read vs §17.6.
4. **`scripts/train_qlora.py` diverged from the notebook (P0).** It fed raw `messages` to TRL's chat template (the exact path `formatting.py` warns is unsafe for Gemma's system role) and used removed/renamed TRL APIs (`TrainingArguments`, `tokenizer=`, `max_seq_length=`) — a second, different, likely-broken training path.
5. **No export verification (P1).** No checksum, size gate, or metadata check.
6. **No evaluation → model quality unproven (confirmed, not fixable without artifacts).**

## Investigated Hypotheses

- **Non-monotonic training loss → instability?** Investigated the schedule (cosine), effective batch 8, ~35 steps/epoch, 10 log points. **Conclusion: ordinary mini-batch noise; not a defect.** Documented as a non-issue.
- **GGUF duplicate-key warnings → corrupt metadata?** Investigated against known `convert_hf_to_gguf.py` behavior for Gemma-3. **Conclusion: benign** — a few keys are written on both a generic and arch-specific path and the writer keeps the last value; safe *as long as the duplicated values agree*. I did not hand-wave it: `verify_gguf_export.py` now parses the GGUF and reports duplicate keys + resolved values so it's *confirmed* rather than assumed. Cannot fully verify without regenerating the file (stated).
- **torchao ABI warning (`cpython-310` .so under python3.13) → fatal?** **Conclusion: non-blocking** optional CUDA-kernel load failure, irrelevant to the CPU merge path; the pinned-and-up-front `torchao>=0.16.0` is what actually matters.
- **`torch_dtype` deprecation → breakage?** **Conclusion: forward-compat warning only**; pinned `transformers` honors it.
- **Dataset quality / contamination / ATT&CK correctness?** Re-ran validation: 277/277 unique, 0 exact/0 fuzzy vs both benchmarks, Kerberoasting trap correctly requires `T1558.003` and forbids `T1060`/`T1068`. **Conclusion: not a problem.**

## Model Quality Assessment

**Unproven.** No benchmark was scored on either arm, so nothing can be said about ATT&CK precision, hallucination, false-premise handling, or do-no-harm regression. Training loss dropping to ~0.5 shows the adapter fit the data, not that it generalizes — especially with 277 examples on a 4B model, where memorization is plausible. The pre-registered success criteria in `exp-002.md` §4 remain the only valid verdict, and they require a re-run + evaluation.

## Changes Implemented

1. **Shared training module — `src/gemma_cyber/training/{__init__,sft.py}` (new).** Single source of truth for Gemma-3 rendering (`format_for_sft`→`to_gemma_chat_text`), completion-only masking (`completion_mask_start`, a pure/tested function reused by the collator), and version-robust TRL config/trainer kwargs. Heavy imports deferred so CI (no torch) still imports it. *Rationale: kills the notebook↔script divergence (problem 4) with one tested implementation.*
2. **`scripts/train_qlora.py` refactored.** Now uses the shared helpers + `SFTConfig`/`processing_class`, formats identically to the notebook, prints a format sample, and defaults to the v0.2 config. *Rationale: the "official" script and the validated run now train the same way.*
3. **`configs/training/requirements-train.txt` (new).** Pinned ML stack with `torchao>=0.16.0` up front and a documented "capture `pip freeze` → freeze `==`" workflow. *Rationale: the confirmed dependency failure (problem 2).*
4. **`scripts/verify_gguf_export.py` (new).** Asserts existence + plausible size + SHA-256 + GGUF metadata/duplicate-key check; writes `verify_gguf.json`; exit 0/1/2. *Rationale: makes "export succeeded" checkable (problem 5).*
5. **`notebooks/colab_qlora_training.ipynb` hardened.** Step 1: pinned install + `pip freeze` capture. Step 4: uses the shared module. Step 5: self-contained + records llama.cpp commit + runs the verifier + asserts. **New Step 5.5: run manifest (commit, dataset hash, checksums, seed) + Google Drive persistence with a download fallback + optional HF Hub backup.** Step 6: rewritten to make the local-machine handoff explicit and point at the verifier/manifest. *Rationale: problems 1 and 3.*
6. **Docs.** `docs/training/README.md` (new full runbook incl. known-benign issues + limitations) and `docs/experiments/exp-002.md` §8 (honest Run-1 outcome: improvement UNPROVEN).
7. **Tests (new).** `tests/test_training_sft.py` (masking boundary, formatting, version-robust kwargs) and `tests/test_verify_gguf.py` (missing/size/checksum). +14 tests.

## Evaluation Results

**No model evaluation results exist and none were fabricated.** Running the benchmarks is BLOCKED: no GGUF artifact, no GPU to regenerate it, and no local Ollama. The evaluation *workflow* is verified reproducible (harness/scorers untouched and green; commands documented in `exp-002.md` §6 and `docs/training/README.md` §5).

## Tests and Verification

- `ruff check src tests scripts` — **PASSED**
- `mypy src tests scripts` (41 files) — **PASSED**
- `pytest` — **PASSED (104, was 90)**
- `python scripts/train_qlora.py --dry-run` — **PASSED (exit 0)**
- `validate_dataset.py … --check-contamination v2 v3` — **PASSED (0 exact/0 fuzzy, 277/277 unique)**
- `verify_gguf_export.py` exit codes (missing=2, ok=0, too-small=1) — **PASSED**
- Notebook JSON valid + all code cells parse — **PASSED**
- Actual training / real GGUF export / GGUF metadata read on a real file / benchmark scoring — **BLOCKED** (no GPU, no artifact, no Ollama).

## Reproducibility

`uv venv --python 3.12 && uv pip install -e ".[dev]"` for local dev. Build/validate data locally, then run the hardened notebook in Colab (pinned deps → captured `pip freeze`; self-contained export → verifier + manifest; Step 5.5 → Drive). Off Colab: `pip install -r configs/training/requirements-train.txt` + `python scripts/train_qlora.py`. The manifest records repo commit, dataset SHA-256, package versions, llama.cpp commit, GGUF/adapter checksums, and seed — enough to reproduce exactly. Full runbook in `docs/training/README.md`.

## Remaining Risks

- **The pins in `requirements-train.txt` are constraint ranges, not hardware-verified `==` locks** — I can't resolve them on a T4 from here, so they're framed as a known-good baseline to be frozen from the first green run's captured `pip freeze`. Honest, but not yet bit-reproducible.
- **GGUF metadata correctness** can only be fully confirmed by running the verifier on a real regenerated file.
- **Model quality stays unproven** until the re-run + evaluation happens.
- **Google Drive `drive.mount` needs interactive auth**; guarded with a download fallback, but persistence still depends on the user completing Step 5.5.

## Final Assessment

The project is well-engineered on the data/eval side; the first experiment failed on **MLOps hygiene, not modeling**. After these changes the repo is meaningfully better in reproducibility (pinned + captured deps, manifests), reliability (self-contained, verified, persisted export), deployment (checked handoff), and maintainability (one shared, tested training path). I make **no claim that the fine-tune improved the model** — that requires re-running the hardened notebook and executing the four pre-registered evaluations. Changes are on branch `p1-eval-hardening` and left uncommitted for your review; say the word and I'll commit them.
