# Training, export, and evaluation runbook (exp-002)

This is the reproduce-from-scratch guide for the first real fine-tune,
`gemma3-cyber:v0.2`. It covers the environment, the training run, artifact
persistence, evaluation, and deployment. It documents the **known failures** of
the first run and how the pipeline was hardened against them.

> **Status of the first run (2026-08-25):** training completed and a GGUF was
> exported, but **the artifacts were lost to a Colab runtime reset and no
> evaluation was run**. Model improvement is therefore **UNPROVEN** — only
> training loss exists, and training loss is not evidence of model quality. See
> `docs/experiments/exp-002.md` §8 and `docs/training/cyber_ai_colab_results.md`.

---

## 1. Environment

Local machine (this repo): inference, dataset build/validation, evaluation. No
GPU required. Python 3.11–3.12, `uv` for the dev env:

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

Training (cloud GPU — free Colab/Kaggle T4/L4): the ML stack is pinned in
[`configs/training/requirements-train.txt`](../../configs/training/requirements-train.txt).

| Component | Constraint / note |
|---|---|
| Python | 3.11–3.13 (Colab currently ships 3.13) |
| GPU | Tesla T4 (15 GB) sufficient at seq_len 1024; L4/A100 faster |
| torch / transformers / peft / trl | see `requirements-train.txt`; transformers ≥ 4.50 for Gemma-3 |
| **torchao** | **≥ 0.16.0, installed up front** — the exact version skew that crashed the first run's merge step |
| bitsandbytes / accelerate / datasets | see `requirements-train.txt` |
| llama.cpp | cloned in the notebook; the resolved commit is recorded to the run manifest |

**Reproducibility rule:** the notebook's Step 1 writes `pip freeze` and Step 5
writes the resolved `llama.cpp` commit into the run manifest. After a green run,
copy those exact versions into `requirements-train.txt` (`==` pins) and paste the
commit into the notebook's `git checkout` line to make the run bit-reproducible.

## 2. Data (deterministic, local)

```bash
python -m gemma_cyber.data.builder_v2                 # -> data/training/sft_v0.2.jsonl
python scripts/build_benchmark_v3.py                  # -> data/evaluation/benchmark_v3.jsonl
python scripts/validate_dataset.py --dataset data/training/sft_v0.2.jsonl \
    --check-contamination data/evaluation/benchmark_v2.jsonl data/evaluation/benchmark_v3.jsonl
```
The validate step must report **0 exact / 0 fuzzy** contamination and unique IDs.
`sft_v0.2` is 277 examples, 277/277 unique answers, 82 fabricated-premise traps.

## 3. Train + export (cloud)

Open `notebooks/colab_qlora_training.ipynb` in Colab (T4 or better) and run the
cells in order. What each step guarantees:

- **Step 1** — pinned install (torchao ≥ 0.16.0 up front) + `pip freeze` capture.
- **Steps 3–4** — QLoRA load and training. Step 4 uses the shared, unit-tested
  `gemma_cyber.training` helpers, so the notebook and `scripts/train_qlora.py`
  format and mask **identically** (Gemma-3 turns via `to_gemma_chat_text`; loss
  only on the model completion).
- **Step 5** — **self-contained** merge + GGUF convert + Q4_K_M quantize. It
  re-imports everything, so a kernel restart between Step 4 and Step 5 no longer
  breaks it. It records the llama.cpp commit and then runs
  `scripts/verify_gguf_export.py`, which **asserts the file is real** (exists,
  plausible size, checksum, readable GGUF metadata) before declaring success.
- **Step 5.5** — writes `run_manifest/manifest.json` (repo commit, dataset hash,
  GGUF + adapter checksums, seed) and **copies artifacts to Google Drive** so a
  runtime reset cannot destroy them. This is the direct fix for the first run's
  data loss. **Do not skip it.**

Headless alternative (any GPU box):
```bash
pip install -r configs/training/requirements-train.txt
python scripts/train_qlora.py --config configs/training/qlora_gemma3_4b_v0.2.yaml
# then merge/convert with llama.cpp (see notebook Step 5) and:
python scripts/verify_gguf_export.py gemma3-cyber-v0.2-Q4_K_M.gguf --min-size-mb 1500
```

## 4. Deploy (local, Ollama)

Download the GGUF off Colab/Drive, re-verify it, then create the Ollama model on
**your own machine** (not the Colab terminal — it has no Ollama):

```bash
python scripts/verify_gguf_export.py gemma3-cyber-v0.2-Q4_K_M.gguf --min-size-mb 1500
# sha256 must match run_manifest/manifest.json's gguf_sha256.
# Point a Modelfile FROM line at the v0.2 GGUF (copy Modelfile.template, edit FROM):
ollama create gemma3-cyber:v0.2 -f Modelfile.template
```

## 5. Evaluate (the only evidence of quality)

Run **base and candidate** on both benchmarks, held-out `test` split, per the
pre-registered design in `docs/experiments/exp-002.md`:

```bash
python scripts/run_baseline.py --model gemma3:4b        --benchmark data/evaluation/benchmark_v2.jsonl --split test --out experiments/exp-002-gemma3-cyber-v0.2/base-v2-test
python scripts/run_baseline.py --model gemma3:4b        --benchmark data/evaluation/benchmark_v3.jsonl --split test --out experiments/exp-002-gemma3-cyber-v0.2/base-v3-test
python scripts/run_baseline.py --model gemma3-cyber:v0.2 --benchmark data/evaluation/benchmark_v2.jsonl --split test --out experiments/exp-002-gemma3-cyber-v0.2/v0.2-v2-test
python scripts/run_baseline.py --model gemma3-cyber:v0.2 --benchmark data/evaluation/benchmark_v3.jsonl --split test --out experiments/exp-002-gemma3-cyber-v0.2/v0.2-v3-test
```
Each run writes `results.json` (machine-readable, full per-item responses +
metadata) and `scorecard.md`. Because responses are stored, re-scoring after a
scorer change needs no re-inference (`scripts/rescore.py`). The candidate passes
only if it clears the criteria in `docs/experiments/exp-002.md` §4.

## 6. Known non-issues (investigated, benign)

- **`WARNING:gguf.gguf_writer:Duplicated key name 'gemma3.*'`** during conversion
  is a known upstream `convert_hf_to_gguf.py` behavior for the Gemma-3
  architecture: a few metadata keys are written on both a generic and an
  arch-specific path; the writer keeps the last value. It is benign **as long as
  the duplicated values agree**, which `scripts/verify_gguf_export.py` surfaces
  (`no_unexpected_duplicate_keys` + the resolved field values) so it can be
  confirmed rather than assumed.
- **`torch_dtype is deprecated`** is a forward-compat warning only; pinned
  `transformers` still honors it.
- **Non-monotonic training loss** (rises at some 10-step log points) is ordinary
  mini-batch noise at effective batch size 8 over ~105 steps — not an instability
  signal on its own.

## 7. Limitations

- 277 training examples on a 4B model is small; generalization must be judged by
  the held-out benchmarks, not the loss curve.
- There is no in-training validation split (deliberately — the benchmarks are the
  held-out signal and must not leak into training). Overfitting is watched via the
  v2 do-no-harm guard at eval time.
- Deterministic scorers are proxies; the LLM judge is a supplement, never the
  primary number (`docs/judge.md`).
