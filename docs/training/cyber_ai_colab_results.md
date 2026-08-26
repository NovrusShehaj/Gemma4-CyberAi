# Cyber AI — Google Colab Experiment Results

**Source:** `colab_qlora_training.ipynb` (Google Colab, live notebook inspection via browser)
**Extraction date:** 2026-08-25
**Extraction method:** Direct visual/DOM inspection of the notebook's markdown cells, code cells, and cell outputs, plus a live Colab terminal session on the same runtime. Long virtualized outputs (in particular the GGUF conversion log in Step 5) could not be scrolled through exhaustively; representative excerpts are captured and this is noted explicitly wherever it applies.

---

## 1. Executive Summary

This notebook runs **exp-002** of the "Gemma-Cyber" project: the first genuine QLoRA fine-tune of `google/gemma-3-4b-it` on a curated cybersecurity instruction dataset (`sft_v0.2.jsonl`, 277 examples), followed by merging the LoRA adapter into the base model and exporting a quantized GGUF file for local Ollama inference.

**Observed outcome:** Training completed (105/105 steps, 3 epochs, ~39.5 minutes), producing a saved LoRA adapter. A later run of the merge/export cell (Step 5) also completed, producing log messages consistent with a successful merge and GGUF export/quantization. However, at the time of this extraction, the Colab runtime had disconnected/reset and its local disk (`/content`) was empty — none of the training or export artifacts (adapter, merged model, `.gguf` file) currently exist in the live environment. Whether the final `.gguf` file was downloaded to the user's local machine before the reset is **unknown**.

No evaluation has been run yet. Step 6 (Ollama model creation + the two pre-registered benchmark evaluations, `benchmark_v2.jsonl` and `benchmark_v3.jsonl`) has not been executed. **There are currently no accuracy/quality metrics for the fine-tuned model** — only training loss is available.

## 2. Project Context

- **Project name (from notebook title):** "Gemma-Cyber v0.2 (exp-002): Free Cloud QLoRA Training & GGUF Export"
- **Repository:** `https://github.com/novrusshehaj/Gemma4-CyberAi` (cloned during Step 2)
- **Stated purpose:** Fine-tune Gemma-3-4B-it on a curated cybersecurity dataset using QLoRA, merge the adapter, and export a quantized GGUF model for local inference via Ollama.
- **Experiment identity:** exp-002 — described in the notebook as **"the project's first real fine-tune"**; the prior model, `gemma3-cyber:v0.1`, is explicitly described as "only a system-prompt alias of the base model" (i.e., not actually fine-tuned).
- **Dataset rationale (as stated in the notebook):** `sft_v0.2` is described as "the first dataset with genuine answer-level diversity (277/277 unique answers vs. v0.1's 91/360) across 15 task types," and it "explicitly teaches exact ATT&CK IDs (e.g. Kerberoasting = T1558.003, not the hallucinated T1060)." This implies v0.1's dataset (or the base model) previously hallucinated an incorrect MITRE ATT&CK ID for Kerberoasting.
- **Gemma-3 chat-format note (from notebook markdown):** "Gemma templates historically reject a standalone system role and use the role name `model` (not `assistant`)." The notebook renders training text with the repo's own `to_gemma_chat_text()` formatter (folds system into the first user turn) instead of relying on the tokenizer's chat template, and masks the prompt so loss is computed only on model turns.
- **Intended downstream use:** A quantized GGUF model (`gemma3-cyber-v0.2-Q4_K_M.gguf`) run locally via Ollama, evaluated against a base `gemma3:4b` model on two benchmark sets.
- **Hardware target (as stated):** "Free Google Colab T4 GPU (15GB VRAM) or L4/A100. Sequence length 1024 keeps a T4 comfortable."

## 3. Environment

| Item | Value | Evidence |
|---|---|---|
| Platform | Google Colab | Notebook UI |
| GPU | Tesla T4, 15360 MiB VRAM | `nvidia-smi` output, Step 1 |
| GPU Driver | 580.82.07 | `nvidia-smi` output, Step 1 |
| CUDA Version (driver-reported) | 13.0 | `nvidia-smi` output, Step 1 |
| GPU utilization at check-in | 0%, no running processes | `nvidia-smi` output, Step 1 |
| Python version | 3.13 (inferred) | Site-packages paths observed in tracebacks/warnings, e.g. `/usr/local/lib/python3.13/dist-packages/peft/...` |
| OS/base image | Not available from the Colab notebook | — |
| Key packages installed (Step 1) | `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`, `pyyaml` | `pip install -q torch transformers peft trl bitsandbytes accelerate datasets pyyaml` (Step 1) — **exact installed version numbers are not printed anywhere in the visible output** (install command does not pin versions) |
| `torchao` version (initial) | 0.10.0 | Exact version string quoted in the `ImportError` raised by `peft.import_utils.is_torchao_available()` |
| `torchao` version required by installed `peft` | `> 0.16.0` | Same `ImportError` message |
| `torchao` version after fix | `>=0.16.0` (exact resolved version not printed) | `pip install -q -U "torchao>=0.16.0"` was run; the installer's resolved version was not captured in visible output |
| llama.cpp | Cloned fresh from `https://github.com/ggerganov/llama.cpp` each Step 5 run | Step 5 code/output |
| Terminal shell | bash, cwd `/content` | Colab "Terminal" panel |

## 4. Models

### 4.1 Base model
- **Name:** `google/gemma-3-4b-it`
- **Gating:** Gated on Hugging Face; requires accepting the license and an `HF_TOKEN` (Step 1.5).
- **Quantization for training (QLoRA):**
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type='nf4'`
  - `bnb_4bit_use_double_quant=True`
  - `bnb_4bit_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16`
- **Download footprint observed:** "Download complete: 7.75GB, 77.7MB/s"; "Reconstruction complete: 8.60GB / 8.60GB"; weight loading "883/883 [00:34<00:00, 132.24it/s]".
- **Deprecation warning observed:** `[transformers] \`torch_dtype\` is deprecated! Use \`dtype\` instead!` (Step 3 output). The notebook code still uses `torch_dtype=...` in multiple places (Steps 3 and 5).

### 4.2 LoRA adapter (`gemma3-cyber:v0.2` adapter)
- **Base:** `google/gemma-3-4b-it`
- **LoRA configuration (`LoraConfig`, Step 3):**
  - `r=16`
  - `lora_alpha=32`
  - `lora_dropout=0.05`
  - `bias='none'`
  - `task_type='CAUSAL_LM'`
  - `target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']`
- **Trainable parameters (observed, `model.print_trainable_parameters()` output, Step 3):**
  `trainable params: 32,788,480 || all params: 4,332,867,952 || tra...` — **the trailing `trainable%` figure was cut off in the rendered output and could not be recovered** (the underlying output text lives in a cross-origin sandboxed iframe that could not be scrolled or read programmatically). *Computed from the two observed integers (not itself an output value read from the notebook):* 32,788,480 / 4,332,867,952 ≈ **0.757%**.
- **Saved to:** `./final_adapter` (confirmed present on disk via terminal `ls`, with `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `README.md`, `tokenizer_config.json`, `tokenizer.json`, timestamped Aug 25 15:13 — see §14).

### 4.3 Merged / exported model
- **Merged model directory (as coded):** `./gemma3-cyber-v0.2-merged` (FP16, produced by `PeftModel.from_pretrained(base_model, './final_adapter').merge_and_unload()`)
- **GGUF (F16) file (as coded):** `gemma3-cyber-v0.2.gguf`
- **Quantized GGUF (as coded):** `gemma3-cyber-v0.2-Q4_K_M.gguf` (quantized with `llama-quantize ... Q4_K_M`)
- **GGUF metadata observed in conversion log:**
  - `context length = 131072`
  - `embedding length = 2560`
  - `feed forward length = 10240`
  - `head count = 8`
  - `key-value head count = 4`
  - `rope scaling type = LINEAR`
  - `rope theta = 1000000.0`
  - `rope theta swa = 10000.0`
  - `rms norm epsilon = 1e-06`
  - `file type = 1` (F16, per `convert_hf_to_gguf.py` conventions)
- **Current existence:** **Not present in the current runtime.** Terminal check at extraction time found no `Gemma4-CyberAi` directory, no merged-model directory, and no `.gguf` files anywhere under `/content`. Whether this file was downloaded to the user's local machine before the runtime reset is **unknown**.

## 5. Datasets

| Item | Value |
|---|---|
| Name | `sft_v0.2.jsonl` |
| Path in repo | `data/training/sft_v0.2.jsonl` |
| Size (examples) | 277 (`Successfully loaded 277 training examples from data/training/sft_v0.2.jsonl.`) |
| Uniqueness/diversity (as stated in notebook markdown) | "277/277 unique answers vs. v0.1's 91/360" |
| Task type coverage (as stated) | "15 task types" |
| Format | JSON Lines; each line has an `id` field and a `messages` field (chat-style turns) |
| Validation performed | `assert len({json.dumps(it["id"]) for it in items}) == len(items), 'duplicate ids!'` — no assertion error was observed, implying all 277 ids were unique |
| Chat formatting | Converted via repo function `gemma_cyber.data.formatting.to_gemma_chat_text()`, not the tokenizer's built-in chat template |
| Loss masking | Custom `CompletionOnlyCollator` masks all tokens up to and including the first `<start_of_turn>model\n`, so loss is computed only on model-turn tokens |
| Known content example (from markdown, not from a dataset row directly) | "Kerberoasting = T1558.003" cited as the correct ATT&CK ID the dataset teaches, contrasted with a "hallucinated T1060" |
| Sample formatted training example (actual text) | **Not available from the Colab notebook.** The code prints `formatted[0]['text'][:300]` under the label "Sample formatted example:", but this specific output text could not be retrieved (output rendered inside a cross-origin sandboxed iframe not reachable by DOM inspection, and not visible in the portion of the virtualized output that could be scrolled into view). |
| Evaluation/held-out datasets referenced (not yet run) | `data/evaluation/benchmark_v2.jsonl` ("do-no-harm / regression anchor"), `data/evaluation/benchmark_v3.jsonl` ("targeted sensitivity instrument — ATT&CK precision, false premises, factual scorer") — both scored on a `test` split |
| `sample_data` folder | Present in `/content` — this is Colab's default demo data folder, unrelated to this project |

## 6. Training Configuration

Configuration from `SFTConfig` / `LoraConfig` / `BitsAndBytesConfig` in Steps 3–4:

| Parameter | Value |
|---|---|
| Base model | `google/gemma-3-4b-it` |
| Quantization (train-time) | 4-bit NF4, double quant, compute dtype bf16 (if supported) else fp16 |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA bias | none |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Task type | CAUSAL_LM |
| Epochs | 3 |
| Per-device train batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size (computed: 2 × 4) | 8 |
| Gradient checkpointing | True |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Optimizer | paged_adamw_8bit |
| Logging steps | 10 |
| Save strategy | epoch |
| Save total limit | 2 |
| Precision | bf16 if `torch.cuda.is_bf16_supported()` else fp16 |
| Seed | 42 |
| Max sequence length | 1024 (set as `max_length` or `max_seq_length`, whichever the installed TRL version supports) |
| Packing | False |
| Dataset text field | `text` |
| `report_to` | none |
| Output dir | `./results_gemma3_cyber_v0.2` |

**Cross-check (computed, not directly observed):** 277 examples ÷ effective batch size 8 ≈ 34.6 → 35 steps/epoch × 3 epochs = 105 steps, which exactly matches the observed `[105/105 39:28, Epoch 3/3]` progress bar. This is consistent, not a discrepancy.

## 7. Experiments and Runs

### Run 1 — Step 1: Environment setup
- **Objective:** Install dependencies, confirm GPU.
- **Result:** Tesla T4 confirmed, 0% utilization, no processes running. Packages installed via unpinned `pip install`.
- **Errors/warnings:** None observed at this step.

### Run 2 — Step 1.5: Hugging Face authentication
- **Objective:** Authenticate to download the gated `google/gemma-3-4b-it` model.
- **Configuration:** Prefers a Colab secret named `HF_TOKEN`; falls back to an interactive login widget.
- **Result:** **Not available from the Colab notebook** — the printed confirmation (`print('Authenticated as:', whoami()['name'])`) was not visible/retrievable in the inspected output.

### Run 3 — Step 2: Repository clone & dataset validation
- **Objective:** Clone `Gemma4-CyberAi` and validate the training dataset.
- **Result (observed):** Clone succeeded ("Receiving objects: 100% (256/256)... Resolving deltas: 100% (97/97), done."). `Successfully loaded 277 training examples from data/training/sft_v0.2.jsonl.` No assertion error on duplicate IDs.
- **Errors/warnings:** None observed.

### Run 4 — Step 3: Base model load (QLoRA) + LoRA wrap
- **Objective:** Load `google/gemma-3-4b-it` in 4-bit, wrap with a LoRA adapter.
- **Result (observed):** Model files downloaded (7.75GB) and reconstructed (8.60GB); weights loaded (883/883 tensors). `trainable params: 32,788,480 || all params: 4,332,867,952` (trailing % truncated — see §4.2).
- **Errors/warnings:** `[transformers] \`torch_dtype\` is deprecated! Use \`dtype\` instead!`

### Run 5 — Step 4: LoRA fine-tuning via `SFTTrainer`
- **Objective:** Train the LoRA adapter on `sft_v0.2.jsonl` for 3 epochs.
- **Configuration:** See §6.
- **Result (observed):**
  - Progress: `[105/105 39:28, Epoch 3/3]` — **105 steps, 39 minutes 28 seconds, 3 epochs.**
  - Training loss table (exact values, logged every 10 steps):

    | Step | Training Loss |
    |---|---|
    | 10 | 2.890367 |
    | 20 | 1.882823 |
    | 30 | 1.570567 |
    | 40 | 1.334106 |
    | 50 | 0.975393 |
    | 60 | 0.948494 |
    | 70 | 0.993106 |
    | 80 | 0.625255 |
    | 90 | 0.510340 |
    | 100 | 0.685614 |
  - Final message: `Training complete! Adapter saved to ./final_adapter`
- **Errors/warnings:** None observed at this step in the successful run.
- **Observation:** Loss is noisy, not monotonically decreasing — it drops sharply through step 50, rises at step 70, drops to its lowest observed value at step 90 (0.510340), then rises again at step 100 (0.685614). See §12 (Potential Issues).

### Run 6 — Step 5, attempt 1: Merge + GGUF export (FAILED)
- **Objective:** Merge the LoRA adapter into the base model and export a quantized GGUF.
- **Result:** **Failed.** `ImportError` raised inside `peft/import_utils.py`, function `is_torchao_available()`:
  > `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`
- **Root cause (as diagnosed during the debugging session, not stated by the notebook itself):** Step 1 installs `peft` unpinned, which resolved to a version whose `PeftModel.from_pretrained()` performs a `torchao` version check even when `torchao` isn't otherwise used by this merge path. Colab's preinstalled `torchao` (0.10.0) was older than the `>0.16.0` this `peft` version requires.
- **Resolution applied:** Added `!pip install -q -U "torchao>=0.16.0"` to the top of the Step 5 cell.

### Run 7 — Step 5, attempt 2: Merge + GGUF export (FAILED, different error)
- **Objective:** Same as attempt 1, after the `torchao` fix.
- **Result:** **Failed**, with two non-fatal warnings followed by a fatal error:
  - `WARNING:torchao:Failed to load /usr/local/lib/python3.13/dist-packages/torchao/_C_cutlass_90a.abi3.so: Could not load this library: ...`
  - `WARNING:torchao:Failed to load /usr/local/lib/python3.13/dist-packages/torchao/_C_mxfp8.cpython-310-x86_64-linux-gnu.so: Could not load this library: ...`
  - `NameError: name 'AutoModelForCausalLM' is not defined` (raised at `base_model = AutoModelForCausalLM.from_pretrained(...)`)
- **Root cause (as diagnosed during debugging):** The Colab runtime had been restarted (kernel state reset) between attempts. The Step 5 cell depended on `AutoModelForCausalLM`, `torch`, `model_id`, and `tokenizer` all having been defined by Steps 1–4 earlier in the *same* kernel session — after a restart, none of those existed, even though `./final_adapter` was still present on disk (Colab restarts the Python kernel but does not necessarily wipe the VM's disk immediately — confirmed via terminal `ls` at the time, which showed `final_adapter/` fully intact with `adapter_config.json` and `adapter_model.safetensors`).
- **Resolution applied:** Rewrote the Step 5 cell to be self-contained — it now re-imports `torch`/`AutoModelForCausalLM`/`AutoTokenizer`, redefines `model_id`, fixes the working directory back to the cloned repo if needed, and reuses or reloads `tokenizer`, independent of whether Steps 1–4 ran earlier in the same kernel session.

### Run 8 — Step 5, attempt 3: Merge + GGUF export (SUCCEEDED)
- **Objective:** Same as above, with the self-contained fix in place.
- **Result (observed):** Completed. Notable observed output, in order:
  1. `Merged model saved.` (after `merged_model.save_pretrained(...)` and `tokenizer.save_pretrained(...)`)
  2. `llama.cpp` cloned fresh.
  3. `convert_hf_to_gguf.py` ran, logging per-tensor `INFO:hf-to-gguf:blk.N.<tensor_name>, torch.floatN` lines for at least blocks 0 through 9 (directly observed via scrolling; the notebook's virtualized output rendering made it impractical to confirm every block from 0 to the model's full depth, but no interruption or error was observed in any portion that was viewed).
  4. GGUF metadata written (`context length = 131072`, `embedding length = 2560`, `feed forward length = 10240`, `head count = 8`, `key-value head count = 4`, `rope scaling type = LINEAR`, `rope theta = 1000000.0`, `rope theta swa = 10000.0`, `rms norm epsilon = 1e-06`, `file type = 1`).
  5. Three warnings: `WARNING:gguf.gguf_writer:Duplicated key name 'gemma3.context_le[ngth]'` and two more `WARNING:gguf.gguf_writer:Duplicated key name 'gemma3.attention....'` (exact full key names were truncated in the visible output and could not be recovered — see §10).
  6. `llama-quantize` was built and run against `Q4_K_M`.
  7. Final printed line: `Exported gemma3-cyber-v0.2-Q4_K_M.gguf ready for Ollama!`
- **Errors:** None observed in this successful run (subject to the virtualized-output caveat above).
- **Post-run state:** At the time of this extraction (after this run), the Colab runtime showed **"Reconnect"** in the UI (i.e., currently disconnected), and a terminal check of `/content` found it **empty except for Colab's default `.config` and `sample_data` folders** — no `Gemma4-CyberAi` repo, no `final_adapter`, no merged model, no `.gguf` files. This means the runtime's disk was reset/reassigned at some point after this successful run completed.

### Run 9 — Step 6: Ollama creation / evaluation
- **Objective:** Download the `.gguf` file locally, create an Ollama model, and run two benchmark evaluations comparing `gemma3-cyber:v0.2` to base `gemma3:4b`.
- **Result:** **Not executed as designed.** A command, `ollama create gemma3-cyber:v0.2 -f Modelfile.template`, was run directly in the **Colab terminal** (not on the user's local machine) and failed:
  > `-bash: ollama: command not found`
- **Observation:** This is expected — Ollama is not installed on the Colab VM, and Step 6's instructions specify running these commands on the user's own computer after downloading the `.gguf` file there, not inside Colab.
- **Evaluation scripts** (`scripts/run_baseline.py` against `benchmark_v2.jsonl` and `benchmark_v3.jsonl`) — **no evidence they have been run.** No output, results, or metrics for either benchmark were found anywhere in the notebook.

## 8. Metrics

The only quantitative metrics observed anywhere in this notebook are the SFTTrainer training-loss values from Step 4 (Run 5), reproduced in full below (all other rows/columns are either not applicable or not available).

| Metric | Step | Value | Source |
|---|---|---|---|
| Training loss | 10 | 2.890367 | Step 4 output |
| Training loss | 20 | 1.882823 | Step 4 output |
| Training loss | 30 | 1.570567 | Step 4 output |
| Training loss | 40 | 1.334106 | Step 4 output |
| Training loss | 50 | 0.975393 | Step 4 output |
| Training loss | 60 | 0.948494 | Step 4 output |
| Training loss | 70 | 0.993106 | Step 4 output |
| Training loss | 80 | 0.625255 | Step 4 output |
| Training loss | 90 | 0.510340 | Step 4 output |
| Training loss | 100 | 0.685614 | Step 4 output |
| Trainable parameters | — | 32,788,480 | Step 3 output |
| Total parameters | — | 4,332,867,952 | Step 3 output |
| Trainable % (computed, not directly observed) | — | ≈0.757% | Computed from the two rows above |
| Training wall-clock time | — | 39 min 28 sec | Step 4 progress bar |
| Total training steps | — | 105 | Step 4 progress bar |
| Epochs | — | 3 | Step 4 progress bar |

**No evaluation metrics (accuracy, precision/recall/F1, ATT&CK-ID precision, hallucination rate, "do-no-harm" pass rate, or any benchmark score) are available from this notebook.** Both `benchmark_v2.jsonl` and `benchmark_v3.jsonl` evaluations are unexecuted.

## 9. Evaluation Results

**Not available from the Colab notebook.** No evaluation of `gemma3-cyber:v0.2` (base-vs-fine-tuned comparison, benchmark_v2 "do-no-harm" anchor, or benchmark_v3 targeted ATT&CK-precision/false-premise/factual scorer) has been run. No sample model generations/outputs were observed anywhere in the notebook (e.g., no output from the `ollama run gemma3-cyber:v0.2 "Explain the MITRE ATT&CK technique for Kerberoasting."` example command — this command has not been run, only pasted as an instruction in Step 6's markdown).

## 10. Errors and Warnings

| # | Error/Warning | Where | Resolved? | Context |
|---|---|---|---|---|
| 1 | `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported` | Step 5, attempt 1, raised inside `peft/import_utils.py::is_torchao_available()` | **Yes** | Fixed by adding `pip install -q -U "torchao>=0.16.0"` to the top of the Step 5 cell. |
| 2 | `WARNING:torchao:Failed to load /usr/local/lib/python3.13/dist-packages/torchao/_C_cutlass_90a.abi3.so: Could not load this library` | Step 5, attempt 2 | **Not resolved; treated as non-blocking** | Optional torchao CUDA extension (Cutlass kernels), not required for this CPU-based merge path. |
| 3 | `WARNING:torchao:Failed to load /usr/local/lib/python3.13/dist-packages/torchao/_C_mxfp8.cpython-310-x86_64-linux-gnu.so: Could not load this library` | Step 5, attempt 2 | **Not resolved; treated as non-blocking** | Same category as #2. **Potential ABI/packaging issue:** filename embeds `cpython-310`, but the environment's site-packages path is `python3.13` — i.e., this specific compiled extension may have been built for a different Python ABI than the one running, which would explain the load failure. This is an inference from the filename and path strings, not a stated diagnosis in any tool output. |
| 4 | `NameError: name 'AutoModelForCausalLM' is not defined` | Step 5, attempt 2, at `base_model = AutoModelForCausalLM.from_pretrained(...)` | **Yes** | Root cause: Colab kernel was restarted between Step 5 attempts, so Steps 1–4's imports/variables no longer existed in memory, even though on-disk artifacts (`final_adapter/`) survived that particular restart. Fixed by making the Step 5 cell self-contained (re-imports, redefines `model_id`, fixes cwd, reloads/reuses tokenizer). |
| 5 | `WARNING:gguf.gguf_writer:Duplicated key name 'gemma3.context_le...'` (truncated) | Step 5, attempt 3 (the successful run), during GGUF metadata writing | **Unknown / not investigated** | Full key name and up to two more duplicate-key warnings (for keys resembling `gemma3.attention.*`) were observed but truncated in the rendered output; could not be recovered via DOM inspection (see §17 for the exact caveat). This may or may not be benign — flagged for downstream investigation (see §12, §15). |
| 6 | `-bash: ollama: command not found` | Colab **Terminal** panel (not a notebook cell), when `ollama create gemma3-cyber:v0.2 -f Modelfile.template` was run there | **Not applicable — expected behavior** | Ollama is not installed on the Colab VM; Step 6 instructions specify running this on the user's local machine after downloading the `.gguf` file. |
| 7 | `[transformers] \`torch_dtype\` is deprecated! Use \`dtype\` instead!` | Step 3 output, during `AutoModelForCausalLM.from_pretrained(...)` | **Not resolved; non-blocking** | Notebook code (Steps 3 and 5) still passes `torch_dtype=...` rather than `dtype=...`. Purely a forward-compatibility warning as of the `transformers` version installed at run time. |
| 8 | Runtime disconnect / disk reset between Step 5 attempt 3 (success) and this extraction | Colab runtime, exact trigger unknown | **Not resolved** | `/content` was found empty (only default `.config`/`sample_data`) at extraction time, despite Step 5 attempt 3 having produced `Merged model saved.` and the final export message earlier. Cause of the disconnect (idle timeout, manual restart, crash) is **unknown**. |

## 11. Confirmed Findings

- The dataset (`sft_v0.2.jsonl`, 277 examples) loaded and validated successfully with no duplicate IDs.
- QLoRA fine-tuning of `google/gemma-3-4b-it` completed all 3 configured epochs (105/105 steps) in 39 minutes 28 seconds on a single T4 GPU, without any observed training-time errors.
- Training loss decreased overall from 2.890367 (step 10) to its lowest observed point of 0.510340 (step 90), though not monotonically (see §12).
- The LoRA adapter (32,788,480 trainable parameters out of 4,332,867,952 total) was saved to `./final_adapter` and, at one point, verified present on disk with all expected files (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files).
- The merge-and-export pipeline (LoRA merge → FP16 save → GGUF conversion → Q4_K_M quantization) ran to its final success message (`Exported gemma3-cyber-v0.2-Q4_K_M.gguf ready for Ollama!`) at least once.
- Two distinct blocking errors were encountered and resolved during Step 5 across three attempts (a `torchao` version incompatibility, then a `NameError` from a kernel restart wiping in-memory state).
- No evaluation of the fine-tuned model's output quality, correctness, or safety has been performed.

## 12. Potential Issues

- **Colab disk ephemerality / lost artifacts:** By the time of this extraction, `/content` was empty — the trained adapter, merged model, and exported `.gguf` no longer exist in the Colab environment. Unless the `.gguf` (or `final_adapter`) was downloaded/copied out (e.g., to Google Drive or the user's local machine) before the disconnect, **this represents lost work that would require re-running Steps 1–5 in full**, including the ~40-minute training step.
- **Non-monotonic training loss:** Loss rose from 0.975393 (step 50) to 0.948494→0.993106 (steps 60→70), a mild increase, then fell to the observed minimum at step 90 (0.510340) before rising again to 0.685614 at step 100. With only 10 logged points over 105 steps and a small dataset (277 examples, effective batch size 8, i.e. ~35 steps/epoch), this pattern is consistent with ordinary mini-batch noise, but could also reflect learning-rate schedule effects near the end of training or instability. **Not enough information to distinguish between these — flagged as a potential issue, not a confirmed one.**
- **`torchao` ABI mismatch:** One of the two failed-to-load `torchao` shared libraries has `cpython-310` in its filename while the environment runs Python 3.13, suggesting a possible wheel/ABI mismatch for that specific optional extension. Unconfirmed as an inference beyond the filename/path evidence.
- **GGUF duplicate-key warnings:** Three `WARNING:gguf.gguf_writer:Duplicated key name '...'` messages appeared during the successful export. It is not established from the notebook whether these are cosmetic (e.g., a known upstream `convert_hf_to_gguf.py` quirk for Gemma-3 architecture metadata) or indicative of a metadata-correctness problem in the exported GGUF file.
- **No held-out/validation split during training:** The `SFTTrainer` configuration shown (Step 4) does not reference an evaluation dataset or `eval_strategy` — training loss is the only signal available during training; there is no validation loss to check for overfitting during training itself.
- **Small dataset size:** 277 examples for a 4B-parameter model fine-tune (even via LoRA) is small; generalization cannot be assessed without the (not-yet-run) held-out benchmark evaluations.
- **`ollama create` was attempted in the wrong environment** (Colab terminal instead of the user's local machine), which — while easily explained and not itself a bug in the notebook — indicates the Step 6 handoff instructions may benefit from being made more explicit about "local machine" meaning the user's own computer, not the Colab terminal.

## 13. Open Questions

- Does the exported `gemma3-cyber-v0.2-Q4_K_M.gguf` file currently exist anywhere outside of Colab (user's local disk, Google Drive, cloud storage)? **Unknown.**
- What are the actual results of `scripts/run_baseline.py` against `benchmark_v2.jsonl` and `benchmark_v3.jsonl` for both the base `gemma3:4b` model and `gemma3-cyber:v0.2`? **Not run.**
- Is the fine-tuned model actually more accurate on MITRE ATT&CK technique IDs than the base model (the stated motivating example, Kerberoasting = T1558.003)? **Not tested.**
- Does the "do-no-harm" regression anchor (`benchmark_v2`) pass — i.e., does fine-tuning on the narrow cybersecurity dataset degrade the model's general behavior? **Not tested.**
- Are the three `gguf.gguf_writer:Duplicated key name` warnings benign, or do they indicate a real metadata problem in the exported GGUF (e.g., double-written `gemma3.context_length` or `gemma3.attention.*` keys causing the wrong value to be read at load time)? **Not established.**
- What caused the training loss to rise between steps 50–70 and again at step 100? **Not established** (see §12).
- What caused the Colab runtime to disconnect/reset after the successful Step 5 run? **Unknown** (idle timeout, manual action, or crash are all plausible; no evidence distinguishes them).
- What exact `torchao`, `peft`, `transformers`, `trl`, `bitsandbytes`, `accelerate`, and `torch` versions were actually resolved by the unpinned Step 1 install? **Not captured** — no `pip freeze` or equivalent was run/observed.

## 14. Reproducibility Information

**Available:**
- Full LoRA/QLoRA/training hyperparameters (§6).
- Exact base model identifier (`google/gemma-3-4b-it`).
- Exact dataset path and example count (`data/training/sft_v0.2.jsonl`, 277 examples) and source repo (`https://github.com/novrusshehaj/Gemma4-CyberAi`).
- Exact merge/export commands and target filenames (Step 5 code).
- Exact GPU type used for this run (Tesla T4, 15GB).
- Confirmed on-disk artifact listing for `final_adapter/` at one point in time (§17), including file sizes.

**Missing / not available from the Colab notebook:**
- Pinned package versions for `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets` (install command is unpinned; no `pip freeze` captured).
- The resolved `torchao` version after the `>=0.16.0` upgrade.
- The actual content of a formatted training example (the "Sample formatted example" print output was not retrievable).
- Any evaluation results, so there is nothing yet to reproduce on the evaluation side.
- The exact llama.cpp commit/version used for conversion and quantization (cloned fresh from the default branch each time, no commit hash captured).
- Random seed effects cannot be assessed without a second run (seed=42 is fixed, but this is a single run).

## 15. Recommended Investigation Areas

*(Areas to investigate — not solutions; for a downstream expert agent to pursue.)*

- Determine whether `gemma3-cyber-v0.2-Q4_K_M.gguf` (or `final_adapter/`) survived anywhere outside the Colab runtime before re-running any part of the pipeline, to avoid unnecessary retraining.
- Run the two pending evaluations (`benchmark_v2.jsonl`, `benchmark_v3.jsonl`) for both `gemma3:4b` (base) and `gemma3-cyber:v0.2`, and record the results per the "do-no-harm guard + v3 hallucination/ATT&CK precision targets" success criteria referenced in the notebook (`docs/experiments/exp-002.md` — not itself inspected as part of this extraction).
- Investigate the cause and impact of the three `gguf.gguf_writer:Duplicated key name` warnings during GGUF export — confirm whether the exported file's metadata (e.g. `gemma3.context_length`, `gemma3.attention.*`) is correct.
- Investigate the non-monotonic training loss curve (rise at steps 60–70 and again at step 100) — consider whether more frequent logging, an eval split, or a learning-rate/scheduler adjustment would clarify whether this is noise or an instability.
- Pin and record exact package versions (`pip freeze`) for full reproducibility, especially `torchao`, `peft`, and `transformers`, given that a version-compatibility break between `peft` and `torchao` already occurred once in this project.
- Clarify/harden the Step 5→Step 6 handoff so it's unambiguous that `ollama create`/`ollama run` must be executed on the user's local machine, not the Colab terminal.
- Consider adding a step that automatically persists the exported `.gguf` (e.g., to Google Drive) immediately after Step 5 completes, given Colab's local disk is not durable across runtime resets.
- Given the small dataset size (277 examples), consider whether additional data, or a held-out validation split during training, would improve confidence in the training results.

## 16. Handoff Instructions for Opus/Grok/Cursor/Claude Code

This document (and its companion `cyber_ai_colab_results.json`) contains **extracted evidence** from a live inspection of the `colab_qlora_training.ipynb` Google Colab notebook and its Colab terminal, performed via browser automation. It is **not** a finished analysis or a set of recommendations to implement — it is source material.

- Treat every line under "Confirmed Findings" (§11) as directly observed.
- Treat "Potential Issues" (§12) and "Open Questions" (§13) as things that require further investigation, not established facts.
- Where this document says "Not available from the Colab notebook" or "Unknown," do not assume a value — either request it from the user, re-run the relevant step, or explicitly flag it as still missing in your own output.
- The JSON file (`cyber_ai_colab_results.json`) mirrors this report in structured form for programmatic consumption; the two should agree, but this Markdown file has the fuller prose context and the exact verbatim error/warning text.
- Two important caveats about extraction completeness are recorded in §17 — please read them before treating any "no error observed" claim as proof of full success.

## 17. Raw/Important Runtime Evidence

### 17.1 Extraction-completeness caveats (please read)

1. **Step 5's GGUF-conversion output is very long (hundreds of `INFO:hf-to-gguf:blk.N...` lines for a multi-layer transformer) and is rendered by Colab inside a virtualized, cross-origin sandboxed `<iframe>`.** This means: (a) standard DOM text extraction could not read it directly — only an accessibility-tree-based search tool could locate specific known strings inside it; (b) scrolling through it manually only renders whatever portion is currently near the viewport, and large stretches were skipped over during scrolling. Blocks 0–9 (of the model's full depth) were directly observed with no errors; the remaining blocks were not individually confirmed, though the final success message was confirmed present (see below).
2. **The exact text of the `trainable%` figure and the full duplicate-key GGUF warning names could not be recovered** — the rendering surface could not be scrolled horizontally via the available inspection tools. Where this matters, it is called out explicitly above rather than guessed.
3. All of the above means: the **absence** of a captured error in an unscrolled portion of output is not proof that no error occurred there. The **presence** of the final `Exported gemma3-cyber-v0.2-Q4_K_M.gguf ready for Ollama!` message (confirmed present via targeted search) is the strongest available evidence that the pipeline ran to completion.

### 17.2 Verbatim error text (Step 5, attempt 1)

```
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_3128/2436018827.py in <cell line: 0>()
      8     trust_remote_code=True
      9 )
---> 10 merged_model = PeftModel.from_pretrained(base_model, './final_adapter')
     11 merged_model = merged_model.merge_and_unload()
     12 merged_model.save_pretrained('./gemma3-cyber-v0.2-merged')

/usr/local/lib/python3.13/dist-packages/peft/import_utils.py in is_torchao_available()
    145
    146     if torchao_version < TORCHAO_MINIMUM_VERSION:
--> 147         raise ImportError(
    148             f"Found an incompatible version of torchao. Found version {torchao_version}, "
    149             f"but only versions above {TORCHAO_MINIMUM_VERSION} are supported"

ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported
```

### 17.3 Verbatim error text (Step 5, attempt 2)

```
WARNING:torchao:Failed to load /usr/local/lib/python3.13/dist-packages/torchao/_C_cutlass_90a.abi3.so: Could not load this library: /usr/local/lib/python3.13/dist-packages/torchao/_C_cutlass_90a.abi3.so
WARNING:torchao:Failed to load /usr/local/lib/python3.13/dist-packages/torchao/_C_mxfp8.cpython-310-x86_64-linux-gnu.so: Could not load this library: /usr/local/lib/python3.13/dist-packages/torchao/_C_mxfp8.cpython-310-x86_64-linux-gnu.so

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_32622/2776368899.py in <cell line: 0>()
      3 from peft import PeftModel
      4
----> 5 base_model = AutoModelForCausalLM.from_pretrained(
      6     model_id,
      7     torch_dtype=torch.float16,

NameError: name 'AutoModelForCausalLM' is not defined
```

### 17.4 Verbatim terminal transcript (disk state checks, at two different points in time)

**Earlier check (final_adapter still present, immediately after Step 4 succeeded and before Step 5's successful run):**
```
/content# ls -la /content && echo --- && ls -la /content/Gemma4-CyberAi/final_adapter
drwxr-xr-x 15 root root 4096 Aug 25 15:13 Gemma4-CyberAi
drwxr-xr-x  1 root root 4096 Aug 20 13:35 sample_data
---
total 96768
-rw-r--r--  1 root root     1153 Aug 25 15:13 adapter_config.json
-rw-------  1 root root 65673296 Aug 25 15:13 adapter_model.safetensors
-rw-r--r--  1 root root     1532 Aug 25 15:13 chat_template.jinja
-rw-r--r--  1 root root     5206 Aug 25 15:13 README.md
-rw-r--r--  1 root root      745 Aug 25 15:13 tokenizer_config.json
-rw-r--r--  1 root root 33384567 Aug 25 15:13 tokenizer.json
```

**Later check (after Step 5's successful run, at the time of this extraction):**
```
/content# ollama create gemma3-cyber:v0.2 -f Modelfile.template
-bash: ollama: command not found
/content# ls -la /content/
total 16
drwxr-xr-x 1 root root 4096 Aug 24 13:21 .
drwxr-xr-x 1 root root 4096 Aug 25 17:42 ..
drwxr-xr-x 4 root root 4096 Aug 24 13:21 .config
drwxr-xr-x 1 root root 4096 Aug 24 13:21 sample_data
```

### 17.5 Sample of GGUF conversion output (Step 5, attempt 3 — representative excerpt, not exhaustive)

```
INFO:hf-to-gguf:blk.4.ffn_down.weight,                              torch.float1[6]
INFO:hf-to-gguf:blk.4.ffn_gate.weight,                              torch.float1[6]
...
INFO:hf-to-gguf:blk.9.output_norm.weight,                           torch.float1[6]
INFO:hf-to-gguf:Set meta model
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:gguf: context length = 131072
INFO:hf-to-gguf:gguf: embedding length = 2560
INFO:hf-to-gguf:gguf: feed forward length = 10240
INFO:hf-to-gguf:gguf: head count = 8
INFO:hf-to-gguf:gguf: key-value head count = 4
INFO:hf-to-gguf:gguf: rope scaling type = LINEAR
INFO:hf-to-gguf:gguf: rope theta = 1000000.0
INFO:hf-to-gguf:gguf: rope theta swa = 10000.0
INFO:hf-to-gguf:gguf: rms norm epsilon = 1e-06
INFO:hf-to-gguf:gguf: file type = 1
WARNING:gguf.gguf_writer:Duplicated key name 'gemma3.context_le...' (truncated in source)
WARNING:gguf.gguf_writer:Duplicated key name 'gemma3.attention....' (truncated in source)
WARNING:gguf.gguf_writer:Duplicated key name 'gemma3.attention....' (truncated in source)
```

Final confirmed output line (located via targeted search, exact text):
```
Exported gemma3-cyber-v0.2-Q4_K_M.gguf ready for Ollama!
```

### 17.6 Step 5 cell code (final, working version, as of this extraction)

```python
!pip install -q -U "torchao>=0.16.0"
import os
if os.path.isdir('/content/Gemma4-CyberAi') and os.path.basename(os.getcwd()) != 'Gemma4-CyberAi': os.chdir('/content/Gemma4-CyberAi')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'google/gemma-3-4b-it'
tokenizer = globals().get('tokenizer') or AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
# Merge LoRA adapter into base model in FP16
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='cpu',
    trust_remote_code=True
)
merged_model = PeftModel.from_pretrained(base_model, './final_adapter')
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained('./gemma3-cyber-v0.2-merged')
tokenizer.save_pretrained('./gemma3-cyber-v0.2-merged')
print('Merged model saved.')
# Clone llama.cpp and convert to GGUF
!git clone https://github.com/ggerganov/llama.cpp
!pip install -q -r llama.cpp/requirements.txt
!python llama.cpp/convert_hf_to_gguf.py ./gemma3-cyber-v0.2-merged --outfile gemma3-cyber-v0.2.gguf --outtype f16
# Quantize to Q4_K_M for local Ollama serving
!cd llama.cpp && make llama-quantize
!./llama.cpp/llama-quantize gemma3-cyber-v0.2.gguf gemma3-cyber-v0.2-Q4_K_M.gguf Q4_K_M
print('Exported gemma3-cyber-v0.2-Q4_K_M.gguf ready for Ollama!')
```

*(Note: the leading `!pip install` line and the entire self-contained setup block above `# Merge LoRA adapter into base model in FP16` were added during this session's debugging to fix Runs 6 and 7; everything from `# Merge LoRA adapter...` onward is original notebook code.)*
