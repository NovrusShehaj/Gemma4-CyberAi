# exp-002r — exp-002 reproduced locally (MLX-LoRA), the FIRST measured fine-tune

> **Status:** executed 2026-08-27 on an Apple M3 Max (128 GB) via `mlx-lm` LoRA.
> exp-002's Colab run trained but lost its artifacts before any evaluation
> (`docs/experiments/exp-002.md` §8). exp-002r closes that gap: it runs the same
> recipe on hardware that persists artifacts, then evaluates both arms under the
> pre-registered criteria (`configs/eval_success_criteria.md`).

## 1. What changed vs exp-002 (and why it is still the same experiment)

| Aspect | exp-002 (Colab, lost) | exp-002r (local, this run) | Same? |
|---|---|---|---|
| Base model | `google/gemma-3-4b-it` | `mlx-community/gemma-3-4b-it-4bit` (the 4-bit MLX conversion of the same weights) | equivalent |
| Method | QLoRA (4-bit NF4 base + bf16 LoRA), transformers/PEFT/TRL | LoRA on 4-bit base + bf16 adapters, `mlx-lm` | equivalent recipe |
| Dataset | `sft_v0.2.jsonl` (277 items) | `sft_v0.2.jsonl` (277 items), same file, same SHA | **identical** |
| Chat format | `to_gemma_chat_text` (system folded into first user turn) | gemma-3-4b-it tokenizer chat template — **verified byte-identical** to `to_gemma_chat_text` | identical |
| Loss | completion-only (prompt masked) | `mask_prompt: true` (prompt masked) | identical |
| LoRA rank / alpha ratio | r=16, alpha=32 (scale 2.0) | rank 16, scale 2.0 | identical |
| Epochs / seq len | 3 / 1024 | 3 / 1024 | identical |
| LoRA breadth | all-linear, all layers | last 16 layers, 7 projection types (mlx default; roadmap §7.2) | **narrower** |
| Effective batch | 8 (2 × grad-accum 4) | 4 (batch 4, no accum) | **smaller** |

The two deviations (LoRA on 16 layers not 34; effective batch 4 not 8) are the
documented Apple-Silicon-practical settings from the roadmap and were forced by
throughput on this machine. They make exp-002r a slightly *weaker* intervention
than exp-002 would have been, not a stronger one — so a pass here is a
conservative result, and a fail is not conclusive that the full-breadth recipe
would also fail. Config: `configs/training/mlx_lora_gemma3_4b_v0.2.yaml`.

## 2. Artifacts (persisted immediately — the exp-002 failure mode)

```
experiments/exp-002r-gemma3-cyber-v0.2/
  mlx_adapters/                 adapters.safetensors + adapter_config.json + train_manifest.json
  fused_model/                  standalone HF-format merged model (mlx_lm.fuse)
  base-v2-test/  v0.2-v2-test/  base + candidate scorecards, benchmark_v2 test split
  base-v3-test/  v0.2-v3-test/  base + candidate scorecards, benchmark_v3 test split
```

Data manifest (`data/training/mlx/sft_v0.2/manifest.json`) records the SHA-256 of
`sft_v0.2.jsonl` and of each generated split, so the exact train/valid partition
is reproducible.

## 3. Commands

```bash
python -m pip install -r configs/training/requirements-train-mlx.txt   # Apple Silicon
python scripts/train_mlx_lora.py -c configs/training/mlx_lora_gemma3_4b_v0.2.yaml --epochs 3
printf 'FROM experiments/exp-002r-gemma3-cyber-v0.2/fused_model\n' > /tmp/Modelfile.v0.2
ollama create gemma3-cyber:v0.2 -f /tmp/Modelfile.v0.2

for bench in v2 v3; do for model in "gemma3:4b:base" "gemma3-cyber:v0.2:v0.2"; do
  tag=${model%%:*}:${model#*:}; tag=${model%:*}; name=${model##*:}
  python scripts/run_baseline.py --model "$tag" \
    --benchmark data/evaluation/benchmark_$bench.jsonl --split test \
    --out experiments/exp-002r-gemma3-cyber-v0.2/$name-$bench-test
done; done
```

## 4. Results — see `experiments/exp-002r-gemma3-cyber-v0.2/RESULTS.md`

**Measured 2026-08-27. Verdict: DOES NOT PASS.**

| | base (MLX 4-bit) | candidate (ep3) |
|---|---:|---:|
| benchmark_v2 test overall | 0.933 | **0.956** (do-no-harm §4.2 PASS) |
| benchmark_v2 hallucination (n=3) | 0.333 | 0.333 (no gain) |
| benchmark_v3 attack_mapping (n=4) | 0.250 | **0.000** (§7 FAIL) |
| Kerberoasting T1060 trap | fails | **still fails** (emits T1060) — §7 FAIL |
| benchmark_v3 false_premise (n=2) | 0.000 | 0.000 (§7 FAIL) |
| benchmark_v3 hallucination (n=2) | 0.500 | 1.000 (secondary ↑) |

The v2 do-no-harm guard holds; every targeted v3 objective fails. 277 examples ×
3 epochs did not teach exact ATT&CK IDs. Full analysis and the exp-003 corrective
plan are in `RESULTS.md`.

## 5. Verdict against pre-registered criteria

`configs/eval_success_criteria.md` §4 (v2, frozen): **§4.1 not met** (hallucination
Δ = 0), §4.2 met (ep3), §4.3 met (ep3). §7 addendum (v3): **all three targets
missed.** Negative result recorded; goalposts unchanged.
