# exp-002r — Results (MEASURED 2026-08-27)

**Verdict: the candidate DOES NOT PASS the pre-registered bar.** The v2 do-no-harm
guard holds, but every targeted v3 objective (the whole reason exp-002 exists)
fails. This is a valid negative result (`configs/eval_success_criteria.md` §5) —
the first *measured* outcome for the project's first fine-tune. Goalposts unchanged.

## Runtime & arms

Four arms through the **same** harness / deterministic scorers / system prompt
(`BASELINE_SYSTEM_PROMPT`), temp 0, `num_predict` 512, on the **same MLX 4-bit
runtime** — the deltas isolate the LoRA adapter.

| Arm | Model | note |
|---|---|---|
| base | `mlx-community/gemma-3-4b-it-4bit` | exp-002r control. v2-test pass_rate **0.933 — identical to the Ollama `gemma3:4b` anchor**, cross-validating the runtime. |
| ep2 | fused iter-122 adapter (lowest validation loss, 1.048) | — |
| ep3 | fused iter-183 adapter (end of epoch 3, val loss 1.222) | **candidate of record** — validation loss is not predictive here; ep3 is clearly better on the benchmark. |

Training: 183 iters / 3 epochs, LoRA r=16 on 16 layers, completion-only loss,
seed 42. Validation loss 5.019 → 1.166 → **1.048 (ep2)** → 1.222 (ep3).

## benchmark_v2 (test, n=45) — frozen do-no-harm anchor

| Metric | base | ep2 | ep3 | Bar | ep3 result |
|---|---:|---:|---:|---|---|
| overall pass_rate | 0.933 | 0.889 | **0.956** | ≥ 0.913 (§4.2) | **PASS** (+1 item: `insuf-0103`) |
| overall mean_score | 0.826 | 0.781 | 0.869 | — | ↑ |
| hallucination (n=3) | 0.333 | 0.333 | 0.333 | ≥ 0.333 held-out (§4.1) | meets floor, **Δ = 0 → no gain** |
| insufficient_evidence (n=3) | 0.667 | 0.333 | 1.000 | must not regress (§4.3) | **PASS** |
| any category −>1 item | — | web_security, insuf | none | ≤ 1 item (§4.3) | **PASS** |

ep2 regressed v2 (0.889, below the 0.913 floor); ep3 does not. Use ep3.

## benchmark_v3 (test, n=12) — targeted ATT&CK / hallucination instrument

| Metric | base | ep2 | ep3 | Bar (§7) | ep3 result |
|---|---:|---:|---:|---|---|
| attack_mapping (n=4) | 0.250 | 0.000 | 0.000 | Δ ≥ **+0.40** | **FAIL** (−0.25, regressed) |
| `v3-attack-kerberoasting-t1060-trap` | FAIL (emits **T1060**) | FAIL | FAIL (still emits **T1060**) | must flip fail→pass | **FAIL** |
| false_premise (n=2) | 0.000 | 0.000 | 0.000 | Δ ≥ **+0.33** | **FAIL** (flat) |
| hallucination (n=2) | 0.500 | 1.000 | 1.000 | (secondary) | ↑ (`v3-halluc-product` resisted) |
| protocol_mechanics (n=2) | 0.000 | 0.500 | 0.500 | (secondary) | ↑ (`v3-proto-dns-tunnel`) |
| evidence_interpretation (n=1) | 1.000 | 0.000 | 0.000 | (secondary) | ↓ (`v3-ev-oom`) |
| overall pass_rate | 0.333 | 0.250 | 0.333 | — | flat |

## Interpretation

- **The core objective failed.** exp-002's hypothesis was that 27 registry-driven
  contrastive ATT&CK items would teach exact IDs and kill the "Kerberoasting →
  T1060" hallucination. After 3 epochs the model **still emits T1060** on the
  flagship trap, and `attack_mapping` pass_rate went *down* (0.25 → 0.00). Free-
  form prompts get the tactic right (Credential Access) and land near-miss IDs
  ("T1059.003") but not **T1558.003**.
- **Real but marginal abstention gains.** `v3-halluc-product` and
  `v3-proto-dns-tunnel` flipped to pass on both checkpoints — the
  hallucination-refusal / insufficient-evidence training did move *something*,
  just far too little (n=2 categories) to clear any bar.
- **Do-no-harm holds for ep3.** −0/+1 items on v2; no catastrophic forgetting at
  epoch 3 with this LoRA breadth. (ep2 did regress — validation-loss-based
  checkpoint selection was actively misleading here.)

## Consequence for the roadmap (drives exp-003+)

1. **Data scale first.** 277 examples is far below where a 4B LoRA generalises
   exact-fact recall instead of pattern-matching. Scale to 1.5–3k
   (`sft_v0.3`) before any further HP work — roadmap §8 exp-003.
2. **Contrastive-item density.** 27 ATT&CK contrastive items diluted across 277
   rows and 3 epochs produced no ID-precision gain. `sft_v0.3` needs a much
   larger, harder ATT&CK-precision family with the wrong-ID negatives spelled out
   per item.
3. **Checkpoint selection.** Select by a held-out *benchmark* run, not sft
   validation loss (they disagreed here). Add a dev-split eval to the train loop.
4. **Epochs / LR.** Keep 2–3 epochs but re-check after scaling; consider LR 1e-4.

Registry: `gemma3-cyber:v0.2` stays **experimental**, `passed_eval=false`,
`eval_ref` → this file.

## Artifacts (`SHA256SUMS.txt`)

| file | sha256 |
|---|---|
| `mlx_adapters/0000122_adapters.safetensors` (ep2) | `3a1168bde0a50b6616eb996528626c62d8a6a0e14544fb35521f77bbfa37077a` |
| `mlx_adapters/0000183_adapters.safetensors` (ep3) | `34e39c7bc27c8ff61e5db342afa0c1e00a5ca0f9f1aab8f15271dbac7d293539` |
| `fused_ep3_4bit/model.safetensors` | `1ea70da8a68526b1abbffb1eff6738319f961550348ee90ecea0bedd497ef702` |

Data: `sft_v0.2.jsonl` sha256 `1a9e0b59a38ad29f39957b0aa961eb4bd77d93c2d6f48dba6147809a6e254f2f`
(`mlx_adapters/train_manifest.json`).
