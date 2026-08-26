# `gemma-cyber` CLI

A scriptable command-line interface over the shared inference engine
(`gemma_cyber.inference`). The CLI, the web/API service, and the evaluation
harness all generate through the **same** engine, so behavior is identical across
surfaces. Defensive/authorized use only — the CLI serves the model with the
project's safety-forward system prompt and exposes no target interaction or
command execution.

## Install / run

```bash
# From the repo, with the project installed (editable):
uv pip install -e .
gemma-cyber version

# Or without installing, straight from source:
PYTHONPATH=src python -m gemma_cyber.cli.main version
```

## Configuration (environment variables)

All settings resolve from `GEMMA_CYBER_*` env vars, overridable per-command by
flags. Defaults are deterministic (temperature 0, seed 0) so scripted runs are
reproducible.

| Variable | Default | Meaning |
|---|---|---|
| `GEMMA_CYBER_OLLAMA_HOST` | `http://localhost:11434` | Model runtime URL |
| `GEMMA_CYBER_MODEL` | `gemma3:4b` | Model tag or registry alias (`production`, a version) |
| `GEMMA_CYBER_TEMPERATURE` | `0.0` | Sampling temperature |
| `GEMMA_CYBER_SEED` | `0` | Random seed |
| `GEMMA_CYBER_NUM_PREDICT` | `512` | Max output tokens |
| `GEMMA_CYBER_TIMEOUT` | `180` | Per-request timeout (s) |
| `GEMMA_CYBER_MAX_RETRIES` | `2` | Retries on transient service errors |
| `GEMMA_CYBER_REGISTRY_PATH` | `data/models/registry.json` | Model registry file |
| `GEMMA_CYBER_ENV` | `dev` | `dev` / `test` / `prod` |
| `GEMMA_CYBER_LOG_LEVEL` | `INFO` | Log level |

## Commands

### `ask` — one-shot question
```bash
gemma-cyber ask "What ATT&CK technique is Kerberoasting?"
gemma-cyber ask --stream "Explain lateral movement."      # stream tokens
echo "long prompt on stdin" | gemma-cyber ask             # read stdin
gemma-cyber --json ask "..." > out.json                   # machine-readable
gemma-cyber ask --model gemma3-cyber:v0.2 "..."           # pick a version
gemma-cyber ask --no-system "..."                         # send no system prompt
```

### `chat` — interactive streaming REPL
```bash
gemma-cyber chat
gemma-cyber chat --model production        # serve the promoted model
```

### `health` — runtime + model readiness (probe)
```bash
gemma-cyber health          # exit 0 ready, 2 service down, 3 model missing
gemma-cyber --json health
```

### `eval` — run a benchmark through the shared engine
Same inference path as `ask`/`chat`, so what you evaluate is what you serve.
```bash
gemma-cyber eval --benchmark data/evaluation/benchmark_v2.jsonl \
  --split test --out experiments/adhoc/v2-test --model gemma3:4b
gemma-cyber eval --benchmark data/evaluation/benchmark_v3.jsonl \
  --out /tmp/run --gate 0.9        # exit 5 if overall pass_rate < 0.9
```

### `models` — model registry + promotion lifecycle
Versions move `experimental → evaluated → candidate → production`. Promotion to
candidate/production is **gated**: it requires a recorded passing evaluation.
```bash
gemma-cyber models list
gemma-cyber models list --stage production
gemma-cyber models show gemma3-cyber:v0.2
gemma-cyber models register gemma3-cyber:v0.3 --dataset-version sft_v0.3 \
  --experiment exp-003 --base-model gemma3:4b
gemma-cyber models mark-evaluated gemma3-cyber:v0.3 --passed \
  --eval-ref experiments/exp-003/scorecard.md
gemma-cyber models promote gemma3-cyber:v0.3 --to candidate
gemma-cyber models promote gemma3-cyber:v0.3 --to production   # demotes incumbent
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | generic runtime error |
| 2 | model runtime unreachable |
| 3 | requested model/version unavailable |
| 4 | usage / bad arguments |
| 5 | evaluation gate not satisfied (`eval --gate`) |

Stable exit codes make the CLI safe to drive from CI (e.g. gate a build on a
benchmark pass_rate, or block a deploy when `health` is not ready).
```

## Debugging

`--debug` turns on structured logs (request ids, retries) to stderr:
```bash
gemma-cyber --debug ask "..."
```
