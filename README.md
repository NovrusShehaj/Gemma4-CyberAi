# Gemma4-CyberAi

A cybersecurity-specialized language model built on Google's open-weight **`gemma3:4b`**, served locally via **Ollama**, and specialized for defensive/blue-team analysis, cybersecurity education, and authorized CTF/lab-style reasoning.

> **Status:** Working product surfaces (CLI + HTTP API + web UI) over a shared,
> reproducible inference engine, with a versioned evaluation harness and a gated
> model-promotion registry. **The specialized model itself is not yet proven** —
> the first real fine-tune (exp-002) trained once but lost its artifacts before
> evaluation, so any "better than base" claim is still UNPROVEN. Training runs in
> the cloud (free Colab); the scaffold is ready to re-run.
> See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full roadmap, rationale, risks, and safety model.

This project is for **education, CTFs, defensive security, systems you own, and explicitly authorized testing only.** See [Safety](#safety).

---

## What this repo does right now

1. **Shared inference engine** (`gemma_cyber.inference`) — one provider-agnostic
   path to the model (retries, timeouts, health, streaming, env config, a gated
   model registry). The CLI, API, web UI, and the eval harness all use it.
2. **Professional CLI** (`gemma-cyber`) — ask/chat/eval/health/models with stable
   exit codes and deterministic defaults. See [`docs/cli.md`](docs/cli.md).
3. **HTTP API + web chat UI** (`gemma-cyber-serve`) — versioned `/v1` endpoints,
   streaming, input validation, opt-in auth + rate limiting, security headers, and
   a self-contained chat page. See [`docs/api.md`](docs/api.md).
4. **Model registry + promotion lifecycle** — `experimental → evaluated →
   candidate → production`, where promotion is **gated on a passing evaluation**.
5. **Reproducible evaluation** — frozen `benchmark_v2` (112 items, dev/test) +
   targeted `benchmark_v3` (47), hallucination / insufficient-evidence / factual
   scorers, an LLM-judge supplement, a contamination checker, and pre-registered
   success criteria. Same engine as production, so what you evaluate is what you serve.
6. **Cloud QLoRA training scaffold** — notebook + script sharing one tested
   masking module, pinned deps, run-manifest persistence, GGUF export verifier.
7. **CI** — ruff + mypy + `pytest` (156 tests) on every push/PR; environment-safe.

It does **not** (yet) do RAG or run agents/tools — those remain later, gated phases.

---

## Prerequisites

- **Ollama** installed and running (`ollama serve`), with the model pulled:
  ```bash
  ollama pull gemma3:4b
  ```
- **Python 3.11 or 3.12** (the ML stack lags newer Python; this repo pins `<3.13`).
- **[uv](https://github.com/astral-sh/uv)** for environment management.

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick start

```bash
# CLI — one-shot question (deterministic)
gemma-cyber ask "What ATT&CK technique is Kerberoasting?"
gemma-cyber chat                       # interactive streaming REPL
gemma-cyber health                     # runtime + model readiness (exit-coded)
gemma-cyber models list                # registry: versions + promotion stages

# Web API + chat UI
gemma-cyber-serve                      # http://127.0.0.1:8000/  (+ /docs)

# Evaluate any model through the SAME engine the UI uses
gemma-cyber eval --benchmark data/evaluation/benchmark_v2.jsonl \
  --split test --out experiments/adhoc/v2-test --model gemma3:4b

# Tests
pytest
```

Run the whole stack (API + Ollama) with Docker: `docker compose up -d --build`
(see [`docs/deployment.md`](docs/deployment.md)).

## Project structure

```
src/gemma_cyber/
  inference/       # SHARED engine: config, engine (retry/health/stream), registry
  clients/         # deterministic Ollama HTTP client (+ streaming)
  cli/             # gemma-cyber CLI (ask/chat/eval/health/models)
  api/             # FastAPI service + self-contained web chat UI (web/)
  evaluation/      # schema, scorers, harness, LLM judge
  data/            # training-data schema, builders, formatting, contamination
  knowledge/       # verified fact registry (ATT&CK IDs, single source of truth)
  training/        # shared SFT masking/config helpers (script + notebook)
scripts/           # run_baseline, train_qlora, verify_gguf_export, seed_registry, ...
data/evaluation/   # FROZEN benchmarks (tracked; never used for training)
data/models/       # model registry (tracked audit trail)
experiments/       # per-run scorecards
docs/              # cli, api, deployment, security, operations, commercialization, ...
```

## Development workflow

`Inspect → implement → test (pytest) → document → commit`. Change **one** variable per experiment so score deltas are attributable (`PROJECT_PLAN.md` §19).

## Safety

For education, CTFs, HTB/THM-style **authorized** labs, defensive security, research, and systems you own or are explicitly authorized to test. This project is **not** designed to autonomously attack arbitrary external systems. Future tool-using capabilities will be gated behind sandboxing, target allowlists, human approval, logging, and kill switches (`PROJECT_PLAN.md` §24).

**Training data note:** proprietary Hack The Box / TryHackMe content is **not** scraped or committed. See [`DATA_LICENSES.md`](DATA_LICENSES.md) and `PROJECT_PLAN.md` §16.

## License

Project code: Apache-2.0. Note that any model derived from Gemma remains subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its Prohibited Use Policy.
