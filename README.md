# Gemma4-CyberAi

A cybersecurity-specialized language model built on Google's open-weight **`gemma3:4b`**, served locally via **Ollama**, and specialized for defensive/blue-team analysis, cybersecurity education, and authorized CTF/lab-style reasoning.

> **Status:** Milestone 1 complete — deterministic Ollama client + frozen evaluation benchmark + baseline harness. No fine-tuning yet (by design).
> See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full roadmap, rationale, risks, and safety model.

This project is for **education, CTFs, defensive security, systems you own, and explicitly authorized testing only.** See [Safety](#safety).

---

## What this repo does right now

1. Talks to `gemma3:4b` locally through Ollama (deterministic, reproducible).
2. Runs a small, **frozen, original** cybersecurity benchmark (`data/evaluation/benchmark_v1.jsonl`) — including hallucination and "insufficient-evidence" traps.
3. Produces a **baseline scorecard** so we can later prove whether specialization actually helped.

It does **not** (yet) fine-tune, do RAG, or run agents/tools — those are later phases in `PROJECT_PLAN.md`.

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
# 1. Smoke-test inference
python scripts/chat.py "What is lateral movement?"

# 2. Run the baseline benchmark against gemma3:4b
python scripts/run_baseline.py
#   -> writes experiments/baseline_gemma3-4b/{results.json,scorecard.md}

# 3. Run the test suite
pytest
```

Later, to evaluate a fine-tuned model with the *same* benchmark:

```bash
python scripts/run_baseline.py --model gemma3-cyber:v0.1 --out experiments/exp-001
```

## Project structure

```
src/gemma_cyber/
  clients/ollama_client.py   # deterministic Ollama HTTP client (inference only)
  evaluation/schema.py       # benchmark item schema + loader (pydantic)
  evaluation/scorers.py      # deterministic scorers (mcq/keyword/insufficient/hallucination)
  evaluation/harness.py      # runs benchmark -> results.json + scorecard.md
scripts/                     # run_baseline.py, chat.py
data/evaluation/             # FROZEN benchmark (tracked; never used for training)
experiments/                 # per-run scorecards (baseline, future fine-tunes)
tests/                       # schema/scorer/harness/client tests
PROJECT_PLAN.md              # full roadmap
DATA_LICENSES.md             # data source -> license map
```

## Development workflow

`Inspect → implement → test (pytest) → document → commit`. Change **one** variable per experiment so score deltas are attributable (`PROJECT_PLAN.md` §19).

## Safety

For education, CTFs, HTB/THM-style **authorized** labs, defensive security, research, and systems you own or are explicitly authorized to test. This project is **not** designed to autonomously attack arbitrary external systems. Future tool-using capabilities will be gated behind sandboxing, target allowlists, human approval, logging, and kill switches (`PROJECT_PLAN.md` §24).

**Training data note:** proprietary Hack The Box / TryHackMe content is **not** scraped or committed. See [`DATA_LICENSES.md`](DATA_LICENSES.md) and `PROJECT_PLAN.md` §16.

## License

Project code: Apache-2.0. Note that any model derived from Gemma remains subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its Prohibited Use Policy.
