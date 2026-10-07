# Copilot instructions for Gemma4-CyberAi

## Project shape

This is a Python 3.11–3.12 project managed with `uv`. It is a defensive/authorized-use
cybersecurity assistant built around a local Ollama `gemma3:4b` runtime. The repository
contains production-facing CLI, HTTP API/web UI, evaluation, model-registry, dataset, and
training-support code; the specialized model itself is not currently proven to outperform
the base model.

The main dependency direction is inward:

```text
CLI / FastAPI API / evaluation harness
              ↓
      gemma_cyber.inference
              ↓
     Ollama HTTP client/runtime
```

- `src/gemma_cyber/inference/` owns shared settings, deterministic generation defaults,
  retries/timeouts, health/readiness, structured inference errors, and registry resolution.
  All user-facing generation surfaces and the benchmark harness should use
  `InferenceEngine`; do not duplicate provider or retry logic in a surface layer.
- `src/gemma_cyber/clients/` is the low-level Ollama transport. It should remain unaware
  of FastAPI, CLI, evaluation, and agent concerns.
- `src/gemma_cyber/cli/` provides `gemma-cyber` (`ask`, `chat`, `eval`, `health`, and
  `models`) with stable exit codes. `src/gemma_cyber/api/` provides the versioned FastAPI
  `/v1` service and ships the self-contained web assets in `api/web/`.
- `src/gemma_cyber/evaluation/` runs frozen JSONL benchmarks through the same engine,
  applies deterministic scorers, and writes `results.json` plus `scorecard.md` under an
  experiment directory. The optional LLM judge supplements, but never replaces, the
  deterministic score.
- `src/gemma_cyber/inference/registry.py` and `data/models/registry.json` implement the
  auditable lifecycle `experimental → evaluated → candidate → production`. Promotion to
  `candidate` or `production` requires a recorded passing evaluation; hosted deployments
  normally treat the registry as read-only GitOps data.
- `src/gemma_cyber/data/`, `knowledge/`, and `training/` own dataset schemas/formatting,
  verified facts, contamination checks, and shared SFT/MLX helpers. Tracked evaluation
  benchmarks must not be used as training data.
- `src/gemma_cyber/agent/` is a separate, opt-in local terminal-agent product enabled by
  the `[agent]` extra. It is the only surface that can execute tools. The API, web UI,
  and `gemma-cyber` CLI must not import it; `tests/agent/test_import_boundary.py` enforces
  this separation.

## Setup and commands

Create/use the supported environment from the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

The normal CI-equivalent checks use the committed lockfile:

```bash
uv sync --locked --extra dev
uv run --no-sync ruff check src tests scripts
uv run --no-sync mypy src tests scripts
uv run --no-sync pytest
```

Run one test file, one test function, or a focused area with:

```bash
uv run --no-sync pytest tests/test_api.py
uv run --no-sync pytest tests/test_api.py::test_name
uv run --no-sync pytest tests/agent/
```

Security and packaging checks used by CI:

```bash
uv run --no-sync bandit -r src -ll
uv run --no-sync pip-audit --progress-spinner off --skip-editable
uv build
```

Tests are designed to be environment-safe and do not require Ollama, downloaded model
weights, or external APIs. For manual runtime checks, start Ollama and pull the model:

```bash
ollama serve
ollama pull gemma3:4b
gemma-cyber health
gemma-cyber-serve
```

Useful reproducible evaluation command:

```bash
gemma-cyber eval --benchmark data/evaluation/benchmark_v2.jsonl \
  --split test --out experiments/adhoc/v2-test --model gemma3:4b
```

Use `uv run ...` in automation and prefer `uv sync --locked`; changing
`pyproject.toml` requires updating `uv.lock` rather than silently resolving a new graph.

## Conventions and invariants

- Configuration is environment-driven through `GEMMA_CYBER_*`, with explicit constructor
  overrides taking precedence. Keep defaults deterministic (`temperature=0`, `seed=0`)
  and validate unsafe/invalid hosted configuration at startup via `Settings.validate()`.
  Never put secrets in the repository or logs; use redacted settings and structured logs.
- Preserve the shared safety-forward system prompt and the no-tools boundary on hosted
  surfaces. Do not add RAG, shell execution, target interaction, or agent imports to the
  FastAPI/API/web/`gemma-cyber` path without changing the documented security design and
  its boundary tests.
- API changes belong in the versioned `/v1` surface and should preserve request IDs,
  structured error responses, bounded input validation, security headers, auth/
  authorization behavior, rate limiting, and capacity controls. Do not expose runtime
  URLs, filesystem paths, tokens, prompts, or model responses in operational logs/errors.
  Production must fail closed if authentication is not configured.
- Model registry writes are deliberate lifecycle transitions, not arbitrary field edits.
  Keep provenance and evaluation references, preserve audit history, and use the existing
  atomic persistence/read-only behavior. A model must pass the evaluation gate before
  promotion.
- Evaluation output must remain reproducible and attributable: change one experimental
  variable at a time, retain benchmark/split/settings metadata, and do not overwrite the
  deterministic scorer with an LLM-judge result.
- The local agent is fail-closed by design: default mode is `read-only`; workspace jail,
  credential-path denial, destructive-command denial, output sanitization, approval, and
  audit behavior are security controls. Read `docs/agent.md` and `docs/security.md`
  before modifying `src/gemma_cyber/agent/`.
- API dependencies are optional; the base package should remain usable without FastAPI/
  uvicorn. Agent dependencies are also optional and must not leak into the hosted API
  import path. Keep lazy imports and the package/install-boundary tests intact.
- Web assets under `src/gemma_cyber/api/web/` are package runtime data. Changes must
  preserve wheel inclusion and the package CI smoke check.
- Follow the repository workflow: inspect → implement → run targeted tests, then the
  relevant full checks → update directly related docs. Keep changes surgical and do not
  treat experimental artifacts or generated caches as source.

## Authoritative references

- `README.md`: supported setup, product surfaces, structure, and safety scope.
- `docs/cli.md`, `docs/api.md`, `docs/agent.md`: command and runtime contracts.
- `docs/security.md`: security boundaries and required controls.
- `docs/deployment.md`: Docker Compose, hosted configuration, GitOps registry, and
  operational expectations.
- `.github/workflows/ci.yml`: authoritative CI commands and package/container checks.
- `pyproject.toml`: supported Python versions, extras, pytest/ruff/mypy configuration.
