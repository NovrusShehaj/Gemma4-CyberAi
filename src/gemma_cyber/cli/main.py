"""gemma-cyber CLI — chat, ask, eval, health, and model-registry management.

Design:
  * Every model call goes through :class:`~gemma_cyber.inference.InferenceEngine`,
    so the CLI shares the exact inference path used by the API and the benchmark.
  * Deterministic by default (temperature/seed from settings) — a scripted
    ``gemma-cyber ask`` is reproducible.
  * Scripting-friendly: stable exit codes, ``--json`` machine output, secrets
    never printed. ``--debug`` turns on structured logs to stderr.
  * The CLI is defensive-only by construction: it just serves the model with the
    project's safety-forward system prompt; it exposes no target interaction,
    no command execution, and no offensive tooling.

Exit codes:
    0  success
    1  generic runtime error
    2  model runtime unreachable (service down)
    3  requested model/version unavailable
    4  usage / bad arguments
    5  evaluation gate not satisfied (eval command, when --gate is set)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from gemma_cyber.inference import (
    InferenceEngine,
    ModelRegistry,
    Settings,
    load_settings,
)
from gemma_cyber.inference.errors import (
    InferenceError,
    ModelUnavailableError,
    RegistryError,
    ServiceUnavailableError,
)
from gemma_cyber.inference.registry import STAGES as _STAGES
from gemma_cyber.inference.registry import ModelRecord

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SERVICE_DOWN = 2
EXIT_MODEL_UNAVAILABLE = 3
EXIT_USAGE = 4
EXIT_GATE = 5

PROG = "gemma-cyber"


# -- construction helpers (patched in tests) --------------------------------

def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "host", None):
        overrides["ollama_host"] = args.host
    for name in ("temperature", "seed", "num_predict"):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val
    return load_settings(**overrides)


def make_engine(args: argparse.Namespace) -> InferenceEngine:
    """Build the shared engine from CLI args + environment. Patched in tests."""
    return InferenceEngine.from_settings(_settings_from_args(args), model=getattr(args, "model", None))


def make_registry(args: argparse.Namespace) -> ModelRegistry:
    settings = _settings_from_args(args)
    return ModelRegistry(settings.registry_path)


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


# -- commands ---------------------------------------------------------------

def cmd_version(args: argparse.Namespace) -> int:
    from gemma_cyber import __version__

    settings = _settings_from_args(args)
    info = {
        "version": __version__,
        "resolved_model": settings.model,
        "ollama_host": settings.ollama_host,
        "environment": settings.environment,
    }
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"{PROG} {__version__}")
        print(f"  model:       {settings.model}")
        print(f"  host:        {settings.ollama_host}")
        print(f"  environment: {settings.environment}")
    return EXIT_OK


def cmd_health(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    status = engine.health()
    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        mark = "OK" if status.ok else "NOT READY"
        print(f"[{mark}] model={status.model} host={status.host}")
        print(f"  service_reachable: {status.service_reachable}")
        print(f"  model_present:     {status.model_present}")
        if status.detail:
            print(f"  detail:            {status.detail}")
    if status.ok:
        return EXIT_OK
    return EXIT_SERVICE_DOWN if not status.service_reachable else EXIT_MODEL_UNAVAILABLE


def cmd_ask(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        _eprint("error: empty prompt")
        return EXIT_USAGE

    # Only override the system prompt when explicitly asked; otherwise let the
    # engine apply the settings default.
    kwargs: dict[str, Any] = {}
    if args.no_system:
        kwargs["system"] = None
    elif args.system:
        kwargs["system"] = args.system

    try:
        if args.stream and not args.json:
            for chunk in engine.stream(prompt, **kwargs):
                if chunk.text:
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
            print()
            return EXIT_OK
        result = engine.generate(prompt, **kwargs)
    except ModelUnavailableError as exc:
        _eprint(f"error: {exc}")
        return EXIT_MODEL_UNAVAILABLE
    except ServiceUnavailableError as exc:
        _eprint(f"error: {exc}")
        return EXIT_SERVICE_DOWN
    except InferenceError as exc:
        _eprint(f"error: {exc}")
        return EXIT_ERROR

    if args.json:
        print(json.dumps({
            "model": result.model,
            "prompt": result.prompt,
            "response": result.text,
            "options": result.options,
        }, indent=2))
    else:
        print(result.text.strip())
    return EXIT_OK


def cmd_chat(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    ready = engine.health()
    if not ready.ok:
        _eprint(f"error: {ready.detail}")
        return EXIT_SERVICE_DOWN if not ready.service_reachable else EXIT_MODEL_UNAVAILABLE
    print(f"{PROG} chat — model {engine.model}. Ctrl-D or /exit to quit.\n")
    while True:
        try:
            line = input("you> ")
        except EOFError:
            print()
            break
        if line.strip() in ("/exit", "/quit"):
            break
        if not line.strip():
            continue
        try:
            sys.stdout.write("bot> ")
            for chunk in engine.stream(line):
                if chunk.text:
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
            print("\n")
        except InferenceError as exc:
            _eprint(f"\nerror: {exc}")
    return EXIT_OK


def cmd_models(args: argparse.Namespace) -> int:
    try:
        reg = make_registry(args)
    except RegistryError as exc:
        _eprint(f"error: {exc}")
        return EXIT_ERROR

    sub = args.models_command
    if sub == "list":
        records = reg.list(stage=args.stage)
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2))
        else:
            if not records:
                print("(no registered models)")
            for r in records:
                gate = "✓" if r.passed_eval else "·"
                print(f"{r.stage:13s} {gate} {r.version:24s} base={r.base_model} "
                      f"dataset={r.dataset_version or '-'}")
        return EXIT_OK

    if sub == "show":
        try:
            rec = reg.get(args.version)
        except RegistryError as exc:
            _eprint(f"error: {exc}")
            return EXIT_USAGE
        print(json.dumps(rec.to_dict(), indent=2))
        return EXIT_OK

    if sub == "promote":
        try:
            rec = reg.promote(args.version, args.to, reason=args.reason)
        except RegistryError as exc:
            _eprint(f"error: {exc}")
            return EXIT_USAGE
        print(f"promoted {rec.version} -> {rec.stage}")
        return EXIT_OK

    if sub == "mark-evaluated":
        try:
            rec = reg.mark_evaluated(args.version, passed=args.passed, eval_ref=args.eval_ref)
        except RegistryError as exc:
            _eprint(f"error: {exc}")
            return EXIT_USAGE
        print(f"{rec.version}: passed_eval={rec.passed_eval} stage={rec.stage}")
        return EXIT_OK

    if sub == "register":
        rec = ModelRecord(
            version=args.version,
            base_model=args.base_model,
            dataset_version=args.dataset_version,
            experiment=args.experiment,
            notes=args.notes,
        )
        try:
            reg.register(rec, overwrite=args.overwrite)
        except RegistryError as exc:
            _eprint(f"error: {exc}")
            return EXIT_USAGE
        print(f"registered {rec.version} [{rec.stage}]")
        return EXIT_OK

    _eprint("error: no models subcommand")
    return EXIT_USAGE


def cmd_eval(args: argparse.Namespace) -> int:
    """Run a benchmark through the shared engine — the same path chat uses."""
    from gemma_cyber.evaluation.harness import run_benchmark

    engine = make_engine(args)
    ready = engine.health()
    if not ready.ok:
        _eprint(f"error: {ready.detail}")
        return EXIT_SERVICE_DOWN if not ready.service_reachable else EXIT_MODEL_UNAVAILABLE
    try:
        report = run_benchmark(
            engine,
            benchmark_path=args.benchmark,
            out_dir=args.out,
            split=args.split,
            experiment_name=args.experiment,
        )
    except (ValueError, FileNotFoundError) as exc:
        _eprint(f"error: {exc}")
        return EXIT_USAGE
    except InferenceError as exc:
        _eprint(f"error: {exc}")
        return EXIT_ERROR

    overall = report["overall"] if "overall" in report else report.get("aggregates", {}).get("overall", {})
    pass_rate = overall.get("pass_rate")
    print(f"eval complete: model={report.get('model')} "
          f"pass_rate={pass_rate} -> {args.out}")
    if args.gate is not None and pass_rate is not None and pass_rate < args.gate:
        _eprint(f"gate FAILED: pass_rate {pass_rate} < required {args.gate}")
        return EXIT_GATE
    return EXIT_OK


# -- parser -----------------------------------------------------------------

def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", help="Model version/tag or registry alias "
                   "(e.g. gemma3:4b, gemma3-cyber:v0.2, production).")
    p.add_argument("--host", help="Ollama host URL (overrides GEMMA_CYBER_OLLAMA_HOST).")


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, help="Sampling temperature (default 0).")
    p.add_argument("--seed", type=int, help="Random seed (default 0).")
    p.add_argument("--num-predict", type=int, dest="num_predict",
                   help="Max output tokens (default 512).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Cybersecurity assistant CLI over the shared inference engine "
                    "(defensive/authorized use only).",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logs to stderr.")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Answer a single question and exit.")
    p_ask.add_argument("prompt", nargs="?", help="The question (or read from stdin).")
    p_ask.add_argument("--stream", action="store_true", help="Stream tokens as they arrive.")
    p_ask.add_argument("--system", help="Override the system prompt.")
    p_ask.add_argument("--no-system", action="store_true", help="Send no system prompt.")
    _add_common_model_args(p_ask)
    _add_gen_args(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive streaming chat.")
    _add_common_model_args(p_chat)
    _add_gen_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_health = sub.add_parser("health", help="Check runtime + model readiness.")
    _add_common_model_args(p_health)
    p_health.set_defaults(func=cmd_health)

    p_ver = sub.add_parser("version", help="Show version and resolved config.")
    _add_common_model_args(p_ver)
    p_ver.set_defaults(func=cmd_version)

    p_eval = sub.add_parser("eval", help="Run a benchmark via the shared engine.")
    p_eval.add_argument("--benchmark", required=True, help="Path to a benchmark JSONL.")
    p_eval.add_argument("--out", required=True, help="Output directory for results.")
    p_eval.add_argument("--split", choices=["dev", "test"], help="Only this split.")
    p_eval.add_argument("--experiment", help="Experiment name for the scorecard.")
    p_eval.add_argument("--gate", type=float,
                        help="Fail (exit 5) if overall pass_rate < this value.")
    _add_common_model_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_models = sub.add_parser("models", help="Inspect/manage the model registry.")
    msub = p_models.add_subparsers(dest="models_command", required=True)
    m_list = msub.add_parser("list", help="List registered model versions.")
    m_list.add_argument("--stage", choices=list(_STAGES), help="Filter by stage.")
    m_show = msub.add_parser("show", help="Show one version's full record.")
    m_show.add_argument("version")
    m_prom = msub.add_parser("promote", help="Promote a version to a stage (gated).")
    m_prom.add_argument("version")
    m_prom.add_argument("--to", required=True, choices=list(_STAGES))
    m_prom.add_argument("--reason")
    m_mark = msub.add_parser("mark-evaluated", help="Record an eval outcome (gate input).")
    m_mark.add_argument("version")
    m_mark.add_argument("--passed", action="store_true", help="The eval passed.")
    m_mark.add_argument("--eval-ref", dest="eval_ref", help="Path to the scorecard.")
    m_reg = msub.add_parser("register", help="Register a new model version.")
    m_reg.add_argument("version")
    m_reg.add_argument("--base-model", dest="base_model", default="gemma3:4b")
    m_reg.add_argument("--dataset-version", dest="dataset_version")
    m_reg.add_argument("--experiment")
    m_reg.add_argument("--notes")
    m_reg.add_argument("--overwrite", action="store_true")
    p_models.set_defaults(func=cmd_models)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "debug", False))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _eprint("\ninterrupted")
        return EXIT_ERROR
    except BrokenPipeError:  # piped into head, etc.
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
