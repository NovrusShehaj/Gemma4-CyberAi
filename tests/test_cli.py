"""CLI tests: dispatch, exit codes, output modes, registry commands.

The engine is faked so the CLI is tested with no Ollama. Commands that touch a
model patch ``make_engine`` to return a scriptable engine; registry commands use
a real ModelRegistry on a tmp file (pointed at via GEMMA_CYBER_REGISTRY_PATH).
"""

from __future__ import annotations

import json

import pytest

import gemma_cyber.cli.main as cli
from gemma_cyber.inference.engine import HealthStatus, StreamChunk


class FakeEngine:
    def __init__(self, *, ok=True, service=True, model_present=True,
                 text="fake answer", raise_exc=None):
        self.model = "gemma3:4b"
        self._ok = ok
        self._service = service
        self._present = model_present
        self._text = text
        self._raise = raise_exc

    def health(self):
        return HealthStatus(
            ok=self._ok, service_reachable=self._service,
            model_present=self._present, model=self.model, host="http://fake",
            detail="" if self._ok else "not ready",
        )

    def generate(self, prompt, **kw):
        if self._raise:
            raise self._raise
        from gemma_cyber.clients.ollama_client import GenerationResult
        return GenerationResult(text=self._text, model=self.model, prompt=prompt,
                                system=kw.get("system"), options={})

    def stream(self, prompt, **kw):
        if self._raise:
            raise self._raise
        for piece in ("fa", "ke"):
            yield StreamChunk(request_id="rid", text=piece)
        yield StreamChunk(request_id="rid", text="", done=True)


@pytest.fixture
def patch_engine(monkeypatch):
    def _install(engine):
        monkeypatch.setattr(cli, "make_engine", lambda args: engine)
        return engine
    return _install


def test_version_json(capsys):
    rc = cli.main(["--json", "version"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert "version" in out and "resolved_model" in out


def test_ask_plain(patch_engine, capsys):
    patch_engine(FakeEngine(text="lateral movement is ..."))
    rc = cli.main(["ask", "what is lateral movement?"])
    assert rc == 0
    assert "lateral movement is" in capsys.readouterr().out


def test_ask_json(patch_engine, capsys):
    patch_engine(FakeEngine(text="answer"))
    rc = cli.main(["--json", "ask", "q"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["response"] == "answer"


def test_ask_empty_prompt_is_usage_error(patch_engine):
    patch_engine(FakeEngine())
    rc = cli.main(["ask", "   "])
    assert rc == cli.EXIT_USAGE


def test_ask_stream(patch_engine, capsys):
    patch_engine(FakeEngine())
    rc = cli.main(["ask", "--stream", "q"])
    assert rc == 0
    assert "fake" in capsys.readouterr().out


def test_ask_service_down_exit_code(patch_engine):
    from gemma_cyber.inference.errors import ServiceUnavailableError
    patch_engine(FakeEngine(raise_exc=ServiceUnavailableError("down")))
    rc = cli.main(["ask", "q"])
    assert rc == cli.EXIT_SERVICE_DOWN


def test_ask_model_unavailable_exit_code(patch_engine):
    from gemma_cyber.inference.errors import ModelUnavailableError
    patch_engine(FakeEngine(raise_exc=ModelUnavailableError("no model")))
    rc = cli.main(["ask", "q"])
    assert rc == cli.EXIT_MODEL_UNAVAILABLE


def test_health_ok(patch_engine, capsys):
    patch_engine(FakeEngine(ok=True))
    rc = cli.main(["health"])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_health_service_down(patch_engine):
    patch_engine(FakeEngine(ok=False, service=False, model_present=False))
    rc = cli.main(["health"])
    assert rc == cli.EXIT_SERVICE_DOWN


def test_health_model_missing(patch_engine):
    patch_engine(FakeEngine(ok=False, service=True, model_present=False))
    rc = cli.main(["health"])
    assert rc == cli.EXIT_MODEL_UNAVAILABLE


# -- registry commands (real registry on tmp file) --------------------------

@pytest.fixture
def registry_env(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    monkeypatch.setenv("GEMMA_CYBER_REGISTRY_PATH", str(path))
    return path


def test_models_register_list_show(registry_env, capsys):
    assert cli.main(["models", "register", "gemma3-cyber:v0.2",
                     "--dataset-version", "sft_v0.2"]) == 0
    assert cli.main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "gemma3-cyber:v0.2" in out and "experimental" in out
    assert cli.main(["models", "show", "gemma3-cyber:v0.2"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["dataset_version"] == "sft_v0.2"


def test_models_promote_gate_blocks_ungated(registry_env):
    cli.main(["models", "register", "m"])
    # Cannot jump to candidate without a passing eval.
    rc = cli.main(["models", "promote", "m", "--to", "candidate"])
    assert rc == cli.EXIT_USAGE


def test_models_full_promotion_flow(registry_env, capsys):
    cli.main(["models", "register", "m"])
    assert cli.main(["models", "mark-evaluated", "m", "--passed",
                     "--eval-ref", "x/scorecard.md"]) == 0
    assert cli.main(["models", "promote", "m", "--to", "candidate"]) == 0
    assert cli.main(["models", "promote", "m", "--to", "production"]) == 0
    capsys.readouterr()
    cli.main(["models", "show", "m"])
    rec = json.loads(capsys.readouterr().out)
    assert rec["stage"] == "production"


def test_no_command_errors():
    with pytest.raises(SystemExit):
        cli.main([])
