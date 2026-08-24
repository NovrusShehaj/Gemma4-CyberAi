"""Client tests. Network-dependent tests are skipped if Ollama is unavailable."""

import pytest

from gemma_cyber.clients import OllamaClient


def test_client_defaults():
    c = OllamaClient()
    assert c.model == "gemma3:4b"
    assert c.host == "http://localhost:11434"


def test_host_trailing_slash_stripped():
    c = OllamaClient(host="http://localhost:11434/")
    assert c.host == "http://localhost:11434"


@pytest.mark.skipif(
    not OllamaClient().is_available(), reason="Ollama service not reachable"
)
def test_generate_live_smoke():
    c = OllamaClient()
    if not c.has_model("gemma3:4b"):
        pytest.skip("gemma3:4b not pulled")
    result = c.generate("Reply with the single word: pong", num_predict=16)
    assert isinstance(result.text, str) and result.text.strip()
