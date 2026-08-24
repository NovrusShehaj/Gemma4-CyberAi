"""Harness tests using a stub client (no Ollama required)."""

import json

from gemma_cyber.evaluation.harness import run_benchmark
from gemma_cyber.clients.ollama_client import GenerationResult


class StubClient:
    """Deterministic fake client that always returns a fixed response."""

    model = "stub-model"

    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt, system=None, temperature=0.0, seed=0, num_predict=None):
        return GenerationResult(
            text=self._response, model=self.model, prompt=prompt,
            system=system, options={"temperature": temperature, "seed": seed},
        )


def _write_benchmark(path):
    rows = [
        {"id": "a1", "category": "cat1", "domain": "general", "scorer": "mcq",
         "question": "q", "choices": {"A": "x", "B": "y"}, "answer": "A"},
        {"id": "a2", "category": "cat2", "domain": "general", "scorer": "keyword",
         "question": "q", "expected_keywords": ["alpha", "beta"], "keyword_threshold": 0.5},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_run_benchmark_writes_outputs(tmp_path):
    bench = tmp_path / "b.jsonl"
    _write_benchmark(bench)
    out = tmp_path / "exp"
    report = run_benchmark(StubClient("Answer: A. alpha and beta."), bench, out)

    assert (out / "results.json").exists()
    assert (out / "scorecard.md").exists()
    assert report["overall"]["count"] == 2
    assert report["overall"]["pass_rate"] == 1.0
    assert report["model"] == "stub-model"
    assert set(report["by_category"]) == {"cat1", "cat2"}


def test_run_benchmark_records_failures(tmp_path):
    bench = tmp_path / "b.jsonl"
    _write_benchmark(bench)
    out = tmp_path / "exp"
    report = run_benchmark(StubClient("Answer: B. nothing relevant."), bench, out)
    # mcq wrong (chose B, expected A) and keywords missing -> both fail.
    assert report["overall"]["pass_rate"] == 0.0
