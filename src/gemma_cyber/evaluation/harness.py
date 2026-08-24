"""Evaluation harness: run a model over a benchmark and produce a scorecard.

Outputs (written under an experiment directory):
  * results.json  -- full per-item results + aggregates + run metadata
  * scorecard.md  -- human-readable summary, diffable in git

The same harness is used for the base model and any future fine-tuned model, so
comparisons are apples-to-apples (PROJECT_PLAN.md §18, §19).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from gemma_cyber.evaluation.schema import BenchmarkItem, load_benchmark
from gemma_cyber.evaluation.scorers import ScoreResult, score_item

# Neutral system prompt used for the BASELINE. Kept minimal on purpose: the
# baseline should reflect the base model's own ability, lightly framed.
BASELINE_SYSTEM_PROMPT = (
    "You are a careful cybersecurity assistant used for education, defensive "
    "security, and authorized testing (CTF/lab environments). Reason from the "
    "evidence provided. If the evidence is insufficient to answer, say so "
    "explicitly rather than guessing. Do not fabricate CVEs, tool output, or facts."
)


class SupportsGenerate(Protocol):
    """Minimal interface the harness needs from a model client."""

    model: str

    def generate(self, prompt: str, system: str | None = ...,
                 temperature: float = ..., seed: int = ...,
                 num_predict: int | None = ...): ...


def _aggregate(results: list[ScoreResult], items: list[BenchmarkItem]) -> dict:
    by_cat: dict[str, dict[str, float]] = {}
    cat_of = {it.id: it.category for it in items}
    for r in results:
        cat = cat_of[r.item_id]
        agg = by_cat.setdefault(cat, {"score_sum": 0.0, "passed": 0, "count": 0})
        agg["score_sum"] += r.score
        agg["passed"] += int(r.passed)
        agg["count"] += 1
    categories = {
        cat: {
            "count": a["count"],
            "pass_rate": round(a["passed"] / a["count"], 3),
            "mean_score": round(a["score_sum"] / a["count"], 3),
        }
        for cat, a in sorted(by_cat.items())
    }
    n = len(results)
    return {
        "overall": {
            "count": n,
            "pass_rate": round(sum(r.passed for r in results) / n, 3),
            "mean_score": round(sum(r.score for r in results) / n, 3),
        },
        "by_category": categories,
    }


def run_benchmark(
    client: SupportsGenerate,
    benchmark_path: str | Path,
    out_dir: str | Path,
    system_prompt: str = BASELINE_SYSTEM_PROMPT,
    temperature: float = 0.0,
    seed: int = 0,
    num_predict: int = 512,
    experiment_name: str | None = None,
    split: str | None = None,
    judge: "JudgeScorer | None" = None,
) -> dict:
    """Run every benchmark item through `client`, score, and persist results.

    If `split` is given ("dev" or "test"), only items with that split are run.
    This lets the held-out `test` set be evaluated separately from `dev`
    (see data/evaluation/README.md).

    If `judge` (a `JudgeScorer`) is given, each item is ALSO graded by the LLM judge and
    the judge verdict is attached to that item's record (`judge_passed`, `judge_score`,
    `judge_detail`, `judge_error`). The deterministic score remains the primary,
    reproducible number; the judge is a supplement and never overwrites it.
    """
    items = load_benchmark(benchmark_path)
    if split is not None:
        items = [it for it in items if it.split == split]
        if not items:
            raise ValueError(f"no items with split={split!r} in {benchmark_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_item = []
    results: list[ScoreResult] = []
    judge_pass = judge_total = judge_errors = 0
    started = time.time()
    for item in items:
        prompt = item.render_prompt()
        gen = client.generate(
            prompt, system=system_prompt, temperature=temperature,
            seed=seed, num_predict=num_predict,
        )
        result = score_item(item, gen.text)
        results.append(result)
        record = {
            "id": item.id,
            "category": item.category,
            "domain": item.domain,
            "difficulty": item.difficulty,
            "scorer": item.scorer,
            "score": result.score,
            "passed": result.passed,
            "detail": result.detail,
            "response": gen.text,
        }
        if judge is not None:
            verdict = judge.evaluate(item, gen.text)
            record.update({
                "judge_passed": verdict.passed,
                "judge_score": verdict.score,
                "judge_detail": verdict.reason,
                "judge_error": verdict.error,
            })
            judge_total += 1
            judge_pass += int(verdict.passed)
            judge_errors += int(verdict.error is not None)
        per_item.append(record)
    duration = round(time.time() - started, 1)

    aggregates = _aggregate(results, items)
    report = {
        "experiment": experiment_name or out_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": getattr(client, "model", "unknown"),
        "benchmark_path": str(benchmark_path),
        "benchmark_size": len(items),
        "split": split,
        "settings": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "system_prompt": system_prompt,
        },
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "duration_seconds": duration,
        **aggregates,
        "items": per_item,
    }
    if judge is not None:
        report["judge"] = {
            "model": judge.model,
            "prompt_version": judge.prompt_version,
            "temperature": judge.temperature,
            "seed": judge.seed,
            "count": judge_total,
            "pass_rate": round(judge_pass / judge_total, 3) if judge_total else None,
            "errors": judge_errors,
        }

    (out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "scorecard.md").write_text(_render_scorecard(report), encoding="utf-8")
    return report


def rescore_results(results_path: str | Path, benchmark_path: str | Path) -> dict:
    """Recompute scores from an existing results.json WITHOUT re-running the model.

    Model outputs are deterministic and already stored, so when scorers improve
    we can re-score cheaply instead of paying for inference again. Writes updated
    results.json + scorecard.md next to the input file.
    """
    results_path = Path(results_path)
    prior = json.loads(results_path.read_text(encoding="utf-8"))
    items = {it.id: it for it in load_benchmark(benchmark_path)}

    results: list[ScoreResult] = []
    item_list: list[BenchmarkItem] = []
    new_items = []
    for rec in prior["items"]:
        item = items[rec["id"]]
        item_list.append(item)
        r = score_item(item, rec["response"])
        results.append(r)
        rec = {**rec, "score": r.score, "passed": r.passed, "detail": r.detail}
        new_items.append(rec)

    report = {**prior, **_aggregate(results, item_list), "items": new_items,
              "rescored": True}
    results_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (results_path.parent / "scorecard.md").write_text(
        _render_scorecard(report), encoding="utf-8"
    )
    return report


def _render_scorecard(report: dict) -> str:
    o = report["overall"]
    lines = [
        f"# Scorecard — {report['experiment']}",
        "",
        f"- **Model:** `{report['model']}`",
        f"- **Timestamp (UTC):** {report['timestamp_utc']}",
        f"- **Benchmark:** `{report['benchmark_path']}` ({report['benchmark_size']} items)",
        f"- **Settings:** temp={report['settings']['temperature']}, "
        f"seed={report['settings']['seed']}, num_predict={report['settings']['num_predict']}",
        f"- **Duration:** {report['duration_seconds']}s",
        "",
        f"## Overall: pass_rate = **{o['pass_rate']}**, mean_score = **{o['mean_score']}** "
        f"(n={o['count']})",
        "",
        "## By category",
        "",
        "| Category | n | Pass rate | Mean score |",
        "|---|---:|---:|---:|",
    ]
    for cat, a in report["by_category"].items():
        lines.append(f"| {cat} | {a['count']} | {a['pass_rate']} | {a['mean_score']} |")
    lines += ["", "## Per-item", "", "| ID | Cat | Scorer | Pass | Score | Detail |",
              "|---|---|---|:--:|---:|---|"]
    for it in report["items"]:
        chk = "✅" if it["passed"] else "❌"
        detail = it["detail"].replace("|", "\\|")[:80]
        lines.append(
            f"| {it['id']} | {it['category']} | {it['scorer']} | {chk} | {it['score']} | {detail} |"
        )
    return "\n".join(lines) + "\n"
