#!/usr/bin/env python3
"""Calibrate the LLM judge against a human/reference-labeled calibration set.

Measures judge-vs-gold agreement on the items in `data/evaluation/judge_calibration.jsonl`
(target: >= 80%, per the roadmap). The "gold" label is each row's `human_verdict` when
every row has one; otherwise it falls back to `reference_verdict` (engineer-adjudicated,
pending independent human verification) and says so explicitly in the output.

The judge model MUST differ from the model whose responses are being judged (default judge
gemma4:26b vs. subject gemma3:4b) to avoid self-grading bias. Requires a reachable Ollama
with the judge model pulled; if unavailable it reports that and exits 3 (does not fake a
result).

Usage:
    python scripts/judge_calibration.py
    python scripts/judge_calibration.py --judge-model gemma4:12b --out experiments/judge_calibration
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.clients import OllamaClient  # noqa: E402
from gemma_cyber.evaluation.judge import DEFAULT_JUDGE_MODEL, JudgeScorer  # noqa: E402
from gemma_cyber.evaluation.schema import load_benchmark  # noqa: E402

CALIB = "data/evaluation/judge_calibration.jsonl"
BENCHMARK = "data/evaluation/benchmark_v2.jsonl"


def _load_calibration(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the LLM judge vs. reference labels.")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--calibration", default=CALIB)
    ap.add_argument("--benchmark", default=BENCHMARK)
    ap.add_argument("--out", default="experiments/judge_calibration")
    ap.add_argument("--min-agreement", type=float, default=0.80)
    args = ap.parse_args()

    rows = _load_calibration(args.calibration)
    items = {it.id: it for it in load_benchmark(args.benchmark)}

    # Gold basis: prefer human labels if fully present, else engineer reference labels.
    have_all_human = all(r.get("human_verdict") in ("PASS", "FAIL") for r in rows)
    gold_field = "human_verdict" if have_all_human else "reference_verdict"
    gold_basis = "human_verdict" if have_all_human else "reference_verdict (engineer-adjudicated, pending human verification)"

    client = OllamaClient(model=args.judge_model, host=args.host)
    if not client.is_available():
        print(f"ERROR: Ollama not reachable at {args.host}.", file=sys.stderr)
        return 3
    if not client.has_model(args.judge_model):
        print(f"ERROR: judge model '{args.judge_model}' not pulled.", file=sys.stderr)
        return 3

    judge = JudgeScorer(client=client)

    agree = disagree = judge_errors = 0
    agree_vs_det = 0
    per_item = []
    for r in rows:
        item = items[r["id"]]
        verdict = judge.evaluate(item, r["response"])
        judge_label = "PASS" if verdict.passed else "FAIL"
        gold = r[gold_field]
        is_agree = judge_label == gold
        agree += int(is_agree)
        disagree += int(not is_agree)
        judge_errors += int(verdict.error is not None)
        det_label = "PASS" if r["deterministic_passed"] else "FAIL"
        agree_vs_det += int(judge_label == det_label)
        per_item.append({
            "id": r["id"], "category": r["category"], "scorer": r["scorer"],
            "gold": gold, "judge": judge_label, "agree": is_agree,
            "deterministic": det_label, "judge_score": verdict.score,
            "judge_reason": verdict.reason, "judge_error": verdict.error,
        })

    n = len(rows)
    agreement = round(agree / n, 3) if n else 0.0
    report = {
        "judge_model": judge.model,
        "judge_prompt_version": judge.prompt_version,
        "judge_settings": {"temperature": judge.temperature, "seed": judge.seed,
                           "num_predict": judge.num_predict},
        "calibration_file": args.calibration,
        "n_items": n,
        "gold_basis": gold_basis,
        "agreements": agree,
        "disagreements": disagree,
        "agreement": agreement,
        "judge_errors": judge_errors,
        "judge_vs_deterministic_agreement": round(agree_vs_det / n, 3) if n else 0.0,
        "target": args.min_agreement,
        "target_met": agreement >= args.min_agreement,
        "items": per_item,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "calibration.md").write_text(_render(report), encoding="utf-8")

    print(f"\nJudge: {judge.model}  | gold basis: {gold_basis}")
    print(f"Agreement: {agree}/{n} = {agreement:.1%}  (target >= {args.min_agreement:.0%})")
    print(f"Judge errors: {judge_errors}  | judge-vs-deterministic: {report['judge_vs_deterministic_agreement']:.1%}")
    print("Disagreements:")
    for it in per_item:
        if not it["agree"]:
            print(f"  {it['id']:14} gold={it['gold']} judge={it['judge']} "
                  f"({it['scorer']}) — {it['judge_reason'][:70]}")
    print(f"\n{'MET' if report['target_met'] else 'NOT MET'}: agreement "
          f"{agreement:.1%} vs target {args.min_agreement:.0%} -> {out_dir}/")
    return 0


def _render(rep: dict) -> str:
    lines = [
        f"# Judge calibration — {rep['judge_model']}",
        "",
        f"- **Judge model:** `{rep['judge_model']}` (prompt `{rep['judge_prompt_version']}`, "
        f"temp={rep['judge_settings']['temperature']}, seed={rep['judge_settings']['seed']})",
        f"- **Gold basis:** {rep['gold_basis']}",
        f"- **Items:** {rep['n_items']}",
        f"- **Agreement:** {rep['agreements']}/{rep['n_items']} = **{rep['agreement']:.1%}** "
        f"(target ≥ {rep['target']:.0%}) — {'MET' if rep['target_met'] else 'NOT MET'}",
        f"- **Judge errors:** {rep['judge_errors']}",
        f"- **Judge vs deterministic agreement:** {rep['judge_vs_deterministic_agreement']:.1%}",
        "",
        "| ID | Scorer | Gold | Judge | Agree | Det | Judge reason |",
        "|---|---|:--:|:--:|:--:|:--:|---|",
    ]
    for it in rep["items"]:
        chk = "✅" if it["agree"] else "❌"
        reason = (it["judge_reason"] or "").replace("|", "\\|")[:60]
        err = f" [ERROR: {it['judge_error']}]" if it["judge_error"] else ""
        lines.append(f"| {it['id']} | {it['scorer']} | {it['gold']} | {it['judge']} | {chk} "
                     f"| {it['deterministic']} | {reason}{err} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
