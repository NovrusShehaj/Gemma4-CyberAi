# Data Licenses and Provenance

Every dataset artifact in this repo must have a known license and recorded provenance
(PROJECT_PLAN.md §16). Rows without a license are rejected by the schema validator.

## Tracked datasets

| Path | Contents | Source | License | Notes |
|---|---|---|---|---|
| `data/evaluation/benchmark_v1.jsonl` | Frozen evaluation benchmark (25 items) | **Original**, authored for this project | CC-BY-4.0 | No third-party content. Never used for training (contamination control). |

## Policy

- ❌ **No scraping** of Hack The Box / TryHackMe walkthroughs, solutions, flags, questions, or paid material. This content is copyrighted and governed by their Terms of Service; using it as data is very likely prohibited and would also contaminate benchmarks. (See PROJECT_PLAN.md §16.)
- ✅ Prefer **original authored** examples and appropriately-licensed public data (e.g., MITRE ATT&CK, Sigma/YARA repos, NVD/CVE data) — each recorded here with its license before use.
- 🔒 **Evaluation vs. training separation:** `data/evaluation/` is frozen and never enters `data/training/`. A contamination check will run before any training run (Phase 3+).
- 🧾 Anything of uncertain legal status goes to `data/quarantine/` (git-ignored) and never into `data/training/`.

## Model license

Any model fine-tuned from Gemma remains **"a Gemma"** under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its Prohibited Use Policy,
regardless of the training data. Distribution requires passing those terms downstream.
