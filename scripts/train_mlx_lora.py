#!/usr/bin/env python3
"""Local (Apple-Silicon) MLX-LoRA training for Gemma-Cyber — exp-002r and later.

This is the LOCAL counterpart to `scripts/train_qlora.py` (the CUDA/cloud path).
It exists because Unsloth/bitsandbytes do not train on Apple Silicon, but
`mlx-lm` LoRA does, and this machine (M3 Max, 128 GB) can train the 4B tier
locally in minutes — removing the Colab dependency that lost the exp-002
artifacts.

Pipeline:
  1. Render `_source_dataset` -> `<data>/{train,valid}.jsonl` deterministically
     (`gemma_cyber.training.mlx_data`), recompute `iters` from the real train
     count and `--epochs`.
  2. Run `mlx_lm.lora` with a materialised config (completion-only loss).
  3. Fuse the adapters into a standalone model dir (`mlx_lm.fuse`).
  4. Print the exact `ollama create` command to register the candidate.

Usage:
  python scripts/train_mlx_lora.py --config configs/training/mlx_lora_gemma3_4b_v0.2.yaml --dry-run
  python scripts/train_mlx_lora.py --config configs/training/mlx_lora_gemma3_4b_v0.2.yaml --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.training.mlx_data import build_mlx_dataset  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML config: {path}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=3, help="epochs over the training split (recomputes iters)")
    ap.add_argument("--dry-run", action="store_true", help="prep data + materialise config, do not train")
    ap.add_argument("--skip-fuse", action="store_true", help="train only, do not fuse the adapters")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    source = cfg.pop("_source_dataset")
    val_fraction = float(cfg.pop("_val_fraction", 0.12))
    data_dir = REPO / cfg["data"]

    print("=" * 64)
    print("MLX-LoRA training —", args.config.name)
    print("=" * 64)

    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        print("ERROR: mlx / mlx-lm not installed. `pip install mlx-lm` (Apple Silicon only).", file=sys.stderr)
        return 3

    manifest = build_mlx_dataset(REPO / source, data_dir, val_fraction=val_fraction)
    counts: dict[str, int] = manifest["counts"]
    print(f"data: {counts['train']} train / {counts['valid']} valid  ->  {data_dir}")

    steps_per_epoch = math.ceil(counts["train"] / cfg["batch_size"])
    cfg["iters"] = steps_per_epoch * args.epochs
    # keep the LR schedule horizon in sync with the real iter count
    if isinstance(cfg.get("lr_schedule"), dict) and isinstance(cfg["lr_schedule"].get("arguments"), list):
        args_list = list(cfg["lr_schedule"]["arguments"])
        if len(args_list) >= 2:
            args_list[1] = cfg["iters"]
            cfg["lr_schedule"]["arguments"] = args_list
    print(f"iters: {cfg['iters']} ({steps_per_epoch} steps/epoch x {args.epochs} epochs, batch {cfg['batch_size']})")

    materialised = data_dir.parent / f"_mlx_config_{args.config.stem}.yaml"
    materialised.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"materialised mlx config -> {materialised}")

    if args.dry_run:
        print("\nDRY RUN OK. Data prepared, config materialised, mlx-lm importable.")
        return 0

    adapter_path = REPO / cfg["adapter_path"]
    adapter_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(materialised)]
    print("\n$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != 0:
        print(f"ERROR: mlx_lm lora exited {rc}", file=sys.stderr)
        return rc

    (adapter_path / "train_manifest.json").write_text(
        json.dumps({"data_manifest": manifest, "config": cfg, "epochs": args.epochs}, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.skip_fuse:
        print("\nTrained. Adapters at", adapter_path)
        return 0

    fused_dir = adapter_path.parent / "fused_model"
    fuse_cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", cfg["model"],
        "--adapter-path", str(adapter_path),
        "--save-path", str(fused_dir),
    ]
    print("\n$", " ".join(fuse_cmd), flush=True)
    rc = subprocess.run(fuse_cmd, cwd=REPO).returncode
    if rc != 0:
        print(f"ERROR: mlx_lm fuse exited {rc}", file=sys.stderr)
        return rc

    print("\n" + "=" * 64)
    print("DONE. Fused model:", fused_dir)
    print("Register the candidate with Ollama:")
    print(f'  printf \'FROM {fused_dir}\\n\' > /tmp/Modelfile.v0.2 && \\')
    print("  ollama create gemma3-cyber:v0.2 -f /tmp/Modelfile.v0.2")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
