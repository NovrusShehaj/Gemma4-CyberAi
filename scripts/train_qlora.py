#!/usr/bin/env python3
"""Cloud QLoRA training script for Gemma-Cyber v0.1.

Designed for cloud execution (RunPod, Modal, GCP, AWS, or Google Colab) on a GPU instance
with PyTorch, CUDA, transformers, peft, trl, and bitsandbytes installed.

Usage:
    python scripts/train_qlora.py --config configs/training/qlora_gemma3_4b.yaml
    python scripts/train_qlora.py --config configs/training/qlora_gemma3_4b.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.data.schema import load_training_dataset  # noqa: E402


def check_training_environment() -> dict[str, Any]:
    """Check availability of GPU and deep learning libraries."""
    status: dict[str, Any] = {}
    try:
        import torch

        status["torch"] = True
        status["cuda_available"] = torch.cuda.is_available()
        status["device_count"] = torch.cuda.device_count() if status["cuda_available"] else 0
        if status["cuda_available"]:
            status["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        status["torch"] = False
        status["cuda_available"] = False
        status["device_count"] = 0

    try:
        import transformers  # noqa: F401

        status["transformers"] = True
    except ImportError:
        status["transformers"] = False

    try:
        import peft  # noqa: F401

        status["peft"] = True
    except ImportError:
        status["peft"] = False

    try:
        import trl  # noqa: F401

        status["trl"] = True
    except ImportError:
        status["trl"] = False

    try:
        import bitsandbytes  # noqa: F401

        status["bitsandbytes"] = True
    except ImportError:
        status["bitsandbytes"] = False

    return status


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration in {config_path}")
    return data


def run_training(config: dict[str, Any], dry_run: bool = False) -> int:
    env_status = check_training_environment()
    print("=" * 60)
    print("Gemma-Cyber QLoRA Training Pipeline")
    print("=" * 60)
    print(f"PyTorch available:      {env_status.get('torch', False)}")
    print(f"CUDA available:         {env_status.get('cuda_available', False)}")
    if env_status.get("cuda_available"):
        print(f"GPU Device:             {env_status.get('device_name')}")
    print(f"Transformers available: {env_status.get('transformers', False)}")
    print(f"PEFT available:         {env_status.get('peft', False)}")
    print(f"TRL available:          {env_status.get('trl', False)}")
    print(f"bitsandbytes available: {env_status.get('bitsandbytes', False)}")
    print("=" * 60)

    # Validate dataset
    dataset_path = config["data"]["train_dataset_path"]
    print(f"\nLoading and validating dataset: {dataset_path}")
    items = load_training_dataset(dataset_path)
    print(f"Successfully loaded {len(items)} training examples.")

    if dry_run:
        print("\nDry run completed successfully. Dataset schema and configuration verified.")
        return 0

    if not env_status.get("torch") or not env_status.get("cuda_available"):
        print(
            "\nERROR: Training requires a GPU environment with PyTorch and CUDA.\n"
            "This local repository is configured for lightweight inference and evaluation.\n"
            "To train, run this script in Google Colab, Kaggle, RunPod, or a GPU VM.\n"
            "Use notebooks/colab_qlora_training.ipynb for a one-click cloud training setup.",
            file=sys.stderr,
        )
        return 3

    missing_deps = [
        lib
        for lib in ("transformers", "peft", "trl", "bitsandbytes")
        if not env_status.get(lib, False)
    ]
    if missing_deps:
        print(
            f"\nERROR: Missing cloud training dependencies: {', '.join(missing_deps)}.\n"
            f"Install them with: pip install torch transformers peft trl bitsandbytes accelerate",
            file=sys.stderr,
        )
        return 3

    # Dynamic imports in GPU environment
    import torch
    from datasets import Dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    model_id = config["model"]["base_model_name_or_path"]
    print(f"\nInitializing 4-bit Quantization for base model: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
        target_modules=config["lora"]["target_modules"],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Convert training items to Hugging Face chat dataset
    formatted_data = [{"messages": it.to_chat_dict()} for it in items]
    hf_dataset = Dataset.from_list(formatted_data)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        learning_rate=float(config["training"]["learning_rate"]),
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        optim=config["training"]["optim"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        seed=config["training"]["seed"],
        report_to=config["training"]["report_to"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        peft_config=lora_cfg,
        max_seq_length=config["data"]["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\nStarting QLoRA fine-tuning...")
    trainer.train()

    final_adapter_dir = Path(config["training"]["output_dir"]) / "final_adapter"
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"\nTraining completed! Adapter saved to: {final_adapter_dir}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cloud QLoRA training script for Gemma-Cyber.")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/training/qlora_gemma3_4b.yaml",
        help="Path to training YAML config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset, and environment without launching training",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    return run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
