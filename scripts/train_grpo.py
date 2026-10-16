"""GRPO on QASPER, continued from the DPO adapter.

The reward is the frozen function in scripts/rewards.py. Run that file first:

    python scripts/rewards.py

SciTLDR rows are excluded. Learning rate, generations, completion length, rank,
and the reward weights stay at the frozen values. KL beta is left at the
installed GRPOConfig default because the frozen plan does not set it.

After this process exits, score the adapter with:

    python scripts/evaluate.py --adapter models/grpo_adapter --output outputs/grpo --load-in-4bit --overwrite
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

from analyze_qasper import read_jsonl
from analyze_scitldr import load_tokenizer
from evaluate import Tee
from rewards import qasper_reward
from train_dpo import disable_output_fp32_cast, filter_kwargs, load_frozen, load_policy, unwrap_output_fp32_cast

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def last_logged(log_history: list[dict], key: str):
    value = None
    for event in log_history:
        if key in event:
            value = event[key]
    return value


def qasper_rows(path: Path) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row.get("task") == "paper_qa"]
    if not rows:
        raise SystemExit(f"no QASPER rows in {path}")
    return rows


def to_grpo_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        if "prompt_messages" not in row or "completion" not in row:
            raise SystemExit(f"{row.get('id')} is missing prompt_messages or completion")
        references = list(row["metadata"].get("references") or [])
        converted.append(
            {
                "prompt": row["prompt_messages"],
                "gold_completion": row["completion"],
                "references_json": json.dumps(references, ensure_ascii=False),
                "paper": row.get("prompt") or "",
            }
        )
    return converted


def training_config(frozen: dict) -> dict:
    return {
        "model_path": frozen["model_path"],
        "dpo_adapter": "models/dpo_adapter",
        "policy_adapter": "default",
        "cutoff_len": int(frozen["cutoff_len"]),
        "max_completion_length": 512,
        "benchmark_split": frozen["benchmark_split"],
        "train_file": "data/processed/sft_train.jsonl",
        "task": "paper_qa",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": TARGET_MODULES,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "bf16": True,
        "learning_rate": 1e-6,
        "num_train_epochs": 1,
        "num_generations": 4,
        "steps_per_generation": 4,
        "reward": "1.00*AnswerF1 + 0.60*EvidenceF1 + 0.20*Format - 0.50*Unsupported - 0.10*Length",
        "optim": "paged_adamw_8bit",
        "seed": 42,
        "save_strategy": "steps",
        "save_steps": 100,
    }


def latest_checkpoint(output_dir: Path) -> Path | None:
    found = []
    for path in output_dir.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if path.is_dir() and step.isdigit():
            found.append((int(step), path))
    if not found:
        return None
    return sorted(found)[-1][1]


def train(root: Path, frozen_path: Path, resume: bool) -> dict:
    frozen = load_frozen(frozen_path)
    config = training_config(frozen)
    output_dir = root / "outputs" / "grpo"
    adapter_dir = root / "models" / "grpo_adapter"
    log_dir = root / "logs" / "grpo"
    dpo_dir = root / config["dpo_adapter"]
    if not dpo_dir.exists():
        raise SystemExit(f"missing DPO adapter: {dpo_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    log_handle = (log_dir / "run.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("trl, peft, datasets, and bitsandbytes are required") from exc

    disable_output_fp32_cast()
    print("kept GRPO logits in bf16", flush=True)
    model_path = Path(config["model_path"])
    tokenizer = load_tokenizer(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(model_path, dpo_dir, "default", None)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"trainable params: {trainable}", flush=True)
    if trainable > 50_000_000:
        raise SystemExit("more than one adapter is trainable")
    model.print_trainable_parameters()

    train_rows = qasper_rows(root / config["train_file"])
    print(f"qasper train rows {len(train_rows)}", flush=True)
    train_dataset = Dataset.from_list(to_grpo_rows(train_rows))

    grpo_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-6,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit",
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "num_generations": 4,
        "max_completion_length": 512,
        "steps_per_generation": 4,
        "report_to": "none",
        "seed": 42,
        "remove_unused_columns": False,
    }
    accepted = inspect.signature(GRPOConfig).parameters
    accepts_extra = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in accepted.values())
    if "use_vllm" in accepted or accepts_extra:
        grpo_kwargs["use_vllm"] = False
    kept, unknown = filter_kwargs(GRPOConfig, grpo_kwargs)
    if unknown:
        raise SystemExit(f"installed trl GRPOConfig does not accept: {', '.join(unknown)}")
    arguments = GRPOConfig(**kept)
    beta = getattr(arguments, "beta", None)
    config["beta"] = beta
    config["generation_batch_size"] = getattr(arguments, "generation_batch_size", None)
    config["steps_per_generation"] = getattr(arguments, "steps_per_generation", config["steps_per_generation"])
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"grpo beta {beta}", flush=True)
    print(
        f"steps_per_generation {config['steps_per_generation']} generation_batch_size {config['generation_batch_size']}",
        flush=True,
    )

    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "reward_funcs": qasper_reward,
        "args": arguments,
        "train_dataset": train_dataset,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    unwrap_output_fp32_cast(trainer.model)
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad = False
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"trainable params after trainer init: {trainable}", flush=True)
    if trainable > 50_000_000:
        raise SystemExit("reference adapter is trainable")
    if beta not in (None, 0, 0.0) and "ref" not in getattr(model, "peft_config", {}):
        raise SystemExit("GRPOTrainer did not copy the DPO adapter into a frozen reference adapter")

    checkpoint = latest_checkpoint(output_dir)
    if resume and checkpoint is None:
        raise SystemExit(
            "no checkpoint-* directory under outputs/grpo. "
            "The interrupted run saved only at the end of the epoch, so those steps cannot be resumed."
        )
    if resume:
        print(f"resume from {checkpoint}", flush=True)
    elif checkpoint is not None:
        print(f"checkpoint exists at {checkpoint}; pass --resume to continue it", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if resume else None)
    elapsed = time.perf_counter() - started
    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

    save_target = trainer.model
    unwrap = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    if unwrap is not None:
        save_target = unwrap(trainer.model)
    save_target.set_adapter("default")
    if "ref" in getattr(save_target, "peft_config", {}):
        save_target.delete_adapter("ref")
    save_params = inspect.signature(save_target.save_pretrained).parameters
    if "selected_adapters" in save_params:
        save_target.save_pretrained(str(adapter_dir), selected_adapters=["default"])
    else:
        save_target.save_pretrained(str(adapter_dir))
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"GRPO adapter was not saved to {adapter_dir}")
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    runtime = float(result.metrics.get("train_runtime", elapsed))
    training = {
        "train_loss": result.metrics.get("train_loss"),
        "epoch": result.metrics.get("epoch", 1),
        "global_step": trainer.state.global_step,
        "gpu_peak_memory_gb": peak_gb,
        "training_time_sec": round(runtime, 1),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "trainable_params": trainable,
        "qasper_train_rows": len(train_rows),
        "beta": beta,
    }
    for key in ("reward", "reward_std", "kl", "entropy", "completions/mean_length"):
        training[key.replace("/", "_")] = last_logged(trainer.state.log_history, key)
    summary = {"training": training, "config": config}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run frozen QASPER GRPO from the DPO adapter.")
    parser.add_argument("--frozen", type=Path, default=root / "data" / "stats" / "frozen_config.json")
    parser.add_argument("--resume", action="store_true", help="Continue from the latest outputs/grpo/checkpoint-*")
    args = parser.parse_args()
    train(root, args.frozen, args.resume)


if __name__ == "__main__":
    main()
