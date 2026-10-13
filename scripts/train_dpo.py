"""DPO from the saved SFT adapter.

The policy adapter starts as a copy of models/sft_adapter. The reference is
that same SFT adapter with training disabled, so DPO does not use the base
model as the reference and does not load a second 7B model.

Writes outputs/dpo/config.json before training. The trained adapter is saved
to models/dpo_adapter/ and can be loaded on the base model by itself.

After this process exits, score the adapter with:

    python scripts/evaluate.py --adapter models/dpo_adapter --output outputs/dpo --load-in-4bit --overwrite
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path

from analyze_qasper import read_jsonl
from analyze_scitldr import load_tokenizer
from evaluate import Tee

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LOG_KEYS = (
    "eval_loss",
    "learning_rate",
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
    "eval_rewards/accuracies",
    "eval_rewards/margins",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_frozen(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing frozen config: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cutoff_len", "model_path", "benchmark_split", "max_new_tokens"):
        if key not in frozen:
            raise SystemExit(f"{path} has no {key}")
    return frozen


def require_audit(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing preference stats: {path}")
    stats = json.loads(path.read_text(encoding="utf-8"))
    audit = stats.get("audit") or {}
    if not audit.get("passed"):
        raise SystemExit("preference audit has not passed; do not start DPO")
    return stats


def training_config(frozen: dict, epochs: int) -> dict:
    cutoff = int(frozen["cutoff_len"])
    max_new_tokens = int(frozen["max_new_tokens"])
    return {
        "model_path": frozen["model_path"],
        "sft_adapter": "models/sft_adapter",
        "policy_adapter": "dpo",
        "reference_adapter": "sft",
        "cutoff_len": cutoff,
        "max_new_tokens": max_new_tokens,
        "max_length": cutoff,
        "max_prompt_length": cutoff - max_new_tokens,
        "max_completion_length": max_new_tokens,
        "benchmark_split": frozen["benchmark_split"],
        "train_file": "data/preference/preference_train.jsonl",
        "eval_file": "data/preference/preference_val.jsonl",
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
        "learning_rate": 5e-6,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "num_train_epochs": epochs,
        "beta": 0.1,
        "optim": "paged_adamw_8bit",
        "seed": 42,
    }


def to_dpo_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        for key in ("prompt", "chosen", "rejected"):
            if key not in row:
                raise SystemExit(f"{row.get('id')} is missing {key}")
        converted.append({"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]})
    return converted


def last_logged(log_history: list[dict], key: str):
    value = None
    for event in log_history:
        if key in event:
            value = event[key]
    return value


def filter_kwargs(config_cls, values: dict) -> tuple[dict, list[str]]:
    accepted = inspect.signature(config_cls).parameters
    accepts_extra = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in accepted.values())
    kept = {}
    unknown = []
    for key, value in values.items():
        if key in accepted or accepts_extra:
            kept[key] = value
        else:
            unknown.append(key)
    return kept, unknown


def disable_output_fp32_cast() -> None:
    import sys

    import accelerate.utils.operations as operations

    original = operations.convert_outputs_to_fp32

    def keep_dtype(model_forward):
        return model_forward

    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "convert_outputs_to_fp32", None) is original:
            module.convert_outputs_to_fp32 = keep_dtype
    operations.convert_outputs_to_fp32 = keep_dtype


def unwrap_output_fp32_cast(module) -> int:
    removed = 0
    seen = set()
    pending = [module]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        forward = getattr(current, "forward", None)
        wrapped = getattr(forward, "__wrapped__", None)
        original = getattr(wrapped, "model_forward", None)
        if callable(original):
            current.forward = original
            removed += 1
        for name in ("base_model", "model", "module"):
            child = getattr(current, name, None)
            if child is not None and child is not current:
                pending.append(child)
    return removed


def load_policy(model_path: Path, sft_dir: Path, policy_name: str, reference_name: str | None):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quant,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if reference_name is None:
        model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)
        model.set_adapter(policy_name)
        removed = unwrap_output_fp32_cast(model)
        print(f"removed fp32 output casts {removed}", flush=True)
        return model
    model = PeftModel.from_pretrained(model, str(sft_dir), adapter_name=reference_name, is_trainable=False)
    model.load_adapter(str(sft_dir), adapter_name=policy_name, is_trainable=True)
    model.set_adapter(policy_name)
    removed = unwrap_output_fp32_cast(model)
    print(f"removed fp32 output casts {removed}", flush=True)
    return model


def train(root: Path, frozen_path: Path, epochs: int) -> dict:
    frozen = load_frozen(frozen_path)
    require_audit(root / "data" / "stats" / "preference_stats.json")
    config = training_config(frozen, epochs)
    output_dir = root / "outputs" / "dpo"
    adapter_dir = root / "models" / "dpo_adapter"
    log_dir = root / "logs" / "dpo"
    sft_dir = root / config["sft_adapter"]
    if not sft_dir.exists():
        raise SystemExit(f"missing SFT adapter: {sft_dir}")
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
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit("trl, peft, datasets, and bitsandbytes are required") from exc

    disable_output_fp32_cast()
    print("kept DPO logits in bf16", flush=True)

    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    named_ref = "model_adapter_name" in trainer_params and "ref_adapter_name" in trainer_params
    if named_ref:
        config["policy_adapter"] = "dpo"
        config["reference_adapter"] = "sft"
    else:
        config["policy_adapter"] = "default"
        config["reference_adapter"] = "ref"
        print("DPOTrainer copies the default adapter into a frozen ref adapter", flush=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    model_path = Path(config["model_path"])
    tokenizer = load_tokenizer(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = load_policy(
        model_path,
        sft_dir,
        config["policy_adapter"],
        config["reference_adapter"] if named_ref else None,
    )
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"trainable params: {trainable} / {total} ({trainable / total:.4%})", flush=True)
    if trainable > 50_000_000:
        raise SystemExit("reference adapter is trainable; keep the sft adapter frozen")
    model.print_trainable_parameters()

    train_rows = read_jsonl(root / config["train_file"])
    eval_rows = read_jsonl(root / config["eval_file"])
    if not train_rows or not eval_rows:
        raise SystemExit("preference train or validation file is empty")
    train_dataset = Dataset.from_list(to_dpo_rows(train_rows))
    eval_dataset = Dataset.from_list(to_dpo_rows(eval_rows))

    dpo_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-6,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit",
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "beta": 0.1,
        "max_length": config["max_length"],
        "max_prompt_length": config["max_prompt_length"],
        "max_completion_length": config["max_completion_length"],
        "report_to": "none",
        "seed": 42,
        "remove_unused_columns": False,
    }
    if "eval_strategy" not in inspect.signature(DPOConfig).parameters and "evaluation_strategy" in inspect.signature(DPOConfig).parameters:
        dpo_kwargs["evaluation_strategy"] = dpo_kwargs.pop("eval_strategy")
    accepted = inspect.signature(DPOConfig).parameters
    accepts_extra = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in accepted.values())
    if "warmup_ratio" in accepted or accepts_extra:
        dpo_kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in accepted:
        steps_per_epoch = math.ceil(len(train_rows) / config["gradient_accumulation_steps"])
        dpo_kwargs["warmup_steps"] = max(1, round(0.03 * steps_per_epoch * epochs))
        config["warmup_steps"] = dpo_kwargs["warmup_steps"]
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"warmup_steps {dpo_kwargs['warmup_steps']}", flush=True)
    if "max_completion_length" not in accepted and not accepts_extra:
        if "max_target_length" in accepted:
            dpo_kwargs["max_target_length"] = dpo_kwargs.pop("max_completion_length")
        else:
            dpo_kwargs.pop("max_completion_length", None)
            print("drop max_completion_length; truncation uses max_length", flush=True)
    if "max_prompt_length" not in accepted and not accepts_extra:
        dpo_kwargs.pop("max_prompt_length", None)
        print("drop max_prompt_length; truncation uses max_length", flush=True)
    if "truncation_mode" in accepted or accepts_extra:
        dpo_kwargs["truncation_mode"] = "keep_end"
        config["truncation_mode"] = "keep_end"
        print("truncation_mode keep_end", flush=True)
    if "precompute_ref_log_probs" in accepted or accepts_extra:
        dpo_kwargs["precompute_ref_log_probs"] = True
        config["precompute_ref_log_probs"] = True
        print("precompute_ref_log_probs true", flush=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    kept, unknown = filter_kwargs(DPOConfig, dpo_kwargs)
    if "beta" not in kept:
        unknown = [key for key in unknown if key != "beta"]
    if unknown:
        raise SystemExit(f"installed trl DPOConfig does not accept: {', '.join(unknown)}")
    arguments = DPOConfig(**kept)

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if "beta" not in kept:
        trainer_kwargs["beta"] = 0.1
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if named_ref:
        trainer_kwargs["model_adapter_name"] = config["policy_adapter"]
        trainer_kwargs["ref_adapter_name"] = config["reference_adapter"]
    trainer_unknown = [key for key in trainer_kwargs if key not in trainer_params and trainer_params.get("kwargs") is None]
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in trainer_params.values()):
        trainer_unknown = []
    if trainer_unknown:
        raise SystemExit(f"installed trl DPOTrainer does not accept: {', '.join(trainer_unknown)}")
    trainer = DPOTrainer(**trainer_kwargs)
    removed = unwrap_output_fp32_cast(trainer.model)
    print(f"removed fp32 output casts after trainer init {removed}", flush=True)
    if not named_ref and "ref" not in getattr(model, "peft_config", {}):
        raise SystemExit("DPOTrainer did not copy the SFT adapter into a frozen reference adapter")
    for name, param in model.named_parameters():
        if ".ref." in name or ".sft." in name:
            param.requires_grad = False
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"trainable params after trainer init: {trainable}", flush=True)
    if trainable > 50_000_000:
        raise SystemExit("reference adapter is trainable; keep the sft adapter frozen")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

    save_target = trainer.model
    unwrap = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    if unwrap is not None:
        save_target = unwrap(trainer.model)
    policy_name = config["policy_adapter"]
    save_target.set_adapter(policy_name)
    for extra in ("ref", "sft"):
        if extra != policy_name and extra in getattr(save_target, "peft_config", {}):
            save_target.delete_adapter(extra)
    save_params = inspect.signature(save_target.save_pretrained).parameters
    if "selected_adapters" in save_params:
        save_target.save_pretrained(str(adapter_dir), selected_adapters=[policy_name])
    else:
        save_target.save_pretrained(str(adapter_dir))
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"DPO adapter was not saved to {adapter_dir}")
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    runtime = float(result.metrics.get("train_runtime", elapsed))
    training = {
        "train_loss": result.metrics.get("train_loss"),
        "epoch": result.metrics.get("epoch", epochs),
        "global_step": trainer.state.global_step,
        "gpu_peak_memory_gb": peak_gb,
        "training_time_sec": round(runtime, 1),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "trainable_params": trainable,
        "total_params": total,
    }
    for key in LOG_KEYS:
        training[key.replace("/", "_")] = last_logged(trainer.state.log_history, key)
    summary = {"training": training, "config": config}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run frozen DPO from the SFT adapter.")
    parser.add_argument("--frozen", type=Path, default=root / "data" / "stats" / "frozen_config.json")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs != 1:
        raise SystemExit("frozen DPO uses 1 epoch")
    train(root, args.frozen, args.epochs)


if __name__ == "__main__":
    main()
