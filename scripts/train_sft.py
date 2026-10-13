"""QLoRA-SFT on the frozen SciTLDR and QASPER training split.

Writes outputs/sft/config.json before training. The adapter is saved to
models/sft_adapter/. Rank, learning rate, and cutoff stay at the frozen
values. Pass --epochs 1 only when two epochs will not finish, and record
that choice.

After this process exits, score the adapter with:

    python scripts/evaluate.py --adapter models/sft_adapter --output outputs/sft --load-in-4bit --overwrite
"""

from __future__ import annotations

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


def training_config(frozen: dict, epochs: int) -> dict:
    return {
        "model_path": frozen["model_path"],
        "cutoff_len": int(frozen["cutoff_len"]),
        "max_new_tokens": int(frozen["max_new_tokens"]),
        "benchmark_split": frozen["benchmark_split"],
        "train_file": "data/processed/sft_train.jsonl",
        "eval_file": "data/processed/sft_val.jsonl",
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
        "learning_rate": 2e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "num_train_epochs": epochs,
        "completion_only_loss": True,
        "optim": "paged_adamw_8bit",
        "seed": 42,
    }


def to_trl_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        if "prompt_messages" not in row or "completion" not in row:
            raise SystemExit(f"{row.get('id')} is missing prompt_messages or completion")
        converted.append(
            {
                "prompt": row["prompt_messages"],
                "completion": [{"role": "assistant", "content": row["completion"]}],
            }
        )
    return converted


def last_logged(log_history: list[dict], key: str):
    value = None
    for event in log_history:
        if key in event:
            value = event[key]
    return value


def train(root: Path, frozen_path: Path, epochs: int) -> dict:
    frozen = load_frozen(frozen_path)
    config = training_config(frozen, epochs)
    output_dir = root / "outputs" / "sft"
    adapter_dir = root / "models" / "sft_adapter"
    log_dir = root / "logs" / "sft"
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
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("trl, peft, datasets, and bitsandbytes are required: pip install trl peft bitsandbytes datasets") from exc

    if "completion_only_loss" not in inspect.signature(SFTConfig).parameters:
        raise SystemExit("trl is too old for completion_only_loss; install a current trl")

    model_path = Path(config["model_path"])
    tokenizer = load_tokenizer(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quant,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.print_trainable_parameters()

    train_rows = read_jsonl(root / config["train_file"])
    eval_rows = read_jsonl(root / config["eval_file"])
    if not train_rows or not eval_rows:
        raise SystemExit("sft_train.jsonl or sft_val.jsonl is empty")
    train_dataset = Dataset.from_list(to_trl_rows(train_rows))
    eval_dataset = Dataset.from_list(to_trl_rows(eval_rows))
    trained_tokens = sum(int(row["metadata"]["packed_tokens"]) for row in train_rows) * epochs

    sft_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit",
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "max_length": int(config["cutoff_len"]),
        "packing": False,
        "completion_only_loss": True,
        "report_to": "none",
        "seed": 42,
        "remove_unused_columns": False,
    }
    accepted = inspect.signature(SFTConfig).parameters
    accepts_extra = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in accepted.values())
    if "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        sft_kwargs["evaluation_strategy"] = sft_kwargs.pop("eval_strategy")
    if "warmup_ratio" in accepted or accepts_extra:
        sft_kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in accepted:
        steps_per_epoch = math.ceil(len(train_rows) / config["gradient_accumulation_steps"])
        sft_kwargs["warmup_steps"] = max(1, round(0.03 * steps_per_epoch * epochs))
        config["warmup_steps"] = sft_kwargs["warmup_steps"]
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"warmup_steps {sft_kwargs['warmup_steps']}", flush=True)
    unknown = [key for key in sft_kwargs if key not in accepted and not accepts_extra]
    if unknown:
        raise SystemExit(f"installed trl SFTConfig does not accept: {', '.join(unknown)}")
    arguments = SFTConfig(**sft_kwargs)
    if "warmup_ratio" not in sft_kwargs and "warmup_steps" not in sft_kwargs and hasattr(arguments, "warmup_ratio"):
        arguments.warmup_ratio = 0.03
    trainer_kwargs = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    runtime = float(result.metrics.get("train_runtime", elapsed))
    summary = {
        "training": {
            "train_loss": result.metrics.get("train_loss"),
            "eval_loss": last_logged(trainer.state.log_history, "eval_loss"),
            "learning_rate": last_logged(trainer.state.log_history, "learning_rate"),
            "epoch": result.metrics.get("epoch", epochs),
            "global_step": trainer.state.global_step,
            "gpu_peak_memory_gb": peak_gb,
            "training_time_sec": round(runtime, 1),
            "trained_tokens": trained_tokens,
            "tokens_per_second": round(trained_tokens / runtime, 2) if runtime else None,
            "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        },
        "config": config,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run frozen QLoRA-SFT.")
    parser.add_argument("--frozen", type=Path, default=root / "data" / "stats" / "frozen_config.json")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    train(root, args.frozen, args.epochs)


if __name__ == "__main__":
    main()
