"""Merge the chosen GRPO adapter into a BF16 copy of the base model.

Training loaded the base in 4-bit. A serving checkpoint has to be a full BF16
model, so this script loads the base on CPU, merges models/grpo_adapter, and
writes models/final_merged/. It records best_adapter only after the weights
exist.

The launch reason is the FINAL-002 choice: higher Evidence F1 and ROUGE-L than
SFT, lower Unsupported than DPO, with the SFT Unsupported and Answer F1 cost
accepted. Training reward is not the reason.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LAUNCH_REASON = (
    "Evidence F1 0.610712 and ROUGE-L 0.341764 are above SFT "
    "(0.59959 and 0.339676). Unsupported 0.128877 is below DPO 0.131633. "
    "Accepted cost versus SFT: Unsupported 0.116471 to 0.128877, "
    "Answer F1 0.596897 to 0.589487. Training reward 0.875 is not the reason."
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def directory_gb(path: Path) -> float:
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return round(total / (1024 ** 3), 3)


def record_choice(selection_path: Path, adapter: str) -> None:
    current = {}
    if selection_path.exists():
        current = json.loads(selection_path.read_text(encoding="utf-8"))
    current["best_model"] = "GRPO"
    current["best_adapter"] = adapter
    current["launch_reason"] = LAUNCH_REASON
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def merge(root: Path, frozen_path: Path, adapter: Path, output: Path, overwrite: bool) -> dict:
    if not frozen_path.exists():
        raise SystemExit(f"missing frozen config: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    model_path = Path(frozen["model_path"])
    if not model_path.exists():
        raise SystemExit(f"missing base model: {model_path}")
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"missing adapter: {adapter}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise SystemExit(f"{output} already exists; pass --overwrite to replace it")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    print(f"merge {adapter} into {model_path}", flush=True)
    print("load base in bf16 on cpu", flush=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    elapsed = time.perf_counter() - started

    weights = list(output.glob("*.safetensors")) + list(output.glob("*.bin"))
    if not (output / "config.json").exists() or not weights:
        raise SystemExit(f"merged model was not saved to {output}")
    if (output / "adapter_config.json").exists():
        raise SystemExit(f"{output} still contains an adapter config")

    adapter_rel = str(adapter.relative_to(root)).replace("\\", "/")
    record_choice(root / "outputs" / "final_eval" / "selection.json", adapter_rel)
    summary = {
        "base_model": str(model_path),
        "adapter": adapter_rel,
        "output": str(output),
        "dtype": "bfloat16",
        "device": "cpu",
        "weight_files": [path.name for path in weights],
        "output_gb": directory_gb(output),
        "merge_time_sec": round(elapsed, 1),
        "best_adapter": adapter_rel,
    }
    (output / "merge_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Merge the GRPO adapter into the BF16 base model.")
    parser.add_argument("--frozen", type=Path, default=root / "data" / "stats" / "frozen_config.json")
    parser.add_argument("--adapter", type=Path, default=root / "models" / "grpo_adapter")
    parser.add_argument("--output", type=Path, default=root / "models" / "final_merged")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    merge(root, args.frozen, args.adapter, args.output, args.overwrite)


if __name__ == "__main__":
    main()
