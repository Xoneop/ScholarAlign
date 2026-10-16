"""Collect the finished benchmark runs into one comparison.

Reads the metrics and predictions already written by scripts/evaluate.py.
It does not generate new answers. Reward and train loss are copied only for
the table; they do not break a tie between Evidence F1 and Unsupported.

Writes:

- outputs/final_eval/metrics.json
- outputs/final_eval/comparison.csv
- outputs/final_eval/qualitative_samples.jsonl
- outputs/final_eval/selection.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from analyze_qasper import read_jsonl
from evaluate import is_unsupported, parse_json_answer

SOURCES = (
    ("Base", "base", "outputs/base_eval", None),
    ("SFT", "sft", "outputs/sft", "models/sft_adapter"),
    ("DPO", "dpo", "outputs/dpo", "models/dpo_adapter"),
    ("GRPO", "grpo", "outputs/grpo", "models/grpo_adapter"),
    ("RLOO", "rloo", "outputs/rloo", "models/rloo_adapter"),
)
REQUIRED = ("base", "sft", "dpo", "grpo")
RULE = [
    "先看 QASPER Evidence F1 和 Unsupported Answer Rate",
    "再看 SciTLDR ROUGE-L",
    "train loss、reward 曲线不单独作为入选理由",
    "定性样例里出现明显无依据生成时，不选该 checkpoint",
]
SAMPLE_SEED = 42
TLDR_SAMPLE = 20
QA_SAMPLE = 20
DISAGREEMENT_SAMPLE = 10


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing metrics: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = payload.get("benchmark")
    if isinstance(benchmark, dict) and "scitldr" in benchmark and "qasper" in benchmark:
        scores = benchmark
    elif "scitldr" in payload and "qasper" in payload:
        scores = payload
    else:
        raise SystemExit(f"{path} has no benchmark scores")
    return {"payload": payload, "benchmark": scores, "training": payload.get("training")}


def index_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"missing predictions: {path}")
    indexed = {}
    for row in read_jsonl(path):
        if not row.get("id"):
            raise SystemExit(f"{path} has a row without id")
        indexed[row["id"]] = row
    return indexed


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return json.dumps(value)
    return str(value)


def row_unsupported(row: dict) -> bool | None:
    if row.get("task") != "paper_qa":
        return None
    parsed = parse_json_answer(row.get("model_answer") or "")
    if parsed is None:
        return True
    return is_unsupported(parsed["answer"], parsed["evidence"], row.get("input") or "")


def comparison_row(label: str, loaded: dict) -> dict:
    scores = loaded["benchmark"]
    training = loaded["training"] or {}
    scitldr = scores["scitldr"]
    qasper = scores["qasper"]
    reward = training.get("reward")
    return {
        "Model": label,
        "ROUGE-L": scitldr.get("rougeL"),
        "BERTScore": scitldr.get("bertscore"),
        "Answer F1": qasper.get("answer_f1"),
        "Evidence F1": qasper.get("evidence_f1"),
        "Unsupported Rate": qasper.get("unsupported_answer_rate"),
        "Reward": reward,
        "tokens/s": scores.get("tokens_per_second"),
        "GPU GB": scores.get("gpu_peak_memory_gb"),
    }


def pareto_front(rows: list[dict]) -> list[dict]:
    front = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            evidence_ok = other["Evidence F1"] >= candidate["Evidence F1"]
            unsupported_ok = other["Unsupported Rate"] <= candidate["Unsupported Rate"]
            strict = (
                other["Evidence F1"] > candidate["Evidence F1"]
                or other["Unsupported Rate"] < candidate["Unsupported Rate"]
            )
            if evidence_ok and unsupported_ok and strict:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def metric_number(value, label: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SystemExit(f"{label} {name} is {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"{label} {name} is {value!r}")
    if number != number:
        raise SystemExit(f"{label} {name} is NaN")
    return number


def choose(front: list[dict]) -> tuple[dict | None, str]:
    if not front:
        return None, "Pareto front is empty, so best_adapter is unset."
    if len(front) == 1:
        chosen = front[0]
        return chosen, (
            f"{chosen['Model']} is the only model on the Evidence F1 / Unsupported Pareto front."
        )
    first = front[0]
    equal = all(
        item["Evidence F1"] == first["Evidence F1"] and item["Unsupported Rate"] == first["Unsupported Rate"]
        for item in front
    )
    if equal:
        best_rouge = max(item["ROUGE-L"] for item in front)
        tied = [item for item in front if item["ROUGE-L"] == best_rouge]
        if len(tied) == 1:
            best = tied[0]
            return best, (
                "Evidence F1 and Unsupported Answer Rate are tied. "
                f"{best['Model']} has the higher SciTLDR ROUGE-L."
            )
        names = ", ".join(item["Model"] for item in tied)
        return None, f"Evidence F1, Unsupported, and ROUGE-L are tied among {names}."
    names = ", ".join(item["Model"] for item in front)
    return None, (
        f"{names} remain on the Pareto front. Evidence F1 and Unsupported Answer Rate "
        "do not pick a single model. Reward and train loss are not used to break this tie. "
        "Review qualitative_samples.jsonl before setting best_adapter."
    )


def sample_ids(grouped: dict[str, list[str]], predictions: dict[str, dict[str, dict]]) -> list[str]:
    rng = random.Random(SAMPLE_SEED)
    chosen = []
    for task, count in (("tldr", TLDR_SAMPLE), ("paper_qa", QA_SAMPLE)):
        ids = grouped[task]
        if len(ids) < count:
            raise SystemExit(f"need {count} {task} rows, found {len(ids)}")
        chosen.extend(rng.sample(ids, count))
    chosen_set = set(chosen)
    disagreements = []
    for item_id in grouped["paper_qa"]:
        sft_flag = row_unsupported(predictions["sft"][item_id])
        grpo_flag = row_unsupported(predictions["grpo"][item_id])
        if grpo_flag and not sft_flag and item_id not in chosen_set:
            disagreements.append(item_id)
    if len(disagreements) > DISAGREEMENT_SAMPLE:
        disagreements = rng.sample(disagreements, DISAGREEMENT_SAMPLE)
    return chosen + disagreements


def build(root: Path) -> dict:
    loaded = {}
    adapters = {}
    for label, key, relative, adapter in SOURCES:
        metrics_path = root / relative / "metrics.json"
        if key == "rloo" and not metrics_path.exists():
            continue
        loaded[key] = load_metrics(metrics_path)
        loaded[key]["label"] = label
        loaded[key]["source"] = str(metrics_path)
        adapters[key] = adapter
    missing = [key for key in REQUIRED if key not in loaded]
    if missing:
        raise SystemExit(f"missing required runs: {', '.join(missing)}")

    predictions = {}
    for key in loaded:
        relative = dict((item[1], item[2]) for item in SOURCES)[key]
        predictions[key] = index_predictions(root / relative / "predictions.jsonl")

    shared = set(predictions["base"])
    for key in REQUIRED:
        shared &= set(predictions[key])
    if len(shared) < TLDR_SAMPLE + QA_SAMPLE:
        raise SystemExit(f"only {len(shared)} ids are present in every required prediction file")

    grouped = {"tldr": [], "paper_qa": []}
    for item_id in sorted(shared):
        task = predictions["base"][item_id].get("task")
        if task not in grouped:
            raise SystemExit(f"unknown task for {item_id}: {task}")
        grouped[task].append(item_id)

    rows = []
    for label, key, _relative, _adapter in SOURCES:
        if key not in loaded:
            continue
        rows.append(comparison_row(label, loaded[key]))
    if not rows:
        raise SystemExit(f"no comparison rows from {', '.join(loaded)}")
    for row in rows:
        for name in ("Evidence F1", "Unsupported Rate", "ROUGE-L"):
            if row[name] is None:
                raise SystemExit(f"{row['Model']} is missing {name}")
            row[name] = metric_number(row[name], row["Model"], name)
    front = pareto_front(rows)
    chosen, reason = choose(front)
    sample = sample_ids(grouped, predictions)

    output_dir = root / "outputs" / "final_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    qualitative_path = output_dir / "qualitative_samples.jsonl"
    unsupported_counts = {loaded[key]["label"]: 0 for key in loaded}
    with qualitative_path.open("w", encoding="utf-8") as handle:
        for item_id in sample:
            base = predictions["base"][item_id]
            record = {
                "id": item_id,
                "task": base.get("task"),
                "input": base.get("input"),
                "reference": base.get("reference"),
                "base_output": base.get("model_answer"),
                "sft_output": predictions["sft"][item_id].get("model_answer"),
                "dpo_output": predictions["dpo"][item_id].get("model_answer"),
                "rl_output": predictions["grpo"][item_id].get("model_answer"),
            }
            flags = {
                "base": row_unsupported(predictions["base"][item_id]),
                "sft": row_unsupported(predictions["sft"][item_id]),
                "dpo": row_unsupported(predictions["dpo"][item_id]),
                "grpo": row_unsupported(predictions["grpo"][item_id]),
            }
            if "rloo" in predictions and item_id in predictions["rloo"]:
                record["rloo_output"] = predictions["rloo"][item_id].get("model_answer")
                flags["rloo"] = row_unsupported(predictions["rloo"][item_id])
            record["unsupported"] = flags
            for name, flag in flags.items():
                if flag:
                    unsupported_counts[loaded[name]["label"]] += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    fields = [
        "Model",
        "ROUGE-L",
        "BERTScore",
        "Answer F1",
        "Evidence F1",
        "Unsupported Rate",
        "Reward",
        "tokens/s",
        "GPU GB",
    ]
    comparison_path = output_dir / "comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: cell(row[name]) for name in fields})

    best_model = chosen["Model"] if chosen else None
    best_key = next((key for key, value in loaded.items() if value["label"] == best_model), None)
    selection = {
        "rule": RULE,
        "best_model": best_model,
        "best_adapter": adapters.get(best_key) if best_key else None,
        "pareto_front": [item["Model"] for item in front],
        "reason": reason,
        "qualitative_sample_count": len(sample),
        "qualitative_seed": SAMPLE_SEED,
        "qualitative_unsupported_counts": unsupported_counts,
        "gpu_gb_source": "benchmark gpu_peak_memory_gb from evaluate.py, not the training peak",
        "reward_source": "training.reward when that field exists; blank otherwise",
        "candidates": rows,
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "sources": {key: value["source"] for key, value in loaded.items()},
        "comparison": rows,
        "qualitative_samples": str(qualitative_path),
        "qualitative_sample_count": len(sample),
        "selection": selection,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    print(f"qualitative samples {len(sample)}", flush=True)
    for row in rows:
        print(
            f"{row['Model']} rougeL {row['ROUGE-L']} evidence_f1 {row['Evidence F1']} "
            f"unsupported {row['Unsupported Rate']}",
            flush=True,
        )
    return metrics


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build the unified final evaluation from finished runs.")
    parser.parse_args()
    build(root)


if __name__ == "__main__":
    main()
