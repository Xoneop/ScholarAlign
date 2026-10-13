"""Build the SciTLDR interim set from the local Arrow dataset.

Reads data/raw/scitldr/hf_dataset and writes:

- data/interim/scitldr_clean.jsonl
- data/stats/scitldr_build_stats.json

Official train / validation / test splits are kept. One paper stays one sample.
The first reference is the training answer; every reference is stored.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

INSTRUCTION = "Write a concise scientific TLDR that captures the main contribution of the paper."
SPLIT_NAMES = {
    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "dev": "validation",
    "test": "test",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_piece(value) -> str:
    """Normalize one field and keep sentence line breaks."""
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            return str(value)
    return value


def build_record(row: dict, split: str) -> tuple[dict | None, str | None]:
    paper_id = clean_piece(row.get("paper_id"))
    sentences = [clean_piece(sentence) for sentence in as_list(row.get("source"))]
    context = "\n".join(sentence for sentence in sentences if sentence)
    references = [clean_piece(target) for target in as_list(row.get("target"))]
    references = [target for target in references if target]

    if not paper_id:
        return None, "missing_paper_id"
    if not context:
        return None, "missing_context"
    if not references:
        return None, "missing_target"

    ic = row.get("ic")
    if isinstance(ic, str):
        lowered = ic.strip().lower()
        if lowered in {"true", "1"}:
            ic = True
        elif lowered in {"false", "0"}:
            ic = False
    record = {
        "id": f"scitldr_{paper_id}",
        "task": "tldr",
        "instruction": INSTRUCTION,
        "context": context,
        "answer": references[0],
        "references": references,
        "metadata": {
            "paper_id": paper_id,
            "split": split,
            "ic": to_jsonable(ic),
            "source_labels": to_jsonable(as_list(row.get("source_labels"))),
            "rouge_scores": to_jsonable(as_list(row.get("rouge_scores"))),
            "num_source_sentences": len(sentences),
            "num_references": len(references),
        },
    }
    title = clean_piece(row.get("title")) if "title" in row else ""
    if title:
        record["metadata"]["title"] = title
    return record, None


def load_splits(dataset_dir: Path):
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except ImportError as exc:
        raise SystemExit("datasets is required on the server to read hf_dataset") from exc

    if not dataset_dir.exists():
        raise SystemExit(f"missing SciTLDR dataset: {dataset_dir}")

    split_dirs = [name for name in ("train", "validation", "dev", "test") if (dataset_dir / name).is_dir()]
    has_dataset_dict = (dataset_dir / "dataset_dict.json").exists()
    has_single_dataset = (dataset_dir / "state.json").exists() and not has_dataset_dict
    if split_dirs and not has_dataset_dict and not has_single_dataset:
        loaded = DatasetDict({name: load_from_disk(str(dataset_dir / name)) for name in split_dirs})
    else:
        loaded = load_from_disk(str(dataset_dir))

    if isinstance(loaded, Dataset):
        raise SystemExit(
            "hf_dataset contains one split. Save the full DatasetDict with train, validation, and test."
        )
    if not isinstance(loaded, DatasetDict):
        raise SystemExit(f"unsupported dataset object: {type(loaded)}")

    splits = {}
    for name, dataset in loaded.items():
        canonical = SPLIT_NAMES.get(str(name), None)
        if canonical is None:
            raise SystemExit(f"unknown split name: {name}")
        splits[canonical] = dataset
    missing = [name for name in ("train", "validation", "test") if name not in splits]
    if missing:
        raise SystemExit(f"hf_dataset is missing splits: {', '.join(missing)}")
    return splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(dataset_dir: Path, interim_path: Path, stats_path: Path) -> dict:
    records: list[dict] = []
    reasons: Counter = Counter()
    raw_rows = 0
    splits = load_splits(dataset_dir)
    for split in ("train", "validation", "test"):
        for row in splits[split]:
            raw_rows += 1
            record, reason = build_record(dict(row), split)
            if record is None:
                reasons[reason or "invalid"] += 1
                continue
            records.append(record)

    split_counts = Counter(record["metadata"]["split"] for record in records)
    total_references = sum(record["metadata"]["num_references"] for record in records)
    multi_reference_samples = sum(record["metadata"]["num_references"] > 1 for record in records)
    stats = {
        "dataset": "scitldr",
        "source": str(dataset_dir),
        "raw_rows": raw_rows,
        "valid_rows": len(records),
        "invalid_rows": raw_rows - len(records),
        "split_counts": {name: split_counts.get(name, 0) for name in ("train", "validation", "test")},
        "total_references": total_references,
        "multi_reference_samples": multi_reference_samples,
        "invalid_reasons": dict(reasons),
    }
    write_jsonl(interim_path, records)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Clean the local SciTLDR Arrow dataset.")
    parser.add_argument("--dataset", type=Path, default=root / "data" / "raw" / "scitldr" / "hf_dataset")
    parser.add_argument("--interim", type=Path, default=root / "data" / "interim" / "scitldr_clean.jsonl")
    parser.add_argument("--stats", type=Path, default=root / "data" / "stats" / "scitldr_build_stats.json")
    args = parser.parse_args()
    build(args.dataset, args.interim, args.stats)


if __name__ == "__main__":
    main()
