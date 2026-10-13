"""Count Qwen2.5 tokens for the cleaned SciTLDR samples.

Uses the local tokenizer only. The measured string is:

    instruction
    Paper:
    context
    TLDR:
    answer

Writes data/stats/scitldr_token_stats.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LENGTH_CUTOFFS = (2048, 4096, 8192)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sft_preview(record: dict) -> str:
    return (
        f"{record['instruction']}\n"
        f"Paper:\n{record['context']}\n"
        f"TLDR:\n{record['answer']}"
    )


def nearest_rank(sorted_values: list[int], ratio: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, math.ceil(ratio * len(sorted_values)) - 1)
    return int(sorted_values[min(index, len(sorted_values) - 1)])


def summarize_lengths(lengths: list[int]) -> dict:
    ordered = sorted(int(length) for length in lengths)
    count = len(ordered)
    summary = {
        "count": count,
        "mean": round(sum(ordered) / count, 2) if count else 0,
        "p50": nearest_rank(ordered, 0.50),
        "p90": nearest_rank(ordered, 0.90),
        "p95": nearest_rank(ordered, 0.95),
        "p99": nearest_rank(ordered, 0.99),
        "min": ordered[0] if ordered else 0,
        "max": ordered[-1] if ordered else 0,
    }
    for cutoff in LENGTH_CUTOFFS:
        over = sum(length > cutoff for length in ordered)
        summary[f"over_{cutoff}"] = over
        summary[f"over_{cutoff}_ratio"] = round(over / count, 6) if count else 0
    return summary


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tokenizer(model_path: Path):
    if not model_path.exists():
        raise SystemExit(f"missing local tokenizer: {model_path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required on the server to load the Qwen tokenizer") from exc
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def token_lengths(tokenizer, texts: list[str], batch_size: int) -> list[int]:
    lengths = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
        done = min(start + batch_size, len(texts))
        print(f"tokenized {done}/{len(texts)}", flush=True)
    return lengths


def analyze(interim_path: Path, model_path: Path, stats_path: Path, batch_size: int) -> dict:
    records = read_jsonl(interim_path)
    if not records:
        raise SystemExit(f"no records in {interim_path}")
    tokenizer = load_tokenizer(model_path)
    grouped = {"all": records}
    for split in ("train", "validation", "test"):
        grouped[split] = [record for record in records if record["metadata"]["split"] == split]

    stats = {
        "dataset": "scitldr",
        "model_path": str(model_path),
        "add_special_tokens": True,
        "template": "instruction\\nPaper:\\n{context}\\nTLDR:\\n{answer}",
        "splits": {},
    }
    for name, rows in grouped.items():
        lengths = token_lengths(tokenizer, [sft_preview(row) for row in rows], batch_size)
        stats["splits"][name] = summarize_lengths(lengths)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Measure SciTLDR token lengths with the local Qwen tokenizer.")
    parser.add_argument("--interim", type=Path, default=root / "data" / "interim" / "scitldr_clean.jsonl")
    parser.add_argument("--model", type=Path, default=root / "models" / "base" / "Qwen2.5-7B-Instruct")
    parser.add_argument("--stats", type=Path, default=root / "data" / "stats" / "scitldr_token_stats.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    analyze(args.interim, args.model, args.stats, args.batch_size)


if __name__ == "__main__":
    main()
