"""Count Qwen2.5 tokens for cleaned QASPER samples and evidence retention.

The measured string is:

    instruction
    Paper:
    context
    Question:
    question
    Answer:
    {"answer": "...", "evidence": [...]}

Evidence retention at 2048 and 4096 keeps the title and abstract, then gold
evidence paragraphs that fit, then earlier body paragraphs. A sample is
retained when every gold evidence string is still in that truncated paper.

Writes data/stats/qasper_token_stats.json. When SciTLDR train p95 is already
on disk, also writes data/stats/frozen_config.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_scitldr import load_tokenizer, summarize_lengths, token_lengths
from build_qasper import compose_context

RETENTION_CUTOFFS = (2048, 4096)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def completion_text(record: dict) -> str:
    payload = {"answer": record["answer"], "evidence": record["evidence"]}
    return json.dumps(payload, ensure_ascii=False)


def sft_preview(record: dict) -> str:
    return (
        f"{record['instruction']}\n"
        f"Paper:\n{record['context']}\n"
        f"Question:\n{record['question']}\n"
        f"Answer:\n{completion_text(record)}"
    )


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def length_map(tokenizer, texts: list[str], batch_size: int, add_special_tokens: bool) -> dict[str, int]:
    unique = list(dict.fromkeys(texts))
    mapping = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )
        for text, input_ids in zip(batch, encoded["input_ids"]):
            mapping[text] = len(input_ids)
        done = min(start + batch_size, len(unique))
        print(f"encoded pieces {done}/{len(unique)}", flush=True)
    return mapping


def header_context(record: dict) -> str:
    metadata = record["metadata"]
    return compose_context(metadata.get("title") or "", metadata.get("abstract") or "", [])


def no_body_preview(record: dict) -> str:
    return (
        f"{record['instruction']}\n"
        f"Paper:\n{header_context(record)}\n"
        f"Question:\n{record['question']}\n"
        f"Answer:\n{completion_text(record)}"
    )


def evidence_flags(body: list[str], evidence: list[str]) -> list[bool]:
    normalized_body = [norm(block) for block in body]
    flags = []
    for block in normalized_body:
        flags.append(any(span and span in block for span in evidence))
    return flags


def select_indexes(costs: list[int], flags: list[bool], budget: int) -> list[int]:
    order = [index for index, flag in enumerate(flags) if flag]
    order.extend(index for index, flag in enumerate(flags) if not flag)
    chosen = []
    remaining = budget
    for index in order:
        cost = costs[index]
        if cost <= remaining:
            chosen.append(index)
            remaining -= cost
    chosen.sort()
    return chosen


def retention_for_cutoff(record: dict, full_len: int, floor_len: int, costs: list[int], cutoff: int) -> dict:
    evidence = [norm(span) for span in record["evidence"] if norm(span)]
    full_text = norm(record["context"])
    absent = sum(span not in full_text for span in evidence)
    if full_len <= cutoff:
        kept = sum(span in full_text for span in evidence)
        return {
            "spans": len(evidence),
            "spans_retained": kept,
            "sample_retained": kept == len(evidence),
            "zero_body_budget": False,
            "absent_from_full_text": absent,
        }
    body = list(record.get("body_paragraphs") or [])
    flags = evidence_flags(body, evidence)
    budget = cutoff - floor_len
    chosen = select_indexes(costs, flags, budget) if budget > 0 else []
    kept_blocks = [body[index] for index in chosen]
    truncated = norm(
        compose_context(
            record["metadata"].get("title") or "",
            record["metadata"].get("abstract") or "",
            kept_blocks,
        )
    )
    kept = sum(span in truncated for span in evidence)
    return {
        "spans": len(evidence),
        "spans_retained": kept,
        "sample_retained": kept == len(evidence),
        "zero_body_budget": budget <= 0,
        "absent_from_full_text": absent,
    }


def empty_retention() -> dict:
    return {
        "samples_with_evidence": 0,
        "samples_retained": 0,
        "sample_retention_ratio": 0,
        "evidence_spans": 0,
        "spans_retained": 0,
        "span_retention_ratio": 0,
        "samples_with_zero_body_budget": 0,
        "spans_absent_from_full_text": 0,
    }


def accumulate(bucket: dict, result: dict) -> None:
    bucket["samples_with_evidence"] += 1
    bucket["samples_retained"] += int(result["sample_retained"])
    bucket["evidence_spans"] += result["spans"]
    bucket["spans_retained"] += result["spans_retained"]
    bucket["samples_with_zero_body_budget"] += int(result["zero_body_budget"])
    bucket["spans_absent_from_full_text"] += result["absent_from_full_text"]


def finalize_retention(bucket: dict) -> dict:
    samples = bucket["samples_with_evidence"]
    spans = bucket["evidence_spans"]
    bucket["sample_retention_ratio"] = round(bucket["samples_retained"] / samples, 6) if samples else 0
    bucket["span_retention_ratio"] = round(bucket["spans_retained"] / spans, 6) if spans else 0
    return bucket


def freeze_cutoff(scitldr_p95: int, qasper_p95: int) -> int:
    p95 = max(int(scitldr_p95), int(qasper_p95))
    if p95 <= 2048:
        return 2048
    return 4096


def read_scitldr_train_p95(path: Path) -> int | None:
    if not path.exists():
        return None
    stats = json.loads(path.read_text(encoding="utf-8"))
    return int(stats["splits"]["train"]["p95"])


def analyze(
    interim_path: Path,
    model_path: Path,
    stats_path: Path,
    scitldr_stats_path: Path,
    frozen_path: Path,
    batch_size: int,
) -> dict:
    records = read_jsonl(interim_path)
    if not records:
        raise SystemExit(f"no records in {interim_path}")
    for record in records:
        if "body_paragraphs" not in record:
            raise SystemExit("qasper_clean.jsonl has no body_paragraphs; rerun scripts/build_qasper.py")
    tokenizer = load_tokenizer(model_path)
    previews = [sft_preview(record) for record in records]
    lengths = token_lengths(tokenizer, previews, batch_size)
    grouped_lengths = {"all": lengths, "train": [], "validation": [], "test": []}
    for record, length in zip(records, lengths):
        grouped_lengths[record["metadata"]["split"]].append(length)

    floors = [no_body_preview(record) for record in records]
    body_keys = []
    piece_texts = []
    for record in records:
        keys = [f"\n\n{block}" for block in record["body_paragraphs"]]
        body_keys.append(keys)
        piece_texts.extend(keys)
    pieces = length_map(tokenizer, piece_texts, batch_size, add_special_tokens=False)
    floor_lengths = length_map(tokenizer, floors, batch_size, add_special_tokens=True)

    retention = {
        name: {str(cutoff): empty_retention() for cutoff in RETENTION_CUTOFFS}
        for name in ("all", "train", "validation", "test")
    }
    for index, record in enumerate(records):
        evidence = [span for span in record["evidence"] if norm(span)]
        if not evidence:
            continue
        costs = [pieces[key] for key in body_keys[index]]
        floor_len = floor_lengths[floors[index]]
        split = record["metadata"]["split"]
        for cutoff in RETENTION_CUTOFFS:
            result = retention_for_cutoff(record, lengths[index], floor_len, costs, cutoff)
            accumulate(retention["all"][str(cutoff)], result)
            accumulate(retention[split][str(cutoff)], result)
        if (index + 1) % 500 == 0:
            print(f"retention {index + 1}/{len(records)}", flush=True)
    for buckets in retention.values():
        for cutoff in RETENTION_CUTOFFS:
            finalize_retention(buckets[str(cutoff)])

    summaries = {name: summarize_lengths(grouped_lengths[name]) for name in ("all", "train", "validation", "test")}
    qasper_train_p95 = summaries["train"]["p95"]
    scitldr_train_p95 = read_scitldr_train_p95(scitldr_stats_path)
    cutoff_len = None
    if scitldr_train_p95 is not None:
        cutoff_len = freeze_cutoff(scitldr_train_p95, qasper_train_p95)
        frozen = {
            "cutoff_len": cutoff_len,
            "rule": "max(scitldr_train_p95, qasper_train_p95): <=2048 -> 2048, else 4096",
            "scitldr_train_p95": scitldr_train_p95,
            "qasper_train_p95": qasper_train_p95,
            "model_path": str(model_path),
        }
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(frozen, ensure_ascii=False, indent=2))

    stats = {
        "dataset": "qasper",
        "model_path": str(model_path),
        "add_special_tokens": True,
        "template": 'instruction\\nPaper:\\n{context}\\nQuestion:\\n{question}\\nAnswer:\\n{"answer","evidence"}',
        "evidence_retention_definition": (
            "Hard token cap. Always keep title and abstract. Then keep evidence paragraphs that fit, "
            "then earlier body paragraphs. A sample is retained when every gold evidence string remains "
            "in the truncated paper text. Samples with empty evidence are excluded."
        ),
        "splits": summaries,
        "evidence_retention": retention,
        "scitldr_train_p95": scitldr_train_p95,
        "qasper_train_p95": qasper_train_p95,
        "cutoff_len": cutoff_len,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Measure QASPER token lengths and evidence retention.")
    parser.add_argument("--interim", type=Path, default=root / "data" / "interim" / "qasper_clean.jsonl")
    parser.add_argument("--model", type=Path, default=root / "models" / "base" / "Qwen2.5-7B-Instruct")
    parser.add_argument("--stats", type=Path, default=root / "data" / "stats" / "qasper_token_stats.json")
    parser.add_argument("--scitldr-stats", type=Path, default=root / "data" / "stats" / "scitldr_token_stats.json")
    parser.add_argument("--frozen", type=Path, default=root / "data" / "stats" / "frozen_config.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    analyze(args.interim, args.model, args.stats, args.scitldr_stats, args.frozen, args.batch_size)


if __name__ == "__main__":
    main()
