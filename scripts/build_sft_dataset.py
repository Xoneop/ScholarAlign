"""Build cutoff-limited SFT rows for SciTLDR and QASPER.

Reads the frozen cutoff and writes:

- data/processed/sft_train.jsonl
- data/processed/sft_val.jsonl
- data/processed/sft_test.jsonl
- data/stats/truncation_stats.json

Train rows are the only training split. Validation and test are stored for
evaluation and do not enter training. QASPER test has gold labels in the
cleaned data, so benchmark_split is frozen as test.

The chat sequence is system=instruction, user=paper block, assistant=answer.
SciTLDR context over the cutoff keeps the first 75% and last 25% of the
context token budget. QASPER keeps the title and abstract, then evidence
paragraphs that fit, then earlier body paragraphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_qasper import completion_text, evidence_flags, norm, read_jsonl, select_indexes
from analyze_scitldr import load_tokenizer
from build_qasper import compose_context

MAX_NEW_TOKENS = 512
BERTSCORE_MODEL = "distilbert-base-uncased"
SPLIT_FILES = {
    "train": "sft_train.jsonl",
    "validation": "sft_val.jsonl",
    "test": "sft_test.jsonl",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_frozen(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing frozen config: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if "cutoff_len" not in frozen:
        raise SystemExit(f"{path} has no cutoff_len")
    return frozen


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def batch_lengths(tokenizer, texts: list[str], batch_size: int, add_special_tokens: bool) -> list[int]:
    lengths = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
        done = min(start + batch_size, len(texts))
        print(f"tokenized {done}/{len(texts)}", flush=True)
    return lengths


def chat_text(tokenizer, prompt_messages: list[dict], completion: str) -> str:
    messages = list(prompt_messages) + [{"role": "assistant", "content": completion}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not isinstance(rendered, str):
        raise SystemExit("chat template did not return a string")
    return rendered


def build_prompt(record: dict, context: str) -> tuple[list[dict], str, str]:
    instruction = record["instruction"]
    if record["task"] == "tldr":
        user = f"Paper:\n{context}\nTLDR:\n"
        completion = record["answer"]
    elif record["task"] == "paper_qa":
        user = (
            f"Paper:\n{context}\n"
            f"Question:\n{record['question']}\n"
            f"Answer:\n"
        )
        completion = completion_text(record)
    else:
        raise SystemExit(f"unknown task: {record.get('task')}")
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]
    return messages, f"{instruction}\n{user}", completion


def clip_context(tokenizer, context: str, budget: int) -> str:
    ids = tokenizer(context, add_special_tokens=False, truncation=False)["input_ids"]
    if budget >= len(ids):
        return context
    if budget <= 0:
        return ""
    head = (budget * 3) // 4
    tail = budget - head
    parts = []
    if head:
        parts.append(tokenizer.decode(ids[:head], skip_special_tokens=True))
    if tail:
        parts.append(tokenizer.decode(ids[-tail:], skip_special_tokens=True))
    return "\n\n".join(part for part in parts if part)


def chat_length(tokenizer, prompt_messages: list[dict], completion: str) -> int:
    rendered = chat_text(tokenizer, prompt_messages, completion)
    return len(tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"])


def fit_scitldr(tokenizer, record: dict, cutoff: int, source_tokens: int) -> tuple[list[dict], str, str, int]:
    del source_tokens
    ids = tokenizer(record["context"], add_special_tokens=False, truncation=False)["input_ids"]
    completion = record["answer"]
    messages, flat, _ = build_prompt(record, "")
    fallback = (messages, flat, completion, chat_length(tokenizer, messages, completion))
    budget = len(ids) - 1
    step = max(1, len(ids) // 20)
    while budget > 0:
        context = clip_context(tokenizer, record["context"], budget)
        messages, flat, _ = build_prompt(record, context)
        packed = chat_length(tokenizer, messages, completion)
        if packed <= cutoff:
            return messages, flat, completion, packed
        budget -= step
    return fallback


def qasper_context(record: dict, indexes: list[int]) -> str:
    metadata = record["metadata"]
    body = [record["body_paragraphs"][index] for index in sorted(indexes)]
    return compose_context(metadata.get("title") or "", metadata.get("abstract") or "", body)


def drop_body_index(chosen: list[int], flags: list[bool]) -> int:
    fillers = [index for index in chosen if not flags[index]]
    if fillers:
        return max(fillers)
    return max(chosen)


def fit_qasper(
    tokenizer,
    record: dict,
    cutoff: int,
    source_tokens: int,
    costs: list[int],
) -> tuple[list[dict], str, str, int]:
    del source_tokens
    completion = completion_text(record)
    body = list(record["body_paragraphs"])
    evidence = [norm(span) for span in record.get("evidence") or [] if norm(span)]
    flags = evidence_flags(body, evidence)
    header = qasper_context(record, [])
    header_messages, _, _ = build_prompt(record, header)
    header_tokens = chat_length(tokenizer, header_messages, completion)
    if header_tokens > cutoff or not body:
        return header_messages, build_prompt(record, header)[1], completion, header_tokens

    chosen = select_indexes(costs, flags, max(0, cutoff - header_tokens))
    messages, flat, _ = build_prompt(record, qasper_context(record, chosen))
    packed = chat_length(tokenizer, messages, completion)
    while packed > cutoff and chosen:
        chosen.remove(drop_body_index(chosen, flags))
        messages, flat, _ = build_prompt(record, qasper_context(record, chosen))
        packed = chat_length(tokenizer, messages, completion)
    if not chosen and packed > cutoff:
        return header_messages, build_prompt(record, header)[1], completion, header_tokens
    return messages, flat, completion, packed if chosen else header_tokens


def empty_bucket() -> dict:
    return {
        "count": 0,
        "truncated": 0,
        "truncated_ratio": 0,
        "still_over_cutoff": 0,
        "packed_tokens_mean": 0,
        "_packed_sum": 0,
    }


def add_bucket(bucket: dict, truncated: bool, packed: int, cutoff: int) -> None:
    bucket["count"] += 1
    bucket["truncated"] += int(truncated)
    bucket["still_over_cutoff"] += int(packed > cutoff)
    bucket["_packed_sum"] += packed


def finish_bucket(bucket: dict) -> dict:
    count = bucket["count"]
    bucket["truncated_ratio"] = round(bucket["truncated"] / count, 6) if count else 0
    bucket["packed_tokens_mean"] = round(bucket["_packed_sum"] / count, 2) if count else 0
    bucket.pop("_packed_sum")
    return bucket


def choose_benchmark_split(qasper_rows: list[dict]) -> tuple[str, str]:
    test_rows = [row for row in qasper_rows if row["metadata"]["split"] == "test"]
    missing = [
        row for row in test_rows
        if not row["metadata"].get("gold_available", True) or not row.get("answer")
    ]
    if not test_rows or missing:
        return "validation", "QASPER test split has samples without gold labels"
    return "test", "QASPER test split has gold labels"


def update_frozen(path: Path, frozen: dict, benchmark_split: str, reason: str) -> dict:
    existing = frozen.get("benchmark_split")
    if existing is not None and existing != benchmark_split:
        raise SystemExit(f"benchmark_split is already frozen as {existing}")
    existing_tokens = frozen.get("max_new_tokens")
    if existing_tokens is not None and int(existing_tokens) != MAX_NEW_TOKENS:
        raise SystemExit(f"max_new_tokens is already frozen as {existing_tokens}")
    existing_bert = frozen.get("bertscore_model")
    if existing_bert is not None and existing_bert != BERTSCORE_MODEL:
        raise SystemExit(f"bertscore_model is already frozen as {existing_bert}")
    frozen["benchmark_split"] = benchmark_split
    frozen["benchmark_split_reason"] = frozen.get("benchmark_split_reason") or reason
    frozen["max_new_tokens"] = MAX_NEW_TOKENS
    frozen["do_sample"] = False
    frozen["bertscore_model"] = BERTSCORE_MODEL
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    return frozen


def sft_row(
    record: dict,
    prompt_messages: list[dict],
    flat_prompt: str,
    completion: str,
    truncated: bool,
    source_tokens: int,
    packed_tokens: int,
) -> dict:
    metadata = {
        "paper_id": record["metadata"]["paper_id"],
        "split": record["metadata"]["split"],
        "dataset": "scitldr" if record["task"] == "tldr" else "qasper",
        "truncated": truncated,
        "source_tokens": source_tokens,
        "packed_tokens": packed_tokens,
    }
    if record["task"] == "paper_qa":
        metadata["question_id"] = record["metadata"]["question_id"]
        metadata["unanswerable"] = bool(record["metadata"].get("unanswerable"))
        metadata["references"] = list(record.get("references") or [])
        metadata["evidence"] = list(record.get("evidence") or [])
    return {
        "id": record["id"],
        "task": record["task"],
        "prompt_messages": prompt_messages,
        "prompt": flat_prompt,
        "completion": completion,
        "metadata": metadata,
    }


def pack_records(tokenizer, records: list[dict], cutoff: int, batch_size: int) -> list[dict]:
    if not records:
        return []
    prompts = [build_prompt(record, record["context"]) for record in records]
    texts = [chat_text(tokenizer, messages, completion) for messages, _, completion in prompts]
    print(f"measuring {records[0]['task']} chat length", flush=True)
    lengths = batch_lengths(tokenizer, texts, batch_size, add_special_tokens=False)
    packed_rows = []
    over = []
    for record, (messages, flat, completion), source_tokens in zip(records, prompts, lengths):
        if source_tokens <= cutoff:
            packed_rows.append((record, sft_row(record, messages, flat, completion, False, source_tokens, source_tokens)))
        else:
            over.append((record, source_tokens))
    print(f"{records[0]['task']} within cutoff {len(packed_rows)}/{len(records)}", flush=True)

    if records[0]["task"] == "tldr":
        for index, (record, source_tokens) in enumerate(over, start=1):
            messages, flat, completion, packed = fit_scitldr(tokenizer, record, cutoff, source_tokens)
            packed_rows.append((record, sft_row(record, messages, flat, completion, True, source_tokens, packed)))
            print(f"truncated tldr {index}/{len(over)}", flush=True)
    else:
        packed_rows.extend(truncate_qasper(tokenizer, over, cutoff, batch_size))

    order = {record["id"]: index for index, record in enumerate(records)}
    packed_rows.sort(key=lambda item: order[item[0]["id"]])
    return [row for _, row in packed_rows]


def truncate_qasper(tokenizer, over: list[tuple[dict, int]], cutoff: int, batch_size: int) -> list[tuple[dict, dict]]:
    piece_texts = []
    cost_rows = []
    for record, _source_tokens in over:
        keys = [f"\n\n{block}" for block in record["body_paragraphs"]]
        cost_rows.append(keys)
        piece_texts.extend(keys)
    pieces = {}
    if piece_texts:
        unique = list(dict.fromkeys(piece_texts))
        unique_lengths = batch_lengths(tokenizer, unique, batch_size, add_special_tokens=False)
        pieces = dict(zip(unique, unique_lengths))
    packed_rows = []
    for index, ((record, source_tokens), keys) in enumerate(zip(over, cost_rows), start=1):
        costs = [pieces[key] for key in keys]
        messages, flat, completion, packed = fit_qasper(tokenizer, record, cutoff, source_tokens, costs)
        original_user = build_prompt(record, record["context"])[0][1]["content"]
        truncated = messages[1]["content"] != original_user
        packed_rows.append((record, sft_row(record, messages, flat, completion, truncated, source_tokens, packed)))
        if index % 200 == 0 or index == len(over):
            print(f"truncated paper_qa {index}/{len(over)}", flush=True)
    return packed_rows


def build(root: Path, frozen_path: Path, stats_path: Path, batch_size: int) -> dict:
    frozen = load_frozen(frozen_path)
    cutoff = int(frozen["cutoff_len"])
    tokenizer = load_tokenizer(Path(frozen.get("model_path") or root / "models" / "base" / "Qwen2.5-7B-Instruct"))
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("tokenizer has no chat_template")

    scitldr = read_jsonl(root / "data" / "interim" / "scitldr_clean.jsonl")
    qasper = read_jsonl(root / "data" / "interim" / "qasper_clean.jsonl")
    for record in qasper:
        if "body_paragraphs" not in record:
            raise SystemExit("qasper_clean.jsonl has no body_paragraphs; rerun scripts/build_qasper.py")
    benchmark_split, reason = choose_benchmark_split(qasper)

    grouped = {name: [] for name in SPLIT_FILES}
    buckets = {
        task: {split: empty_bucket() for split in SPLIT_FILES}
        for task in ("tldr", "paper_qa")
    }
    for task, records in (("tldr", scitldr), ("paper_qa", qasper)):
        for row in pack_records(tokenizer, records, cutoff, batch_size):
            split = row["metadata"]["split"]
            if split not in grouped:
                raise SystemExit(f"unknown split: {split}")
            grouped[split].append(row)
            add_bucket(buckets[task][split], row["metadata"]["truncated"], row["metadata"]["packed_tokens"], cutoff)

    processed = root / "data" / "processed"
    for split, filename in SPLIT_FILES.items():
        write_jsonl(processed / filename, grouped[split])

    for task_buckets in buckets.values():
        for bucket in task_buckets.values():
            finish_bucket(bucket)
    stats = {
        "cutoff_len": cutoff,
        "benchmark_split": benchmark_split,
        "benchmark_split_reason": reason,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "bertscore_model": BERTSCORE_MODEL,
        "policy": {
            "scitldr": "When the chat sequence exceeds cutoff_len, keep the first 75% and last 25% of the context token budget.",
            "qasper": "When the chat sequence exceeds cutoff_len, keep title and abstract, then evidence paragraphs that fit, then earlier body paragraphs.",
        },
        "chat_format": "system=instruction, user=paper block, assistant=completion",
        "counts": {split: len(rows) for split, rows in grouped.items()},
        "tasks": buckets,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen = update_frozen(frozen_path, frozen, benchmark_split, reason)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(frozen, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build cutoff-limited SciTLDR and QASPER SFT files.")
    parser.add_argument("--frozen", type=Path, default=root / "data" / "stats" / "frozen_config.json")
    parser.add_argument("--stats", type=Path, default=root / "data" / "stats" / "truncation_stats.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    build(root, args.frozen, args.stats, args.batch_size)


if __name__ == "__main__":
    main()
