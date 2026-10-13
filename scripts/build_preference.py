"""Build DPO preference pairs for SciTLDR and QASPER.

Chosen is the gold completion. Rejected is a model output when that output
belongs to the same train or validation row and scores worse than gold.
Test-split predictions are ignored, so the benchmark does not enter DPO.
Remaining rows get one rule corruption. QASPER corruptions cover:

wrong_evidence, unsupported_claim, missing_key_information, overly_verbose,
format_error, contradiction, refusal_when_answerable.

Writes:

- data/preference/preference_train.jsonl
- data/preference/preference_val.jsonl
- data/stats/preference_stats.json

The script exits with an error when fewer than 80 of 100 audited training
pairs have a higher score for chosen than for rejected.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from analyze_qasper import read_jsonl
from evaluate import max_answer_f1, parse_json_answer, token_f1

QASPER_TYPES = (
    "wrong_evidence",
    "unsupported_claim",
    "missing_key_information",
    "overly_verbose",
    "format_error",
    "contradiction",
    "refusal_when_answerable",
)
TLDR_TYPES = (
    "missing_key_information",
    "overly_verbose",
    "contradiction",
    "unsupported_claim",
    "format_error",
)
AUDIT_SIZE = 100
AUDIT_THRESHOLD = 0.8


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_answer(answer: str, evidence: list[str]) -> str:
    return json.dumps({"answer": answer, "evidence": evidence}, ensure_ascii=False)


def short_text(text: str, words: int) -> str:
    return " ".join(text.split()[:words])


def donor_evidence(rows: list[dict], index: int, paper_id: str, current: list[str]) -> str | None:
    current_set = {item for item in current if item}
    count = len(rows)
    for shift in range(1, count + 1):
        other = rows[(index + shift) % count]
        if other["metadata"].get("paper_id") == paper_id:
            continue
        for item in other["metadata"].get("evidence") or []:
            if item and item not in current_set:
                return item
    return None


def donor_completion(rows: list[dict], index: int, paper_id: str) -> str:
    current = rows[index].get("completion") or ""
    count = len(rows)
    for shift in range(1, count + 1):
        other = rows[(index + shift) % count]
        if other["metadata"].get("paper_id") == paper_id:
            continue
        text = other.get("completion") or ""
        if text and text != current:
            return text
    return "This statement comes from a different paper."


def corrupt_qasper(row: dict, error_type: str, index: int, rows: list[dict]) -> str | None:
    parsed = parse_json_answer(row["completion"])
    if parsed is None:
        return None
    answer = parsed["answer"]
    evidence = list(parsed["evidence"])
    unanswerable = bool(row["metadata"].get("unanswerable")) or answer.strip().lower() == "unanswerable"
    if error_type == "refusal_when_answerable":
        if unanswerable:
            return None
        return dump_answer("Unanswerable", [])
    if error_type == "wrong_evidence":
        foreign = donor_evidence(rows, index, row["metadata"].get("paper_id", ""), evidence)
        if not foreign:
            return None
        return dump_answer(answer, [foreign])
    if error_type == "unsupported_claim":
        return dump_answer(
            "The method achieves a 99.9 percent gain on a benchmark that this paper does not evaluate.",
            [],
        )
    if error_type == "missing_key_information":
        short_answer = short_text(answer, 2)
        short_evidence = [short_text(evidence[0], 8)] if evidence else []
        corrupted = dump_answer(short_answer, short_evidence)
        if corrupted == row["completion"]:
            return None
        return corrupted
    if error_type == "overly_verbose":
        long_answer = f"{answer} {answer} {answer} This sentence repeats the claim without adding evidence from the paper."
        long_evidence = evidence + evidence + ["Additional commentary restates the same evidence."]
        return dump_answer(long_answer.strip(), long_evidence)
    if error_type == "format_error":
        return '{"answer": "' + answer.replace('"', "") + '", "evidence": ['
    if error_type == "contradiction":
        if answer.strip().lower() == "yes":
            flipped = "No"
        elif answer.strip().lower() == "no":
            flipped = "Yes"
        else:
            flipped = "No. The paper does not support this claim: " + answer
        return dump_answer(flipped, evidence)
    return None


def corrupt_tldr(row: dict, error_type: str, index: int, rows: list[dict]) -> str | None:
    answer = row["completion"].strip()
    if not answer:
        return None
    if error_type == "missing_key_information":
        short_answer = short_text(answer, 2)
        return None if short_answer == answer else short_answer
    if error_type == "overly_verbose":
        return f"{answer} {answer} {answer} This repeats the summary without adding a new contribution."
    if error_type == "contradiction":
        foreign = donor_completion(rows, index, row["metadata"].get("paper_id", ""))
        if foreign == answer:
            return None
        parsed = parse_json_answer(foreign)
        return parsed["answer"] if parsed else foreign
    if error_type == "unsupported_claim":
        return "This paper proves a universal improvement that is not stated in the source."
    if error_type == "format_error":
        return answer + "\n```"
    return None


def corrupt(row: dict, error_type: str, index: int, rows: list[dict]) -> str | None:
    if row["task"] == "paper_qa":
        return corrupt_qasper(row, error_type, index, rows)
    return corrupt_tldr(row, error_type, index, rows)


def rule_rejection(row: dict, index: int, rows: list[dict]) -> tuple[str, str] | None:
    types = QASPER_TYPES if row["task"] == "paper_qa" else TLDR_TYPES
    ordered = types[index % len(types) :] + types[: index % len(types)]
    for error_type in ordered:
        text = corrupt(row, error_type, index, rows)
        if text and text.strip() != row["completion"].strip():
            return text, error_type
    return None


def pair_score(task: str, text: str, gold: str, references: list[str]) -> float:
    if task == "tldr":
        return token_f1(text, gold)
    parsed = parse_json_answer(text)
    gold_parsed = parse_json_answer(gold)
    if parsed is None or gold_parsed is None:
        return 0.0
    answer_score = max_answer_f1(parsed["answer"], references or [gold_parsed["answer"]])
    evidence_score = token_f1(" ".join(parsed["evidence"]), " ".join(gold_parsed["evidence"]))
    return 0.5 * answer_score + 0.5 * evidence_score


def load_predictions(paths: list[Path], allowed_ids: set[str]) -> dict[str, str]:
    found = {}
    for path in paths:
        if not path or not path.exists():
            continue
        for row in read_jsonl(path):
            row_id = row.get("id")
            answer = row.get("model_answer")
            if row_id in allowed_ids and isinstance(answer, str) and answer.strip():
                found[row_id] = answer
    return found


def infer_error_type(row: dict, rejected: str) -> str:
    if row["task"] != "paper_qa":
        if len(rejected) > len(row["completion"]) * 2:
            return "overly_verbose"
        return "missing_key_information"
    parsed = parse_json_answer(rejected)
    gold = parse_json_answer(row["completion"])
    if parsed is None:
        return "format_error"
    if gold and gold["answer"].strip().lower() != "unanswerable" and parsed["answer"].strip().lower() == "unanswerable":
        return "refusal_when_answerable"
    if gold and gold["evidence"] and token_f1(" ".join(parsed["evidence"]), " ".join(gold["evidence"])) == 0:
        return "wrong_evidence"
    if len(rejected) > len(row["completion"]) * 2:
        return "overly_verbose"
    return "unsupported_claim"


def build_split(rows: list[dict], predictions: dict[str, str]) -> tuple[list[dict], Counter, Counter]:
    pairs = []
    sources = Counter()
    skipped = Counter()
    for index, row in enumerate(rows):
        chosen = row["completion"]
        rejected = None
        error_type = None
        source = None
        predicted = predictions.get(row["id"])
        if predicted and predicted.strip() != chosen.strip():
            references = list(row["metadata"].get("references") or [])
            if pair_score(row["task"], predicted, chosen, references) < pair_score(row["task"], chosen, chosen, references):
                rejected = predicted
                error_type = infer_error_type(row, predicted)
                source = "model"
        if rejected is None:
            ruled = rule_rejection(row, index, rows)
            if ruled is None:
                skipped["no_corruption"] += 1
                continue
            rejected, error_type = ruled
            source = "rule"
        pairs.append(
            {
                "id": f"{row['id']}__{error_type}",
                "task": row["task"],
                "prompt": row["prompt_messages"],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "error_type": error_type,
                "source": source,
                "metadata": {
                    "split": row["metadata"]["split"],
                    "paper_id": row["metadata"].get("paper_id"),
                    "origin_id": row["id"],
                },
            }
        )
        sources[source] += 1
    return pairs, sources, skipped


def summarize(pairs: list[dict]) -> dict:
    return {
        "total": len(pairs),
        "by_task": dict(Counter(pair["task"] for pair in pairs)),
        "by_error_type": dict(Counter(pair["error_type"] for pair in pairs)),
        "by_source": dict(Counter(pair["source"] for pair in pairs)),
    }


def audit(pairs: list[dict], rows_by_id: dict[str, dict]) -> dict:
    if not pairs:
        return {"n": 0, "chosen_better": 0, "ratio": 0, "threshold": AUDIT_THRESHOLD, "passed": False}
    sample_size = min(AUDIT_SIZE, len(pairs))
    chosen_pairs = random.Random(42).sample(pairs, sample_size)
    better = 0
    for pair in chosen_pairs:
        origin = rows_by_id[pair["metadata"]["origin_id"]]
        references = list(origin["metadata"].get("references") or [])
        gold = origin["completion"]
        chosen_score = pair_score(pair["task"], pair["chosen"][0]["content"], gold, references)
        rejected_score = pair_score(pair["task"], pair["rejected"][0]["content"], gold, references)
        if chosen_score > rejected_score:
            better += 1
    ratio = round(better / sample_size, 6)
    return {
        "n": sample_size,
        "chosen_better": better,
        "ratio": ratio,
        "threshold": AUDIT_THRESHOLD,
        "passed": ratio >= AUDIT_THRESHOLD,
    }


def build(root: Path, prediction_paths: list[Path]) -> dict:
    train_rows = read_jsonl(root / "data" / "processed" / "sft_train.jsonl")
    val_rows = read_jsonl(root / "data" / "processed" / "sft_val.jsonl")
    allowed = {row["id"] for row in train_rows + val_rows}
    predictions = load_predictions(prediction_paths, allowed)
    train_pairs, _, train_skipped = build_split(train_rows, predictions)
    val_pairs, _, val_skipped = build_split(val_rows, predictions)
    rows_by_id = {row["id"]: row for row in train_rows}
    checked = audit(train_pairs, rows_by_id)
    qasper_train_types = Counter(pair["error_type"] for pair in train_pairs if pair["task"] == "paper_qa")
    stats = {
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "model_prediction_ids_used": len(predictions),
        "test_predictions_excluded": True,
        "skipped": {"train": dict(train_skipped), "validation": dict(val_skipped)},
        "train": summarize(train_pairs),
        "validation": summarize(val_pairs),
        "qasper_train_error_types": dict(qasper_train_types),
        "qasper_required_error_types": list(QASPER_TYPES),
        "missing_qasper_error_types": [name for name in QASPER_TYPES if qasper_train_types[name] == 0],
        "audit": checked,
    }
    preference_dir = root / "data" / "preference"
    stats_path = root / "data" / "stats" / "preference_stats.json"
    write_jsonl(preference_dir / "preference_train.jsonl", train_pairs)
    write_jsonl(preference_dir / "preference_val.jsonl", val_pairs)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["missing_qasper_error_types"] or not checked["passed"]:
        raise SystemExit("preference audit failed; do not start DPO")
    return stats


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build DPO pairs from gold completions and rule corruptions.")
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="*",
        default=[],
        help="Optional model outputs. Only rows whose ids are in train or validation are used.",
    )
    args = parser.parse_args()
    build(root, args.predictions)


if __name__ == "__main__":
    main()
