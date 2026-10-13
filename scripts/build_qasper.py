"""Build the QASPER interim set from the local v0.3 JSON files.

Reads data/raw/qasper/qasper-{train,dev,test}-v0.3.json and writes:

- data/interim/qasper_clean.jsonl
- data/stats/qasper_build_stats.json

Official train / validation / test splits stay at paper level. One question
becomes one sample. The training answer prefers an answerable annotation that
has evidence; every annotator answer is kept in references.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_scitldr import as_list, clean_piece

INSTRUCTION = (
    'Answer the question based only on the paper. Return JSON with keys "answer" and "evidence". '
    'If the paper does not contain the answer, set answer to "Unanswerable" and evidence to [].'
)
SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def yes_no_text(value) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"yes", "true", "1"}:
            return "Yes"
        if lowered in {"no", "false", "0"}:
            return "No"
    return ""


def is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def evidence_strings(inner: dict) -> list[str]:
    evidence = [clean_piece(item) for item in as_list(inner.get("evidence"))]
    evidence = [item for item in evidence if item]
    if evidence:
        return evidence
    highlighted = [clean_piece(item) for item in as_list(inner.get("highlighted_evidence"))]
    return [item for item in highlighted if item]


def annotator_view(inner: dict, index: int) -> dict | None:
    if not isinstance(inner, dict):
        return None
    if is_true(inner.get("unanswerable")):
        return {
            "index": index,
            "unanswerable": True,
            "text": "Unanswerable",
            "evidence": [],
            "rank": 9,
            "reference_bits": ["Unanswerable"],
        }
    free = clean_piece(inner.get("free_form_answer"))
    yes_no = yes_no_text(inner.get("yes_no"))
    spans = [clean_piece(item) for item in as_list(inner.get("extractive_spans"))]
    spans = [item for item in spans if item]
    evidence = evidence_strings(inner)
    bits = []
    if free:
        bits.append(free)
    if yes_no:
        bits.append(yes_no)
    bits.extend(spans)
    if free:
        text, rank = free, 0
    elif yes_no:
        text, rank = yes_no, 1
    elif spans:
        text, rank = " ".join(spans), 2
    else:
        return None
    if not evidence:
        rank += 3
    return {
        "index": index,
        "unanswerable": False,
        "text": text,
        "evidence": evidence,
        "rank": rank,
        "reference_bits": bits,
    }


def select_qasper_answer(question: dict) -> dict:
    """Pick one training answer and keep every annotator answer as a reference.

    Rank prefers free-form, then yes/no, then extractive spans. An annotation
    that includes evidence outranks one that does not. Unanswerable is last.
    """
    views = []
    for index, wrapper in enumerate(as_list(question.get("answers"))):
        inner = wrapper.get("answer") if isinstance(wrapper, dict) else None
        view = annotator_view(inner if isinstance(inner, dict) else {}, index)
        if view is not None:
            views.append(view)
    references = dedupe(bit for view in views for bit in view["reference_bits"])
    if not views:
        return {
            "answer": "",
            "evidence": [],
            "references": [],
            "unanswerable": False,
            "gold_available": False,
        }
    selected = min(views, key=lambda view: (view["rank"], view["index"]))
    if selected["text"] not in references:
        references.append(selected["text"])
    return {
        "answer": selected["text"],
        "evidence": [] if selected["unanswerable"] else list(selected["evidence"]),
        "references": references,
        "unanswerable": selected["unanswerable"],
        "gold_available": True,
    }


def body_paragraphs(full_text) -> list[str]:
    blocks = []
    for section in as_list(full_text):
        if not isinstance(section, dict):
            continue
        name = clean_piece(section.get("section_name"))
        for paragraph in as_list(section.get("paragraphs")):
            text = clean_piece(paragraph)
            if not text:
                continue
            blocks.append(f"{name}\n{text}" if name else text)
    return blocks


def compose_context(title: str, abstract: str, body: list[str]) -> str:
    parts = [part for part in (title, abstract, *body) if part]
    return "\n\n".join(parts)


def load_split(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing QASPER split: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"expected a paper dict in {path}")
    return loaded


def build_record(paper_id: str, paper: dict, question: dict, split: str) -> tuple[dict | None, str | None]:
    paper_id = clean_piece(paper_id)
    question_id = clean_piece(question.get("question_id"))
    question_text = clean_piece(question.get("question"))
    title = clean_piece(paper.get("title"))
    abstract = clean_piece(paper.get("abstract"))
    body = body_paragraphs(paper.get("full_text"))
    context = compose_context(title, abstract, body)
    if not paper_id:
        return None, "missing_paper_id"
    if not question_id:
        return None, "missing_question_id"
    if not question_text:
        return None, "missing_question"
    if not context:
        return None, "missing_context"
    selected = select_qasper_answer(question)
    if not selected["gold_available"]:
        return None, "missing_gold"
    if not selected["answer"]:
        return None, "missing_answer"
    record = {
        "id": f"qasper_{question_id}",
        "task": "paper_qa",
        "instruction": INSTRUCTION,
        "context": context,
        "question": question_text,
        "answer": selected["answer"],
        "evidence": selected["evidence"],
        "references": selected["references"],
        "body_paragraphs": body,
        "metadata": {
            "paper_id": paper_id,
            "question_id": question_id,
            "split": split,
            "title": title,
            "abstract": abstract,
            "unanswerable": selected["unanswerable"],
            "gold_available": True,
            "num_references": len(selected["references"]),
            "num_evidence": len(selected["evidence"]),
        },
    }
    return record, None


def overlap_counts(papers_by_split: dict[str, set[str]]) -> dict:
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    report = {}
    for left, right in pairs:
        shared = sorted(papers_by_split[left] & papers_by_split[right])
        report[f"{left}_{right}"] = {"count": len(shared), "paper_ids": shared[:20]}
    return report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def empty_split_stats() -> dict:
    return {
        "papers": 0,
        "questions": 0,
        "valid": 0,
        "answerable": 0,
        "unanswerable": 0,
        "with_evidence": 0,
    }


def build(raw_dir: Path, interim_path: Path, stats_path: Path) -> dict:
    records: list[dict] = []
    reasons: Counter = Counter()
    split_stats = {name: empty_split_stats() for name in SPLIT_FILES}
    papers_by_split = {name: set() for name in SPLIT_FILES}
    seen_questions: set[str] = set()
    raw_papers = 0
    raw_questions = 0

    for split, filename in SPLIT_FILES.items():
        papers = load_split(raw_dir / filename)
        print(f"loaded {split}: {len(papers)} papers", flush=True)
        split_stats[split]["papers"] = len(papers)
        raw_papers += len(papers)
        for paper_id, paper in papers.items():
            papers_by_split[split].add(clean_piece(paper_id))
            questions = paper.get("qas") if isinstance(paper, dict) else None
            for question in as_list(questions):
                raw_questions += 1
                split_stats[split]["questions"] += 1
                if not isinstance(question, dict) or not isinstance(paper, dict):
                    reasons["malformed_question"] += 1
                    continue
                record, reason = build_record(str(paper_id), paper, question, split)
                if record is None:
                    reasons[reason or "invalid"] += 1
                    continue
                question_id = record["metadata"]["question_id"]
                if question_id in seen_questions:
                    reasons["duplicate_question_id"] += 1
                    continue
                seen_questions.add(question_id)
                records.append(record)
                split_stats[split]["valid"] += 1
                if record["metadata"]["unanswerable"]:
                    split_stats[split]["unanswerable"] += 1
                else:
                    split_stats[split]["answerable"] += 1
                if record["evidence"]:
                    split_stats[split]["with_evidence"] += 1

    overlap = overlap_counts(papers_by_split)
    overlap_total = sum(item["count"] for item in overlap.values())
    answerable = sum(item["answerable"] for item in split_stats.values())
    unanswerable = sum(item["unanswerable"] for item in split_stats.values())
    with_evidence = sum(item["with_evidence"] for item in split_stats.values())
    stats = {
        "dataset": "qasper",
        "source": str(raw_dir),
        "raw_papers": raw_papers,
        "raw_questions": raw_questions,
        "valid_rows": len(records),
        "invalid_rows": raw_questions - len(records),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "with_evidence": with_evidence,
        "split_counts": {name: split_stats[name]["valid"] for name in SPLIT_FILES},
        "paper_counts": {name: split_stats[name]["papers"] for name in SPLIT_FILES},
        "splits": split_stats,
        "paper_overlap": overlap,
        "invalid_reasons": dict(reasons),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if overlap_total:
        raise SystemExit(f"paper ids cross official splits: {overlap_total}")
    write_jsonl(interim_path, records)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Clean the local QASPER v0.3 JSON files.")
    parser.add_argument("--raw", type=Path, default=root / "data" / "raw" / "qasper")
    parser.add_argument("--interim", type=Path, default=root / "data" / "interim" / "qasper_clean.jsonl")
    parser.add_argument("--stats", type=Path, default=root / "data" / "stats" / "qasper_build_stats.json")
    args = parser.parse_args()
    build(args.raw, args.interim, args.stats)


if __name__ == "__main__":
    main()
