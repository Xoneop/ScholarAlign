"""Evaluate one checkpoint on the frozen benchmark split.

The first run scores the base Qwen2.5-7B-Instruct model and writes:

- outputs/base_eval/config.json
- outputs/base_eval/metrics.json
- outputs/base_eval/predictions.jsonl
- outputs/base_eval/run.log

Generation uses the prompt messages stored in the SFT files, greedy decoding,
and the frozen max_new_tokens. Completed ids are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from analyze_qasper import norm, read_jsonl

SPLIT_FILES = {
    "train": "sft_train.jsonl",
    "validation": "sft_val.jsonl",
    "test": "sft_test.jsonl",
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "that", "this", "is", "are", "was", "were", "be", "as", "at", "it", "its",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for handle in self.files:
            handle.write(data)
            handle.flush()

    def flush(self):
        for handle in self.files:
            handle.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.files[0], "encoding", "utf-8")


def load_frozen(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing frozen config: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cutoff_len", "benchmark_split", "max_new_tokens", "bertscore_model"):
        if key not in frozen:
            raise SystemExit(f"{path} has no {key}; run scripts/build_sft_dataset.py first")
    return frozen


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for row in read_jsonl(path):
        if row.get("id"):
            done.add(row["id"])
    return done


def content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [word for word in words if len(word) >= 4 and word not in STOPWORDS]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = norm(prediction).lower().split()
    gold_tokens = norm(reference).lower().split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    from collections import Counter

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_answer_f1(prediction: str, references: list[str]) -> float:
    candidates = [reference for reference in references if reference]
    if not candidates:
        return 0.0
    return max(token_f1(prediction, reference) for reference in candidates)


def parse_json_answer(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "answer" not in payload or "evidence" not in payload:
        return None
    evidence = payload["evidence"] if isinstance(payload["evidence"], list) else []
    return {"answer": str(payload["answer"]), "evidence": [str(item) for item in evidence]}


def is_unsupported(answer: str, evidence: list[str], paper: str) -> bool:
    if norm(answer).lower() == "unanswerable":
        return False
    blob = f"{norm(paper)} {norm(' '.join(evidence))}".lower()
    words = content_words(answer)
    if not words:
        return norm(answer).lower() not in blob
    hits = sum(word in blob for word in words)
    return hits / len(words) < 0.5


def rouge_means(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not predictions:
        return {name: None for name in totals}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for name in totals:
            totals[name] += scores[name].fmeasure
    count = len(predictions)
    return {name: round(total / count, 6) for name, total in totals.items()}


def bertscore_f1(predictions: list[str], references: list[str], model_name: str) -> tuple[float | None, str]:
    if not predictions:
        return None, "no scitldr rows"
    try:
        import bert_score
        import torch
    except ImportError as exc:
        return None, f"import failed: {exc}"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _precision, _recall, f1 = bert_score.score(
            predictions,
            references,
            model_type=model_name,
            lang="en",
            device=device,
            batch_size=16,
            verbose=False,
        )
        return round(float(f1.mean()), 6), "ok"
    except Exception as exc:
        return None, f"failed: {exc}"


def score_rows(rows: list[dict], bertscore_model: str) -> dict:
    tldr = [row for row in rows if row["task"] == "tldr"]
    qa = [row for row in rows if row["task"] == "paper_qa"]
    rouge = rouge_means(
        [row["model_answer"] for row in tldr],
        [row["reference"] for row in tldr],
    )
    bertscore, bertscore_status = bertscore_f1(
        [row["model_answer"] for row in tldr],
        [row["reference"] for row in tldr],
        bertscore_model,
    )
    answer_scores = []
    evidence_scores = []
    unsupported = 0
    format_errors = 0
    for row in qa:
        gold = parse_json_answer(row["reference"])
        if gold is None:
            raise SystemExit(f"gold completion is not JSON: {row['id']}")
        parsed = parse_json_answer(row["model_answer"])
        references = list(row.get("references") or [])
        if gold["answer"] not in references:
            references.append(gold["answer"])
        if parsed is None:
            format_errors += 1
            unsupported += 1
            answer_scores.append(max_answer_f1(row["model_answer"], references))
            evidence_scores.append(0.0 if gold["evidence"] else 1.0)
            continue
        answer_scores.append(max_answer_f1(parsed["answer"], references))
        evidence_scores.append(token_f1(" ".join(parsed["evidence"]), " ".join(gold["evidence"])))
        if is_unsupported(parsed["answer"], parsed["evidence"], row["input"]):
            unsupported += 1
    qa_count = len(qa)
    return {
        "counts": {"tldr": len(tldr), "paper_qa": qa_count},
        "scitldr": {
            "count": len(tldr),
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bertscore": bertscore,
            "bertscore_status": bertscore_status,
            "bertscore_model": bertscore_model,
        },
        "qasper": {
            "count": qa_count,
            "answer_f1": round(sum(answer_scores) / qa_count, 6) if qa_count else None,
            "evidence_f1": round(sum(evidence_scores) / qa_count, 6) if qa_count else None,
            "unsupported_answer_rate": round(unsupported / qa_count, 6) if qa_count else None,
            "format_error_rate": round(format_errors / qa_count, 6) if qa_count else None,
        },
        "definitions": {
            "answer_f1": "Max whitespace-token F1 between the parsed answer and any reference answer. Unparsed output is scored from the raw text.",
            "evidence_f1": "Whitespace-token F1 between joined predicted evidence and joined gold evidence. Unparsed output scores 0 when gold evidence is non-empty.",
            "unsupported_answer_rate": (
                "Share of QASPER rows whose answer is not Unanswerable and fewer than half of its content words "
                "appear in the prompt or the predicted evidence. Unparsed JSON counts as unsupported."
            ),
            "rouge": "rouge-score F1 with stemming.",
        },
    }


def load_model(model_path: Path, load_in_4bit: bool, adapter_path: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"local_files_only": True, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def generate_answer(model, tokenizer, prompt_messages: list[dict], max_new_tokens: int, cutoff_len: int) -> tuple[str, int]:
    import torch

    input_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=cutoff_len,
    )
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, int(generated.shape[0])


def evaluate(
    root: Path,
    frozen_path: Path,
    output_dir: Path,
    load_in_4bit: bool,
    overwrite: bool,
    adapter_path: Path | None = None,
) -> dict:
    try:
        import rouge_score  # noqa: F401
    except ImportError as exc:
        raise SystemExit("rouge-score is required: pip install rouge-score") from exc

    frozen = load_frozen(frozen_path)
    benchmark_split = frozen["benchmark_split"]
    if benchmark_split not in SPLIT_FILES:
        raise SystemExit(f"unknown benchmark_split: {benchmark_split}")
    dataset_path = root / "data" / "processed" / SPLIT_FILES[benchmark_split]
    rows = read_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"no rows in {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    predictions_path = output_dir / "predictions.jsonl"
    log_handle = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    if overwrite and predictions_path.exists():
        predictions_path.unlink()

    model_path = Path(frozen["model_path"])
    cutoff_len = int(frozen["cutoff_len"])
    max_new_tokens = int(frozen["max_new_tokens"])
    config = {
        "model_path": str(model_path),
        "dataset": str(dataset_path),
        "benchmark_split": benchmark_split,
        "cutoff_len": cutoff_len,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "load_in_4bit": load_in_4bit,
        "bertscore_model": frozen["bertscore_model"],
        "chat_format": "system=instruction, user=paper block, assistant=completion",
        "adapter_path": str(adapter_path) if adapter_path else None,
    }
    config_path = output_dir / "config.json"
    keep_training_config = False
    if config_path.exists():
        current = json.loads(config_path.read_text(encoding="utf-8"))
        keep_training_config = "lora_rank" in current
    if not keep_training_config:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    done = completed_ids(predictions_path)
    pending = [row for row in rows if row["id"] not in done]
    print(f"pending {len(pending)}/{len(rows)}", flush=True)

    import torch

    load_allocated_gb = None
    generated_tokens = 0
    elapsed = 0.0
    if pending:
        from analyze_scitldr import load_tokenizer

        tokenizer = load_tokenizer(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"
        model = load_model(model_path, load_in_4bit, adapter_path)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            load_allocated_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
        print(f"load_allocated_gb {load_allocated_gb}", flush=True)

        started = time.perf_counter()
        with predictions_path.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(pending, start=1):
                if "prompt_messages" not in row:
                    raise SystemExit(f"{row['id']} has no prompt_messages; rerun scripts/build_sft_dataset.py")
                answer, token_count = generate_answer(
                    model, tokenizer, row["prompt_messages"], max_new_tokens, cutoff_len
                )
                generated_tokens += token_count
                prediction = {
                    "id": row["id"],
                    "task": row["task"],
                    "input": row["prompt"],
                    "reference": row["completion"],
                    "model_answer": answer,
                    "generated_tokens": token_count,
                    "references": row["metadata"].get("references", []),
                }
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 20 == 0 or index == len(pending):
                    elapsed_now = time.perf_counter() - started
                    rate = round(generated_tokens / elapsed_now, 2) if elapsed_now else 0
                    print(f"generated {index}/{len(pending)} tokens/s {rate}", flush=True)
        elapsed = time.perf_counter() - started
    saved = {}
    for prediction in read_jsonl(predictions_path):
        saved[prediction["id"]] = prediction
    missing = [row["id"] for row in rows if row["id"] not in saved]
    if missing:
        raise SystemExit(f"missing predictions: {len(missing)}")
    predictions = [saved[row["id"]] for row in rows]
    metrics = score_rows(predictions, frozen["bertscore_model"])
    metrics.update(config)
    metrics["load_allocated_gb"] = load_allocated_gb
    if pending and torch.cuda.is_available():
        metrics["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    else:
        metrics["gpu_peak_memory_gb"] = None
    metrics["generated_tokens"] = generated_tokens
    metrics["generation_time_sec"] = round(elapsed, 1)
    metrics["tokens_per_second"] = round(generated_tokens / elapsed, 2) if elapsed and generated_tokens else None
    metrics_path = output_dir / "metrics.json"
    payload = metrics
    if metrics_path.exists():
        current = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "training" in current:
            current["benchmark"] = metrics
            payload = current
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate the frozen benchmark split.")
    parser.add_argument("--frozen", type=Path, default=root / "data" / "stats" / "frozen_config.json")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "base_eval")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate(root, args.frozen, args.output, args.load_in_4bit, args.overwrite, args.adapter)


if __name__ == "__main__":
    main()
