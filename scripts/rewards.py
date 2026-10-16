"""Frozen QASPER reward for GRPO.

R = 1.00 * Answer F1
  + 0.60 * Evidence F1
  + 0.20 * Format
  - 0.50 * Unsupported
  - 0.10 * Length

Answer F1, Evidence F1, and Unsupported use the same rules as scripts/evaluate.py.
Format is 1 when the completion parses as {"answer", "evidence"} JSON, otherwise 0.
Length is 0 through 256 completion tokens and reaches 1 at 768 tokens.
Unparsed JSON counts as unsupported, and its evidence F1 is 0 when gold evidence exists.

Run `python scripts/rewards.py` before GRPO. A failing check must stop training.
"""

from __future__ import annotations

import json

from evaluate import is_unsupported, max_answer_f1, parse_json_answer, token_f1

ANSWER_WEIGHT = 1.0
EVIDENCE_WEIGHT = 0.6
FORMAT_WEIGHT = 0.2
UNSUPPORTED_WEIGHT = 0.5
LENGTH_WEIGHT = 0.1
LENGTH_FREE_TOKENS = 256
LENGTH_SPAN = 512


def length_penalty(token_count: int) -> float:
    if token_count <= LENGTH_FREE_TOKENS:
        return 0.0
    return min((token_count - LENGTH_FREE_TOKENS) / LENGTH_SPAN, 1.0)


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"])
    return str(completion)


def completion_token_count(text: str, completion_ids) -> int:
    if completion_ids is None:
        return len(text.split())
    if hasattr(completion_ids, "shape"):
        return int(completion_ids.shape[-1])
    return len(list(completion_ids))


def score_completion(
    text: str,
    gold_completion: str,
    references: list[str] | None,
    paper: str,
    token_count: int,
) -> dict:
    gold = parse_json_answer(gold_completion)
    if gold is None:
        raise ValueError("gold completion is not QASPER JSON")
    answer_references = list(references or [])
    if gold["answer"] not in answer_references:
        answer_references.append(gold["answer"])
    parsed = parse_json_answer(text)
    if parsed is None:
        parts = {
            "answer_f1": max_answer_f1(text, answer_references),
            "evidence_f1": 0.0 if gold["evidence"] else 1.0,
            "format": 0.0,
            "unsupported": 1.0,
            "length": length_penalty(token_count),
        }
    else:
        parts = {
            "answer_f1": max_answer_f1(parsed["answer"], answer_references),
            "evidence_f1": token_f1(" ".join(parsed["evidence"]), " ".join(gold["evidence"])),
            "format": 1.0,
            "unsupported": 1.0 if is_unsupported(parsed["answer"], parsed["evidence"], paper) else 0.0,
            "length": length_penalty(token_count),
        }
    parts["reward"] = (
        ANSWER_WEIGHT * parts["answer_f1"]
        + EVIDENCE_WEIGHT * parts["evidence_f1"]
        + FORMAT_WEIGHT * parts["format"]
        - UNSUPPORTED_WEIGHT * parts["unsupported"]
        - LENGTH_WEIGHT * parts["length"]
    )
    return parts


def repeat_to(values, count: int) -> list:
    if values is None:
        return [None] * count
    values = list(values)
    if len(values) == count:
        return values
    if values and count % len(values) == 0:
        times = count // len(values)
        return [item for item in values for _ in range(times)]
    raise ValueError(f"cannot align {len(values)} reward columns with {count} completions")


def qasper_reward(
    prompts,
    completions,
    completion_ids=None,
    gold_completion=None,
    references=None,
    references_json=None,
    paper=None,
    **kwargs,
) -> list[float]:
    del prompts, kwargs
    count = len(completions)
    golds = repeat_to(gold_completion, count)
    if references is None and references_json is not None:
        references = [json.loads(item) if isinstance(item, str) else item for item in references_json]
    reference_rows = repeat_to(references, count)
    papers = repeat_to(paper, count)
    ids = repeat_to(completion_ids, count)
    rewards = []
    for completion, gold, reference_row, source, token_ids in zip(completions, golds, reference_rows, papers, ids):
        text = completion_text(completion)
        rewards.append(
            score_completion(
                text,
                gold,
                list(reference_row or []),
                source or "",
                completion_token_count(text, token_ids),
            )["reward"]
        )
    return rewards


def _json(answer: str, evidence: list[str]) -> str:
    return json.dumps({"answer": answer, "evidence": evidence}, ensure_ascii=False)


def check() -> None:
    paper = "The proposed method improves accuracy by four points on the benchmark in table two."
    answer = "The method improves accuracy by four points."
    evidence = ["The proposed method improves accuracy by four points on the benchmark in table two."]
    gold = _json(answer, evidence)
    good = score_completion(_json(answer, evidence), gold, [answer], paper, 32)
    wrong_evidence = score_completion(
        _json(answer, ["This sentence comes from a different paper about speech recognition."]),
        gold,
        [answer],
        paper,
        32,
    )
    unsupported = score_completion(
        _json("The method achieves a 99.9 percent gain on an unnamed private benchmark.", []),
        gold,
        [answer],
        paper,
        32,
    )
    bad_format = score_completion(answer, gold, [answer], paper, 32)
    long_good = score_completion(_json(answer, evidence), gold, [answer], paper, LENGTH_FREE_TOKENS + LENGTH_SPAN)
    failures = []
    if not good["reward"] > wrong_evidence["reward"]:
        failures.append(f"good {good['reward']} is not above wrong evidence {wrong_evidence['reward']}")
    if not good["reward"] > unsupported["reward"]:
        failures.append(f"good {good['reward']} is not above unsupported {unsupported['reward']}")
    if not good["reward"] > bad_format["reward"]:
        failures.append(f"good {good['reward']} is not above bad format {bad_format['reward']}")
    if not good["reward"] > long_good["reward"]:
        failures.append(f"short good {good['reward']} is not above long good {long_good['reward']}")
    if length_penalty(LENGTH_FREE_TOKENS) != 0.0 or length_penalty(LENGTH_FREE_TOKENS + LENGTH_SPAN) != 1.0:
        failures.append("length penalty endpoints are wrong")
    if failures:
        raise SystemExit("reward unit test failed: " + "; ".join(failures))
    print(
        json.dumps(
            {
                "good": good["reward"],
                "wrong_evidence": wrong_evidence["reward"],
                "unsupported": unsupported["reward"],
                "bad_format": bad_format["reward"],
                "long_good": long_good["reward"],
                "passed": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    check()
