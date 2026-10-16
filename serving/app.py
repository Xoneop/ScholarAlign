"""Project API in front of the vLLM server.

The model stays in vLLM. This process only checks the request and calls
http://127.0.0.1:8001/v1. Generation matches evaluation: temperature 0,
max_tokens 512, and the same system/user chat text used in training.

Start vLLM first:

python -m vllm.entrypoints.openai.api_server --model models/final_merged --served-model-name scholaralign --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8001

Set these before that command. `VLLM_USE_FLASHINFER_SAMPLER=0` skips the
FlashInfer sampling JIT. The installed nvcc does not accept
`--compress-mode=size`. `OMP_NUM_THREADS=1` avoids the libgomp warning.

export OMP_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0

Then:

python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1/chat/completions")
VLLM_MODELS_URL = os.environ.get("VLLM_MODELS_URL", "http://127.0.0.1:8001/v1/models")
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "scholaralign")
MAX_TOKENS = 512
TLDR_INSTRUCTION = "Write a concise scientific TLDR that captures the main contribution of the paper."
QA_INSTRUCTION = (
    'Answer the question based only on the paper. Return JSON with keys "answer" and "evidence". '
    'If the paper does not contain the answer, set answer to "Unanswerable" and evidence to [].'
)

app = FastAPI(title="ScholarAlign")


class TLDRRequest(BaseModel):
    paper: str = Field(min_length=1)


class TLDRResponse(BaseModel):
    tldr: str
    completion_tokens: int | None = None


class PaperQARequest(BaseModel):
    paper: str = Field(min_length=1)
    question: str = Field(min_length=1)


class PaperQAResponse(BaseModel):
    answer: str
    evidence: list[str]
    raw: str
    completion_tokens: int | None = None


def post_json(url: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=detail) from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=str(exc.reason)) from exc


def complete(messages: list[dict]) -> tuple[str, int | None]:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    body = post_json(VLLM_URL, payload)
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="vLLM response has no message content") from exc
    usage = body.get("usage") or {}
    tokens = usage.get("completion_tokens")
    return content, tokens if isinstance(tokens, int) else None


def parse_qa(text: str) -> tuple[str, list[str]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return text.strip(), []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return text.strip(), []
    if not isinstance(payload, dict) or "answer" not in payload or "evidence" not in payload:
        return text.strip(), []
    evidence = payload["evidence"] if isinstance(payload["evidence"], list) else []
    return str(payload["answer"]), [str(item) for item in evidence]


@app.get("/health")
def health() -> dict:
    try:
        post_json(VLLM_MODELS_URL)
    except HTTPException as exc:
        raise HTTPException(status_code=503, detail=exc.detail) from exc
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/api/v1/tldr", response_model=TLDRResponse)
def tldr(body: TLDRRequest) -> TLDRResponse:
    messages = [
        {"role": "system", "content": TLDR_INSTRUCTION},
        {"role": "user", "content": f"Paper:\n{body.paper}\nTLDR:\n"},
    ]
    raw, tokens = complete(messages)
    return TLDRResponse(tldr=raw.strip(), completion_tokens=tokens)


@app.post("/api/v1/paper-qa", response_model=PaperQAResponse)
def paper_qa(body: PaperQARequest) -> PaperQAResponse:
    messages = [
        {"role": "system", "content": QA_INSTRUCTION},
        {
            "role": "user",
            "content": f"Paper:\n{body.paper}\nQuestion:\n{body.question}\nAnswer:\n",
        },
    ]
    raw, tokens = complete(messages)
    raw = raw.strip()
    answer, evidence = parse_qa(raw)
    return PaperQAResponse(answer=answer, evidence=evidence, raw=raw, completion_tokens=tokens)
