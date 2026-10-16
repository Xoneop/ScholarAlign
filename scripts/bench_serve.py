"""Measure the FastAPI service at concurrency 1 and 4.

vLLM must already be serving models/final_merged, and serving/app.py must be
listening on port 8000. A failed concurrency is stored and does not erase a
successful one.

Writes:

- logs/deployment/curl_tldr.json
- logs/deployment/curl_qasper.json
- logs/deployment/bench.json
- logs/deployment/run.log
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

API_URL = "http://127.0.0.1:8000"
VLLM_URL = "http://127.0.0.1:8001/v1/chat/completions"
REQUESTS_PER_LEVEL = 8
PAPER = (
    "This paper studies whether a short abstract is enough to identify the claimed result. "
    "The method compares a baseline reader with a reader that must quote a supporting sentence. "
    "On a small held-out set, the quoting reader names the result and points to the sentence "
    "that states it. The paper does not report a human preference study."
)
QUESTION = "What does the quoting reader do?"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_json(url: str, timeout: int = 30) -> dict:
    req = urlrequest.Request(url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: int = 600) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, time.perf_counter() - started


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def gpu_memory_gb() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    first = output.strip().splitlines()[0].strip()
    return round(float(first) / 1024, 3)


def model_size_gb(root: Path) -> float | None:
    meta = root / "models" / "final_merged" / "merge_meta.json"
    if not meta.exists():
        return None
    payload = json.loads(meta.read_text(encoding="utf-8"))
    value = payload.get("output_gb")
    return None if value is None else float(value)


def stream_ttft_ms() -> float | None:
    payload = {
        "model": "scholaralign",
        "messages": [
            {"role": "system", "content": "Write a concise scientific TLDR that captures the main contribution of the paper."},
            {"role": "user", "content": f"Paper:\n{PAPER}\nTLDR:\n"},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(VLLM_URL, data=data, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urlrequest.urlopen(req, timeout=600) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    return round((time.perf_counter() - started) * 1000, 1)
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    return None


def run_level(concurrency: int) -> dict:
    url = f"{API_URL}/api/v1/tldr"
    payload = {"paper": PAPER}

    def once() -> tuple[float, int]:
        body, elapsed = post_json(url, payload)
        tokens = body.get("completion_tokens")
        if not isinstance(tokens, int):
            raise RuntimeError("response has no completion_tokens")
        return elapsed, tokens

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _index: once(), range(REQUESTS_PER_LEVEL)))
    wall = time.perf_counter() - started
    latencies = [item[0] * 1000 for item in results]
    completion_tokens = sum(item[1] for item in results)
    return {
        "concurrency": concurrency,
        "requests": REQUESTS_PER_LEVEL,
        "gpu_memory_gb": gpu_memory_gb(),
        "ttft_ms": stream_ttft_ms(),
        "ttft_note": "one streamed vLLM request after this concurrency level",
        "tokens_per_second": round(completion_tokens / wall, 2) if wall else None,
        "p50_latency_ms": round(percentile(latencies, 0.50), 1),
        "p95_latency_ms": round(percentile(latencies, 0.95), 1),
        "error": None,
    }


def save_curls(log_dir: Path) -> None:
    tldr_body, _elapsed = post_json(f"{API_URL}/api/v1/tldr", {"paper": PAPER})
    qa_body, _elapsed = post_json(
        f"{API_URL}/api/v1/paper-qa",
        {"paper": PAPER, "question": QUESTION},
    )
    (log_dir / "curl_tldr.json").write_text(
        json.dumps({"request": {"paper": PAPER}, "response": tldr_body}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (log_dir / "curl_qasper.json").write_text(
        json.dumps(
            {"request": {"paper": PAPER, "question": QUESTION}, "response": qa_body},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    root = project_root()
    log_dir = root / "logs" / "deployment"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (log_dir / "run.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    try:
        get_json(f"{API_URL}/health")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"FastAPI health check failed: {exc}") from exc

    save_curls(log_dir)
    runs = []
    for concurrency in (1, 4):
        try:
            run = run_level(concurrency)
        except Exception as exc:
            run = {
                "concurrency": concurrency,
                "requests": REQUESTS_PER_LEVEL,
                "gpu_memory_gb": gpu_memory_gb(),
                "ttft_ms": None,
                "tokens_per_second": None,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "error": str(exc),
            }
        runs.append(run)
        log(json.dumps(run, ensure_ascii=False))
    summary = {"model_size_gb": model_size_gb(root), "runs": runs}
    (log_dir / "bench.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log_handle.close()


if __name__ == "__main__":
    main()
