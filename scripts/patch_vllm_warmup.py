"""Skip the vLLM 0.31 MiniMax warmup import that crashes Qwen startup.

vLLM 0.31.0 calls kernel_warmup for every model. That function imports
MiniMax M3 Triton kernels before checking the architecture. On this
Python 3.10 install, Triton fails while parsing _topk_index_merge_kernel
and the Qwen engine never finishes starting. The import is wrapped so a
failure becomes a no-op. Qwen weights and the frozen serving settings
are unchanged.
"""

from __future__ import annotations

from pathlib import Path


MARKER = "minimax warmup import skipped"


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    needle = "from vllm.model_executor.warmup.minimax_m3_msa_warmup import ("
    start = text.find(needle)
    if start < 0:
        raise SystemExit("minimax warmup import was not found in kernel_warmup.py")
    line_start = text.rfind("\n", 0, start) + 1
    close = text.find(")", start)
    if close < 0:
        raise SystemExit("minimax warmup import has no closing parenthesis")
    line_end = text.find("\n", close)
    if line_end < 0:
        line_end = len(text)
    block = text[line_start:line_end]
    indent = block[: len(block) - len(block.lstrip(" "))]
    inner = "\n".join(indent + "    " + line[len(indent) :] for line in block.splitlines())
    replacement = (
        f"{indent}try:  # {MARKER}\n"
        f"{inner}\n"
        f"{indent}except Exception:\n"
        f"{indent}    def minimax_m3_msa_warmup(worker):\n"
        f"{indent}        return None\n"
    )
    return text[:line_start] + replacement + text[line_end:]


def main() -> None:
    import vllm.model_executor.warmup.kernel_warmup as warmup

    path = Path(warmup.__file__)
    updated = patch_text(path.read_text(encoding="utf-8"))
    if updated == path.read_text(encoding="utf-8"):
        print(f"already patched {path}", flush=True)
        return
    path.write_text(updated, encoding="utf-8")
    print(f"patched {path}", flush=True)


if __name__ == "__main__":
    main()
