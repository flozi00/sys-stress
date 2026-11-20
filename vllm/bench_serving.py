#!/usr/bin/env python3
"""Measure serving throughput of an OpenAI-compatible endpoint.

Stdlib only, so it runs on the GPU host without installing anything.

    python3 bench_serving.py --base-url http://localhost:8000 \
        --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --concurrency 1 8

Reports, per concurrency level: time to first token, decode rate per stream
and aggregate output tokens/s.  Streaming is used so TTFT is real and not
hidden behind a non-streamed response.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPT = (
    "Explain in about 200 words why a Grace-Blackwell module can keep mixture-of-experts "
    "weights in host memory without losing throughput. Cover the coherent NVLink-C2C link, "
    "the difference between pinned and pageable memory, and what happens when the working "
    "set exceeds the HBM capacity."
)


def stream_once(base_url: str, model: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                if first is None:
                    first = time.perf_counter()
                chunks += 1
    end = time.perf_counter()
    ttft = (first - start) if first else float("nan")
    completion = usage.get("completion_tokens") or chunks
    decode = completion / (end - first) if first and end > first else float("nan")
    return {"ttft": ttft, "total": end - start, "tokens": completion, "decode": decode}


def run_level(base_url: str, model: str, concurrency: int, max_tokens: int,
              timeout: float) -> dict:
    results: list[dict] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            r = stream_once(base_url, model, max_tokens, timeout)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            r = {"error": f"{type(exc).__name__}: {exc}"}
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start

    ok = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]
    if not ok:
        return {"concurrency": concurrency, "errors": errors[:3]}
    tokens = sum(r["tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests": len(ok),
        "errors": len(errors),
        "ttft_ms": round(statistics.fmean(r["ttft"] for r in ok) * 1000, 1),
        "decode_tok_s": round(statistics.fmean(r["decode"] for r in ok), 1),
        "aggregate_tok_s": round(tokens / wall, 1),
        "wall_s": round(wall, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    print(f"{'conc':>5} {'TTFT ms':>9} {'decode tok/s':>13} {'total tok/s':>12} {'wall s':>8}")
    for level in args.concurrency:
        r = run_level(args.base_url, args.model, level, args.max_tokens, args.timeout)
        if "errors" in r and "ttft_ms" not in r:
            print(f"{level:>5}  FAILED: {r['errors']}")
            continue
        print(f"{r['concurrency']:>5} {r['ttft_ms']:>9.0f} {r['decode_tok_s']:>13.1f} "
              f"{r['aggregate_tok_s']:>12.1f} {r['wall_s']:>8.1f}"
              + (f"  ({r['errors']} errors)" if r["errors"] else ""))


if __name__ == "__main__":
    main()
