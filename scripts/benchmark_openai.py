from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time

import aiohttp


async def run_one(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
) -> dict:
    started = time.perf_counter()
    first_token = None
    chunks = 0
    output_tokens = 0
    error = None
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            async for raw in response.content:
                for line in raw.decode(errors="replace").splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    if first_token is None:
                        first_token = time.perf_counter()
                    data = json.loads(line[6:])
                    chunks += 1
                    usage = data.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", output_tokens)
    except Exception as exc:
        error = repr(exc)
    ended = time.perf_counter()
    return {
        "ok": error is None,
        "error": error,
        "latency_s": ended - started,
        "ttft_s": None if first_token is None else first_token - started,
        "chunks": chunks,
        "output_tokens": output_tokens or chunks,
    }


async def benchmark(args) -> dict:
    endpoint = f"{args.base_url}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": args.video_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def limited():
            async with semaphore:
                return await run_one(session, endpoint, payload)

        results = await asyncio.gather(*(limited() for _ in range(args.requests)))
    elapsed = time.perf_counter() - started
    successful = [item for item in results if item["ok"]]
    latencies = [item["latency_s"] for item in successful]
    ttfts = [item["ttft_s"] for item in successful if item["ttft_s"] is not None]
    tokens = sum(item["output_tokens"] for item in successful)
    return {
        "requests": args.requests,
        "successful": len(successful),
        "failed": args.requests - len(successful),
        "concurrency": args.concurrency,
        "elapsed_s": elapsed,
        "requests_per_s": len(successful) / elapsed,
        "output_tokens_per_s": tokens / elapsed,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "ttft_mean_s": statistics.fmean(ttfts) if ttfts else None,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "errors": [item["error"] for item in results if not item["ok"]][:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-4b-flashvid")
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--prompt", default="Describe this video concisely.")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(benchmark(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

