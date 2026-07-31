from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def wait_ready(base_url: str, process: subprocess.Popen, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2):
                return True
        except Exception:
            time.sleep(2)
    return False


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--dp", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-num-seqs", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--max-num-batched-tokens", type=int, nargs="+", default=[16384, 32768]
    )
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sweep"))
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    serve = project / ".venv/bin/flashvid-serve"
    benchmark = project / "scripts/benchmark_openai.py"
    python = project / ".venv/bin/python"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    combinations = itertools.product(
        args.dp, args.max_num_seqs, args.max_num_batched_tokens
    )
    for dp, max_seqs, max_tokens in combinations:
        name = f"dp{dp}-seqs{max_seqs}-tokens{max_tokens}"
        log_path = args.output_dir / f"{name}.log"
        report_path = args.output_dir / f"{name}.json"
        command = [
            str(serve),
            args.model,
            "--vision-retention-ratio",
            str(args.ratio),
            "--served-model-name",
            "qwen3.5-4b-flashvid",
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            str(dp),
            "--max-num-seqs",
            str(max_seqs),
            "--max-num-batched-tokens",
            str(max_tokens),
            "--port",
            str(args.port),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                ready = wait_ready(
                    f"http://127.0.0.1:{args.port}",
                    process,
                    args.startup_timeout,
                )
                if ready:
                    result = subprocess.run(
                        [
                            str(python),
                            str(benchmark),
                            "--video-url",
                            args.video_url,
                            "--requests",
                            str(args.requests),
                            "--concurrency",
                            str(args.concurrency),
                            "--output",
                            str(report_path),
                        ],
                        check=False,
                    )
                    if result.returncode == 0 and report_path.exists():
                        report = json.loads(report_path.read_text())
                        report["config"] = {
                            "dp": dp,
                            "max_num_seqs": max_seqs,
                            "max_num_batched_tokens": max_tokens,
                        }
                        reports.append(report)
            finally:
                stop_process(process)

    valid = [report for report in reports if report["failed"] == 0]
    summary = {
        "best": max(valid, key=lambda item: item["output_tokens_per_s"])
        if valid
        else None,
        "runs": reports,
    }
    output = args.output_dir.parent / "best_config.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not valid:
        raise SystemExit("No failure-free configuration completed")


if __name__ == "__main__":
    main()

