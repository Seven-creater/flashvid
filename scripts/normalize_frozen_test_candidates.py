from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashvid_eval.client import OpenAICompatibleClient  # noqa: E402
from scripts.evaluate_mcq import (  # noqa: E402
    _file_sha256,
    _normalize_frozen_candidates,
    _read_manifest,
)


def _write_immutable_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError(f"normalization summary changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frozen, text-only candidate-normalization cache without "
            "rerunning Direct video inference."
        )
    )
    parser.add_argument("--dataset", choices=("lvbench", "lsdbench", "cgbench"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--api-key", default="no")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()

    for label, path in (
        ("manifest", args.manifest),
        ("candidate results", args.candidate_results),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    manifest_sha256 = _file_sha256(args.manifest)
    candidate_sha256 = _file_sha256(args.candidate_results)
    if (
        args.expected_manifest_sha256
        and manifest_sha256 != args.expected_manifest_sha256
    ):
        raise RuntimeError(
            f"manifest SHA-256 mismatch: expected={args.expected_manifest_sha256} "
            f"actual={manifest_sha256}"
        )
    if (
        args.expected_candidate_sha256
        and candidate_sha256 != args.expected_candidate_sha256
    ):
        raise RuntimeError(
            f"candidate SHA-256 mismatch: expected={args.expected_candidate_sha256} "
            f"actual={candidate_sha256}"
        )

    samples = _read_manifest(args.manifest)
    if not samples:
        raise RuntimeError("manifest is empty")
    if any(sample.dataset != args.dataset for sample in samples):
        raise RuntimeError("manifest contains records from a different dataset")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("manifest contains duplicate sample_id values")

    client = OpenAICompatibleClient(
        args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    answers, sources, reasons, _model_calls, scope_sha256 = (
        _normalize_frozen_candidates(
            args.candidate_results,
            samples,
            client,
            args.model,
            args.output,
            read_only=args.read_only,
        )
    )

    output_rows = [
        json.loads(line)
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_ids = [str(row.get("sample_id") or "") for row in output_rows]
    if len(output_rows) != len(samples) or set(output_ids) != set(sample_ids):
        raise RuntimeError(
            "normalized cache does not contain exactly the frozen manifest sample set"
        )
    if any(int(row.get("direct_rerun", -1)) != 0 for row in output_rows):
        raise RuntimeError("normalized cache contains a Direct rerun")

    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "candidate_results": str(args.candidate_results.resolve()),
        "candidate_results_sha256": candidate_sha256,
        "normalization_cache": str(args.output.resolve()),
        "normalization_cache_sha256": _file_sha256(args.output),
        "normalized_candidate_scope_sha256": scope_sha256,
        "sample_count": len(samples),
        "valid_candidate_count": len(answers),
        "unresolved_candidate_count": len(samples) - len(answers),
        "candidate_source_counts": dict(sorted(Counter(sources.values()).items())),
        "normalization_reason_counts": dict(sorted(Counter(reasons.values()).items())),
        "text_normalizer_candidate_count": sum(
            reason == "text_normalizer" for reason in reasons.values()
        ),
        "candidate_rerun": 0,
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    _write_immutable_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
