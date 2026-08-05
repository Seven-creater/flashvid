#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from flashvid_eval.fast_hybrid_sft import (
    build_fast_hybrid_sft_records,
    validate_fast_hybrid_frame_files,
)
from flashvid_eval.privacy import assert_deferred_result_public
from flashvid_eval.qwen_sft import read_jsonl, sha256_file


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export selected Fast Hybrid EVA traces for ms-swift SFT.")
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--minimum-total", type=int, default=300)
    parser.add_argument("--minimum-per-dataset", type=int, default=80)
    args = parser.parse_args()

    if args.minimum_total < 1 or args.minimum_per_dataset < 1:
        raise ValueError("SFT minimum thresholds must be positive integers")

    selected = read_jsonl(args.selected)
    counts = Counter(str(row.get("dataset") or "") for row in selected)
    if len(selected) < args.minimum_total or any(
        counts.get(dataset, 0) < args.minimum_per_dataset
        for dataset in ("lvbench", "lsdbench", "cgbench")
    ):
        raise ValueError(
            "SFT start gate failed: need "
            f"{args.minimum_total} stable samples and at least "
            f"{args.minimum_per_dataset} per dataset"
        )
    records: list[dict] = []
    for row in selected:
        assert_deferred_result_public(row)
        if row.get("_selection_stable") is not True:
            raise ValueError("SFT input contains a trajectory without the stable-selection marker")
        validate_fast_hybrid_frame_files(row)
        records.extend(build_fast_hybrid_sft_records(row))
    _atomic_jsonl(args.output, records)
    summary = {
        "selected": len(selected),
        "selected_by_dataset": dict(sorted(counts.items())),
        "sft_records": len(records),
        "selected_sha256": sha256_file(args.selected),
        "sft_sha256": sha256_file(args.output),
        "final_targets": sum(
            record["metadata"]["assistant_target_types"].count("final")
            for record in records
        ),
        "tool_targets": sum(
            record["metadata"]["assistant_target_types"].count("tool")
            for record in records
        ),
        "minimum_total": args.minimum_total,
        "minimum_per_dataset": args.minimum_per_dataset,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
