from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.perception_memory_badcases import (
    attach_target_diagnostics,
    pair_badcases,
    read_manifest_annotations,
    read_result_files,
    summarize_pairs,
    write_badcase_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair current untrained/SFT Fast Hybrid results without model calls."
    )
    parser.add_argument("--untrained", type=Path, nargs="+", required=True)
    parser.add_argument("--sft", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        help="Optional frozen manifests used only for post-hoc target-hit diagnostics.",
    )
    args = parser.parse_args()

    untrained = read_result_files(args.untrained)
    sft = read_result_files(args.sft)
    rows = pair_badcases(untrained, sft)
    if args.manifest:
        attach_target_diagnostics(rows, read_manifest_annotations(args.manifest))
    summary = summarize_pairs(rows)
    write_badcase_report(args.output_dir, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
