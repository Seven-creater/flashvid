from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.sweep_reporting import build_sweep_report, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute grouped FlashVID accuracy/token Pareto reports offline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to config.output_dir, resolved relative to the config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_output = config.get("output_dir")
    output_dir = args.output_dir
    if output_dir is None:
        if not configured_output:
            raise ValueError("provide --output-dir or config.output_dir")
        output_dir = Path(str(configured_output))
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_sweep_report(config_path)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
