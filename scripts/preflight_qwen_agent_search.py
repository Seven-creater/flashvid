from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_SPLIT_COUNTS = {"train": 200, "dev": 50, "final": 100}
FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:clip|open_clip|paddleocr|whisper|ultralytics|"
    r"sentence_transformers)(?:\.|\s|$)",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("sample_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate sample_id in {path}")
    return rows


def audit(config_path: Path, source_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "datasets": {},
        "forbidden_imports": [],
        "passed": True,
    }
    for dataset, dataset_config in config["datasets"].items():
        split_rows: dict[str, list[dict[str, Any]]] = {}
        split_report: dict[str, Any] = {}
        for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
            spec = dataset_config[split]
            path = Path(spec["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = sha256(path)
            if actual_hash.lower() != str(spec["sha256"]).lower():
                raise RuntimeError(
                    f"{dataset} {split} manifest hash mismatch: {actual_hash}"
                )
            rows = load_manifest(path)
            if len(rows) != expected_count:
                raise RuntimeError(
                    f"{dataset} {split} has {len(rows)} rows; expected {expected_count}"
                )
            split_rows[split] = rows
            split_report[split] = {
                "path": str(path),
                "sha256": actual_hash,
                "samples": len(rows),
                "videos": len({str(row.get("video")) for row in rows}),
            }
        for left, right in (("train", "dev"), ("train", "final"), ("dev", "final")):
            left_ids = {str(row["sample_id"]) for row in split_rows[left]}
            right_ids = {str(row["sample_id"]) for row in split_rows[right]}
            if left_ids & right_ids:
                raise RuntimeError(f"{dataset} sample overlap: {left}/{right}")
            left_videos = {str(row["video"]) for row in split_rows[left]}
            right_videos = {str(row["video"]) for row in split_rows[right]}
            if left_videos & right_videos:
                raise RuntimeError(f"{dataset} video overlap: {left}/{right}")
        report["datasets"][dataset] = split_report

    source_paths = [source_root / "src" / "flashvid_eval" / "qwen_agents"]
    source_paths.extend(
        path
        for path in (
            source_root / "src" / "flashvid_eval" / "qwen_evaluation.py",
            source_root / "src" / "flashvid_eval" / "qwen_trajectories.py",
            source_root / "src" / "flashvid_eval" / "qwen_sft.py",
        )
        if path.exists()
    )
    violations: list[dict[str, str]] = []
    for root in source_paths:
        if not root.exists():
            continue
        paths = root.rglob("*.py") if root.is_dir() else (root,)
        for path in paths:
            match = FORBIDDEN_IMPORT.search(path.read_text(encoding="utf-8"))
            if match:
                violations.append({"path": str(path), "import": match.group(0).strip()})
    report["forbidden_imports"] = violations
    report["passed"] = not violations
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Qwen Agent search inputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.config, args.source_root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
