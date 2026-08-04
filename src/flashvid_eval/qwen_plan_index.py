from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SELECTION_PHASES = frozenset({"protocol_audit", "direct_dev", "agent_dev"})


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validated_run_plan(
    path: Path,
    config_sha256: str,
    *,
    allowed_phases: frozenset[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = _object(resolved, "run plan")
    claimed = str(payload.get("plan_sha256") or "")
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    )
    if claimed != actual:
        raise RuntimeError(f"run plan self-hash is invalid: {resolved}")
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError(f"run plan config hash differs from the experiment: {resolved}")
    phase = str(payload.get("phase") or "")
    if phase not in allowed_phases:
        raise ValueError(f"run plan phase {phase!r} is not allowed in this index: {resolved}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"run plan has no tasks: {resolved}")
    return payload, {
        "phase": phase,
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "plan_sha256": claimed,
    }


def build_selection_plan_index(
    *,
    config_path: Path,
    config_sha256: str,
    run_plan_paths: Iterable[Path],
    q4_think_smoke_rejection: Path | None = None,
) -> dict[str, Any]:
    references: list[dict[str, str]] = []
    seen_files: set[Path] = set()
    seen_tasks: set[tuple[str, str, str]] = set()
    observed_phases: set[str] = set()

    for path in run_plan_paths:
        resolved = path.resolve()
        if resolved in seen_files:
            raise ValueError(f"duplicate run plan supplied: {resolved}")
        seen_files.add(resolved)
        payload, reference = _validated_run_plan(
            resolved,
            config_sha256,
            allowed_phases=SELECTION_PHASES,
        )
        observed_phases.add(reference["phase"])
        for task in payload["tasks"]:
            if not isinstance(task, Mapping):
                raise ValueError(f"run plan task is not an object: {resolved}")
            identity = (
                reference["phase"],
                str(task.get("task_id") or ""),
                str(task.get("dataset") or ""),
            )
            if not identity[1] or not identity[2]:
                raise ValueError(f"run plan task identity is incomplete: {resolved}")
            if identity in seen_tasks:
                raise ValueError(f"duplicate logical Dev task across plans: {identity}")
            seen_tasks.add(identity)
        references.append(reference)

    missing_phases = sorted(SELECTION_PHASES - observed_phases)
    if missing_phases:
        raise ValueError(
            "selection plan index is incomplete; missing phases: " + ",".join(missing_phases)
        )

    rejection = None
    if q4_think_smoke_rejection is not None:
        payload, reference = _validated_run_plan(
            q4_think_smoke_rejection,
            config_sha256,
            allowed_phases=frozenset({"protocol_smoke"}),
        )
        if payload.get("model_filter") not in {None, "q4"}:
            raise ValueError("q4 thinking rejection plan does not target q4")
        rejection = reference

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen_dev_selection_plan_index",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_sha256,
        },
        "run_plans": sorted(
            references,
            key=lambda item: (item["phase"], item["path"]),
        ),
        "q4_think_smoke_rejection": rejection,
    }
    payload["index_sha256"] = canonical_sha256(payload)
    return payload


def write_frozen_index(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite changed frozen index: {path}")
        return file_sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return file_sha256(path)


def load_selection_plan_index(path: Path, config_sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = _object(resolved, "selection plan index")
    claimed = str(payload.get("index_sha256") or "")
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "index_sha256"}
    )
    if claimed != actual:
        raise RuntimeError(f"selection plan index self-hash is invalid: {resolved}")
    if payload.get("kind") != "qwen_dev_selection_plan_index":
        raise ValueError(f"unexpected selection plan index kind: {resolved}")
    config = payload.get("config")
    if not isinstance(config, Mapping) or config.get("sha256") != config_sha256:
        raise RuntimeError("selection plan index config hash differs from the experiment")

    paths: list[str] = []
    for item in payload.get("run_plans") or []:
        if not isinstance(item, Mapping):
            raise ValueError("selection plan index contains a malformed run-plan reference")
        plan_path = Path(str(item.get("path") or ""))
        plan_payload, reference = _validated_run_plan(
            plan_path,
            config_sha256,
            allowed_phases=SELECTION_PHASES,
        )
        del plan_payload
        for key in ("phase", "path", "sha256", "plan_sha256"):
            if item.get(key) != reference[key]:
                raise RuntimeError(f"selection plan reference changed ({key}): {plan_path}")
        paths.append(reference["path"])
    if not paths:
        raise ValueError("selection plan index contains no run plans")

    # Rebuild the canonical payload to re-check phase coverage and logical-task uniqueness.
    rejection = payload.get("q4_think_smoke_rejection")
    rebuilt = build_selection_plan_index(
        config_path=Path(str(config.get("path") or "")),
        config_sha256=config_sha256,
        run_plan_paths=[Path(item) for item in paths],
        q4_think_smoke_rejection=(
            Path(str(rejection.get("path"))) if isinstance(rejection, Mapping) else None
        ),
    )
    if rebuilt != payload:
        raise RuntimeError("selection plan index no longer matches its referenced artifacts")
    return payload
