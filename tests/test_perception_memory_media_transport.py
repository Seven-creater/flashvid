from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.perception_memory_eva import PerceptionMemoryEvaEvaluator
from scripts.evaluate_mcq import _local_media_transport
from scripts.validate_perception_memory_config import validate_config


CONFIG_PATH = Path("configs/experiments/perception_memory_eva_sft.json")


class _FrameTool:
    max_frames_per_call = 128
    selector_identity = {"kind": "test_frame_tool"}


def test_local_media_paths_are_explicit_and_backend_scoped() -> None:
    assert _local_media_transport("perception_memory_eva", False) == "file_url"
    assert _local_media_transport("perception_memory_eva", True) == "path"
    assert _local_media_transport("direct", False) == "file_url"
    with pytest.raises(ValueError, match="only valid with perception_memory_eva"):
        _local_media_transport("fast_hybrid_eva", True)


def test_local_media_transport_is_audited_and_changes_fingerprint(
    tmp_path: Path,
) -> None:
    common = {
        "model": "Qwen3.5-9B",
        "video_root": tmp_path,
        "frame_root": tmp_path / "frames",
        "frame_tool": _FrameTool(),
    }
    file_url = PerceptionMemoryEvaEvaluator(
        OpenAICompatibleClient("http://localhost:8200/v1"),
        **common,
    )
    path = PerceptionMemoryEvaEvaluator(
        OpenAICompatibleClient(
            "http://localhost:8200/v1", local_file_urls_as_paths=True
        ),
        **common,
    )

    assert file_url.static_audit_fields()["local_media_transport"] == "file_url"
    assert path.static_audit_fields()["local_media_transport"] == "path"
    assert file_url.run_fingerprint() != path.run_fingerprint()


def test_frozen_config_requires_transformers_path_transport() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert validate_config(config) == []

    config["runtime"]["local_media_transport"] = "file_url"
    assert any(
        "explicit path local-media transport" in error
        for error in validate_config(config)
    )
