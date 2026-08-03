from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from flashvid_eval.flashvid_budget import (
    ALLOWED_RETENTION_RATIOS,
    BudgetEndpoint,
    BudgetEndpointPool,
    flashvid_token_counts,
    pack_frames_to_mp4,
    perception_cache_key,
)


def _endpoint_payload(key: str = "endpoints") -> dict:
    return {
        key: [
            {
                "retention_ratio": ratio,
                "base_url": f"http://127.0.0.1:{8100 + index}/v1/",
                "model": "Qwen3.5-4B",
                "max_concurrency": 8,
            }
            for index, ratio in enumerate(ALLOWED_RETENTION_RATIOS, start=1)
        ]
    }


@pytest.mark.parametrize("key", ["endpoints", "perception"])
def test_endpoint_pool_loads_both_supported_config_shapes(tmp_path, key):
    payload = _endpoint_payload(key)
    if key == "perception":
        payload["controller"] = {
            "base_url": "http://127.0.0.1:8200/v1",
            "model": "Qwen3.5-9B",
        }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    pool = BudgetEndpointPool.load(path)

    assert tuple(item.retention_ratio for item in pool.endpoints) == (
        ALLOWED_RETENTION_RATIOS
    )
    assert pool.choose("0.10").base_url == "http://127.0.0.1:8101/v1"
    assert pool.choose(1).model == "Qwen3.5-4B"


def test_endpoint_pool_rejects_missing_duplicate_and_unknown_ratios():
    endpoints = [
        BudgetEndpoint(ratio, f"http://localhost:{index}", "model")
        for index, ratio in enumerate(ALLOWED_RETENTION_RATIOS[:-1], start=1)
    ]
    with pytest.raises(ValueError, match="missing=1.00"):
        BudgetEndpointPool(endpoints)
    with pytest.raises(ValueError, match="duplicate endpoint"):
        BudgetEndpointPool([*endpoints, endpoints[0]])
    with pytest.raises(ValueError, match="unsupported retention ratio"):
        BudgetEndpoint(0.75, "http://localhost:8100", "model")


def test_endpoint_backend_revision_is_validated():
    endpoint = BudgetEndpoint(
        1.0,
        "http://localhost:8104",
        "model",
        backend_revision="native_bypass_v1",
    )
    assert endpoint.backend_revision == "native_bypass_v1"
    with pytest.raises(ValueError, match="backend_revision"):
        BudgetEndpoint(
            1.0,
            "http://localhost:8104",
            "model",
            backend_revision=" ",
        )


def test_endpoint_pool_rejects_ambiguous_config(tmp_path):
    payload = _endpoint_payload("endpoints")
    payload["perception"] = payload["endpoints"]
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        BudgetEndpointPool.load(path)


def test_perception_cache_key_is_canonical_and_sensitive(tmp_path):
    video = tmp_path / "video.mp4"
    common = {
        "video": video,
        "start_time": 12.0,
        "end_time": 18.5,
        "nframes": 8,
        "resize": 0.5,
        "model": "Qwen3.5-4B",
        "prompt_hash": hashlib.sha256(b"prompt").hexdigest(),
        "evidence_request": "Observe the action.",
    }

    first = perception_cache_key(retention_ratio=0.1, **common)
    equivalent = perception_cache_key(retention_ratio="0.10", **common)
    changed = perception_cache_key(retention_ratio=0.25, **common)

    assert first == equivalent
    assert len(first) == 64
    assert first != changed


def test_perception_cache_key_isolates_backend_revisions(tmp_path):
    common = {
        "video": tmp_path / "video.mp4",
        "start_time": 12.0,
        "end_time": 18.5,
        "nframes": 8,
        "resize": 0.5,
        "retention_ratio": 1.0,
        "model": "Qwen3.5-4B-FlashVID-r100",
        "prompt_hash": hashlib.sha256(b"prompt").hexdigest(),
    }

    plugin = perception_cache_key(
        backend_revision="flashvid_arch_v1", **common
    )
    bypass = perception_cache_key(
        backend_revision="native_bypass_v1", **common
    )

    assert plugin != bypass


def test_flashvid_token_counts_matches_deployed_floor_and_frame_minimum():
    metadata = {"height": 280, "width": 560}

    raw, retained, effective = flashvid_token_counts(
        metadata, nframes=8, resize=0.5, ratio=0.25
    )
    assert raw == 8 * 5 * 10
    assert retained == 100
    assert effective == 0.25

    raw, retained, effective = flashvid_token_counts(
        metadata, nframes=3, resize=0.5, ratio=0.10
    )
    assert raw == 150
    assert retained == 50
    assert effective == pytest.approx(1 / 3)


def test_flashvid_token_counts_ratio_one_is_exact_bypass():
    assert flashvid_token_counts(
        {"height": 224, "width": 224}, 4, 1.0, 1.0
    ) == (256, 256, 1.0)


def test_pack_frames_preserves_order_and_writes_sidecar(tmp_path, monkeypatch):
    frames = []
    for name in ("frame_2.png", "frame_0.png", "frame_1.png"):
        path = tmp_path / name
        path.write_bytes(b"image")
        frames.append(path)
    output = tmp_path / "media" / "clip.mp4"
    sidecar = tmp_path / "media" / "clip.timestamps.json"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["manifest"] = kwargs["input"]
        Path(command[-1]).write_bytes(b"encoded-mp4")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = pack_frames_to_mp4(
        frames, [3.0, 4.5, 9.0], output, sidecar
    )

    assert output.read_bytes() == b"encoded-mp4"
    assert payload == json.loads(sidecar.read_text(encoding="utf-8"))
    assert [item["timestamp_s"] for item in payload["frames"]] == [3.0, 4.5, 9.0]
    manifest = captured["manifest"]
    assert manifest.index("frame_2.png") < manifest.index("frame_0.png")
    assert manifest.index("frame_0.png") < manifest.index("frame_1.png")
    assert captured["command"][captured["command"].index("-frames:v") + 1] == "3"


def test_pack_frames_validates_timestamps_before_invoking_ffmpeg(
    tmp_path, monkeypatch
):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"image")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("ffmpeg should not be called")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="nondecreasing"):
        pack_frames_to_mp4(
            [frame, frame],
            [2.0, 1.0],
            tmp_path / "clip.mp4",
            tmp_path / "clip.json",
        )
