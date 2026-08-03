"""Qwen-only long-video agent strategies.

The package intentionally depends only on Qwen chat endpoints and the
official EVA frame decoder. It is not wired into the legacy evaluator yet.
"""

from .core import (
    AgentConfig,
    AgentStrategy,
    AgentTrace,
    BaseQwenAgent,
    DuplicateFrameRequestError,
    EvidenceEntry,
    FrameObservation,
    FrameRequest,
    FrameSession,
    FrameTool,
    InferenceProtocol,
    RequestTrace,
    ToolStep,
    parse_answer_json,
    parse_frame_tool_calls,
)
from .strategies import (
    EvaCleanStrategy,
    HierarchicalSearchStrategy,
    IndependentArbitrationStrategy,
    MultiClueMemoryStrategy,
    StoryboardZoomStrategy,
    build_strategy,
)

__all__ = [
    "AgentConfig",
    "AgentStrategy",
    "AgentTrace",
    "BaseQwenAgent",
    "DuplicateFrameRequestError",
    "EvaCleanStrategy",
    "EvidenceEntry",
    "FrameObservation",
    "FrameRequest",
    "FrameSession",
    "FrameTool",
    "HierarchicalSearchStrategy",
    "IndependentArbitrationStrategy",
    "InferenceProtocol",
    "MultiClueMemoryStrategy",
    "RequestTrace",
    "StoryboardZoomStrategy",
    "ToolStep",
    "build_strategy",
    "parse_answer_json",
    "parse_frame_tool_calls",
]
