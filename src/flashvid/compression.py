"""Training-free vision-side compression from FlashVID.

This module contains only the before-LLM path: DySeg, ADTS, TSTM, and
DPC-kNN. It intentionally contains no language-model layer pruning.

The implementation is adapted from Fanziyang-v/FlashVID (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlashVIDConfig:
    retention_ratio: float = 0.1
    alpha: float = 0.7
    temporal_threshold: float = 0.8
    segment_threshold: float = 0.9
    min_segments: int = 4
    attention_query_chunk: int = 128

    def __post_init__(self) -> None:
        if not 0.0 < self.retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not -1.0 <= self.temporal_threshold <= 1.0:
            raise ValueError("temporal_threshold must be in [-1, 1]")
        if not -1.0 <= self.segment_threshold <= 1.0:
            raise ValueError("segment_threshold must be in [-1, 1]")
        if self.min_segments < 1:
            raise ValueError("min_segments must be positive")
        if self.attention_query_chunk < 1:
            raise ValueError("attention_query_chunk must be positive")


@dataclass(frozen=True)
class CompressionResult:
    features: torch.Tensor
    anchor_indices: torch.Tensor
    original_tokens: int

    @property
    def retained_tokens(self) -> int:
        return int(self.features.shape[0])


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), p=2, dim=-1, eps=1e-6)


def dynamic_segments(
    frame_features: torch.Tensor,
    threshold: float = 0.9,
    min_segments: int = 4,
) -> torch.Tensor:
    """Return segment lengths using adjacent-frame cosine transitions."""
    if frame_features.ndim != 2:
        raise ValueError("frame_features must have shape [frames, dim]")
    num_frames = frame_features.shape[0]
    if num_frames == 0:
        raise ValueError("at least one frame is required")
    if num_frames == 1:
        return torch.ones(1, dtype=torch.long, device=frame_features.device)

    normalized = _normalize(frame_features)
    transitions = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    cuts = torch.where(transitions < threshold)[0]

    wanted = min(min_segments, num_frames)
    missing = wanted - (cuts.numel() + 1)
    if missing > 0:
        candidates = transitions.clone()
        candidates[candidates < threshold] = float("inf")
        extra = torch.topk(
            candidates,
            k=min(missing, candidates.numel()),
            largest=False,
        ).indices
        cuts = torch.cat((cuts, extra)).unique(sorted=True)

    boundaries = torch.cat(
        (
            torch.tensor([-1], device=cuts.device, dtype=torch.long),
            cuts.to(torch.long),
            torch.tensor([num_frames - 1], device=cuts.device, dtype=torch.long),
        )
    )
    return torch.diff(boundaries)


def pairwise_cosine_distances(features: torch.Tensor) -> torch.Tensor:
    normalized = _normalize(features)
    return 1.0 - torch.bmm(normalized, normalized.transpose(1, 2))


def attention_diversity_select(
    features: torch.Tensor,
    attention: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ADTS selection, matching the released FlashVID greedy rule."""
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, tokens, dim]")
    if attention.shape != features.shape[:2]:
        raise ValueError("attention must have shape [batch, tokens]")
    batch, tokens, dim = features.shape
    if not 0 <= count <= tokens:
        raise ValueError("count must be between zero and tokens")
    if count == 0:
        empty_idx = torch.empty(batch, 0, dtype=torch.long, device=features.device)
        return features[:, :0], empty_idx
    if tokens == 1:
        idx = torch.zeros(batch, 1, dtype=torch.long, device=features.device)
        return features, idx

    distances = pairwise_cosine_distances(features)
    calibrated = distances * (attention.float() * 1e6).unsqueeze(1)
    keep = torch.zeros(batch, count, dtype=torch.long, device=features.device)
    nearest_nonself = torch.topk(
        calibrated, k=2, dim=1, largest=False
    ).values[:, 1]
    keep[:, 0] = nearest_nonself.argmax(dim=-1)

    for index in range(1, count):
        selected_distances = torch.gather(
            calibrated,
            1,
            keep[:, :index].unsqueeze(-1).expand(-1, -1, tokens),
        )
        minimum = selected_distances.min(dim=1).values
        minimum.scatter_(1, keep[:, :index], -1)
        keep[:, index] = minimum.argmax(dim=-1)

    keep = keep.sort(dim=-1).values
    selected = torch.gather(features, 1, keep.unsqueeze(-1).expand(-1, -1, dim))
    return selected, keep


@torch.no_grad()
def dpc_knn(
    features: torch.Tensor,
    num_clusters: int,
    k: int = 7,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic DPC-kNN assignments and center indices."""
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, tokens, dim]")
    batch, tokens, dim = features.shape
    if not 1 <= num_clusters <= tokens:
        raise ValueError("num_clusters must be in [1, tokens]")
    k = min(max(1, k), tokens)

    distances = torch.cdist(features.float(), features.float()) / math.sqrt(dim)
    if valid_mask is not None:
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, tokens]")
        invalid = ~valid_mask
        sentinel = distances.amax().detach() + 1
        distances = distances.masked_fill(invalid.unsqueeze(1), sentinel)
    else:
        invalid = None

    nearest = torch.topk(distances, k=k, dim=-1, largest=False).values
    density = torch.mean(-(nearest**2), dim=-1).exp()
    if invalid is not None:
        density = density.masked_fill(invalid, 0)

    higher_density = density[:, None, :] > density[:, :, None]
    maximum = distances.flatten(1).amax(dim=-1).view(batch, 1, 1)
    distance_to_higher = torch.where(
        higher_density, distances, maximum
    ).amin(dim=-1)
    score = distance_to_higher * density
    centers = torch.topk(score, k=num_clusters, dim=-1).indices
    center_distances = torch.gather(
        distances, -1, centers.unsqueeze(1).expand(-1, tokens, -1)
    )
    assignments = center_distances.argmin(dim=-1)
    assignments.scatter_(
        -1,
        centers,
        torch.arange(num_clusters, device=features.device)
        .unsqueeze(0)
        .expand(batch, -1),
    )
    return assignments, centers


def temporal_tree_merge(
    features: torch.Tensor,
    available_mask: torch.Tensor,
    threshold: float = 0.8,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """TSTM adjacent-frame tree merging for tokens not selected by ADTS."""
    if features.ndim != 3:
        raise ValueError("features must have shape [frames, tokens, dim]")
    if available_mask.shape != features.shape[:2]:
        raise ValueError("available_mask must have shape [frames, tokens]")

    frames, tokens, dim = features.shape
    work = features.clone()
    counts = torch.ones(frames, tokens, device=features.device, dtype=torch.float32)
    if frames == 1:
        mask = available_mask[0]
        return [work[0, mask]], [torch.where(mask)[0]]

    normalized = _normalize(features)
    similarity = torch.bmm(normalized[1:], normalized[:-1].transpose(1, 2))
    similarity = similarity.masked_fill(~available_mask[1:].unsqueeze(-1), -1)
    similarity = similarity.masked_fill(~available_mask[:-1].unsqueeze(1), -1)
    maximum, anchors = similarity.max(dim=-1)
    maximum = F.pad(maximum, (0, 0, 1, 0), value=-1)
    anchors = F.pad(anchors, (0, 0, 1, 0), value=-1)
    merge_mask = (maximum > threshold) & available_mask
    retained_mask = available_mask & ~merge_mask

    for frame in range(frames - 1, -1, -1):
        retained = retained_mask[frame]
        if retained.any():
            work[frame, retained] /= counts[frame, retained].unsqueeze(-1).to(
                work.dtype
            )

        merging = merge_mask[frame]
        if frame > 0 and merging.any():
            target = anchors[frame, merging]
            accumulated = torch.zeros(
                tokens, dim, dtype=work.dtype, device=work.device
            )
            accumulated.scatter_add_(
                0, target.unsqueeze(-1).expand(-1, dim), work[frame, merging]
            )
            accumulated_counts = torch.bincount(
                target, weights=counts[frame, merging], minlength=tokens
            )
            work[frame - 1] += accumulated
            counts[frame - 1] += accumulated_counts

    output_features: list[torch.Tensor] = []
    output_indices: list[torch.Tensor] = []
    for frame in range(frames):
        output_features.append(work[frame, retained_mask[frame]])
        output_indices.append(torch.where(retained_mask[frame])[0])
    return output_features, output_indices


def _allocate_budget(capacities: Sequence[int], budget: int) -> list[int]:
    if budget < 0 or budget > sum(capacities):
        raise ValueError("budget exceeds capacity")
    if not capacities:
        return []
    if budget == 0:
        return [0] * len(capacities)

    total = sum(capacities)
    raw = [budget * capacity / total for capacity in capacities]
    allocated = [min(capacity, math.floor(value)) for capacity, value in zip(capacities, raw)]
    left = budget - sum(allocated)
    order = sorted(
        range(len(capacities)),
        key=lambda i: (raw[i] - allocated[i], capacities[i] - allocated[i], -i),
        reverse=True,
    )
    while left:
        progressed = False
        for index in order:
            if allocated[index] < capacities[index]:
                allocated[index] += 1
                left -= 1
                progressed = True
                if left == 0:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate token budget")
    return allocated


def _cluster_tokens(
    features: torch.Tensor,
    anchors: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count == 0:
        return features[:0], anchors[:0]
    if count >= features.shape[0]:
        return features, anchors
    assignments, centers = dpc_knn(
        features.unsqueeze(0), num_clusters=count, k=min(7, count)
    )
    assignments = assignments[0]
    centers = centers[0]
    output = torch.zeros(
        count, features.shape[-1], dtype=features.dtype, device=features.device
    )
    output.scatter_add_(
        0, assignments.unsqueeze(-1).expand_as(features), features
    )
    divisor = torch.bincount(assignments, minlength=count).unsqueeze(-1)
    output /= divisor.to(features.dtype)
    return output, anchors[centers]


class FlashVIDCompressor:
    def __init__(self, config: FlashVIDConfig):
        self.config = config

    @torch.no_grad()
    def __call__(
        self,
        video_features: torch.Tensor,
        attention: torch.Tensor,
        *,
        target_tokens: int | None = None,
    ) -> CompressionResult:
        if video_features.ndim != 3:
            raise ValueError("video_features must have shape [frames, tokens, dim]")
        if attention.shape != video_features.shape[:2]:
            raise ValueError("attention must have shape [frames, tokens]")
        frames, tokens_per_frame, dim = video_features.shape
        total_tokens = frames * tokens_per_frame
        if total_tokens == 0:
            raise ValueError("video_features must not be empty")

        if target_tokens is None:
            target_tokens = math.ceil(total_tokens * self.config.retention_ratio)
        if not 1 <= target_tokens <= total_tokens:
            raise ValueError("target_tokens must be in [1, total visual tokens]")
        flat = video_features.reshape(total_tokens, dim)
        if target_tokens == total_tokens:
            return CompressionResult(
                features=flat,
                anchor_indices=torch.arange(
                    total_tokens, device=video_features.device, dtype=torch.long
                ),
                original_tokens=total_tokens,
            )

        adts_budget = min(target_tokens, math.ceil(target_tokens * self.config.alpha))
        per_frame_adts = _allocate_budget(
            [tokens_per_frame] * frames, adts_budget
        )
        selected_features: list[torch.Tensor] = []
        selected_anchors: list[torch.Tensor] = []
        available = torch.ones(
            frames, tokens_per_frame, dtype=torch.bool, device=video_features.device
        )
        for frame, count in enumerate(per_frame_adts):
            chosen, local = attention_diversity_select(
                video_features[frame : frame + 1],
                attention[frame : frame + 1],
                count,
            )
            local = local[0]
            selected_features.append(chosen[0])
            selected_anchors.append(local + frame * tokens_per_frame)
            available[frame, local] = False

        segment_lengths = dynamic_segments(
            video_features.mean(dim=1),
            threshold=self.config.segment_threshold,
            min_segments=self.config.min_segments,
        )
        candidates_by_frame: list[torch.Tensor] = [
            video_features.new_empty((0, dim)) for _ in range(frames)
        ]
        anchors_by_frame: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.long, device=video_features.device)
            for _ in range(frames)
        ]
        offset = 0
        for length_tensor in segment_lengths:
            length = int(length_tensor.item())
            merged, local_anchors = temporal_tree_merge(
                video_features[offset : offset + length],
                available[offset : offset + length],
                threshold=self.config.temporal_threshold,
            )
            for local_frame, (frame_features, frame_anchors) in enumerate(
                zip(merged, local_anchors)
            ):
                frame = offset + local_frame
                candidates_by_frame[frame] = frame_features
                anchors_by_frame[frame] = (
                    frame_anchors + frame * tokens_per_frame
                )
            offset += length

        residual_budget = target_tokens - adts_budget
        capacities = [item.shape[0] for item in candidates_by_frame]
        from_tstm = min(residual_budget, sum(capacities))
        per_frame_context = _allocate_budget(capacities, from_tstm)
        output_features = selected_features.copy()
        output_anchors = selected_anchors.copy()
        for features, anchors, count in zip(
            candidates_by_frame, anchors_by_frame, per_frame_context
        ):
            clustered_features, clustered_anchors = _cluster_tokens(
                features, anchors, count
            )
            output_features.append(clustered_features)
            output_anchors.append(clustered_anchors)

        features = torch.cat(output_features)
        anchors = torch.cat(output_anchors)
        missing = target_tokens - features.shape[0]
        if missing:
            used = torch.zeros(
                total_tokens, dtype=torch.bool, device=video_features.device
            )
            used[anchors] = True
            scores = attention.flatten().masked_fill(used, float("-inf"))
            fill = torch.topk(scores, k=missing).indices
            features = torch.cat((features, flat[fill]))
            anchors = torch.cat((anchors, fill))

        order = anchors.argsort()
        features = features[order]
        anchors = anchors[order]
        if anchors.unique().numel() != target_tokens:
            raise RuntimeError("FlashVID produced duplicate anchor indices")
        return CompressionResult(
            features=features,
            anchor_indices=anchors,
            original_tokens=total_tokens,
        )

