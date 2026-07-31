import math

import pytest
import torch

from flashvid import (
    FlashVIDCompressor,
    FlashVIDConfig,
    attention_diversity_select,
    dpc_knn,
    dynamic_segments,
    temporal_tree_merge,
)


def test_config_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        FlashVIDConfig(retention_ratio=0)
    with pytest.raises(ValueError):
        FlashVIDConfig(retention_ratio=1.01)


def test_dynamic_segments_respects_minimum():
    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )
    lengths = dynamic_segments(features, threshold=0.9, min_segments=3)
    assert lengths.sum().item() == 4
    assert lengths.numel() == 3


def test_adts_matches_released_greedy_reference():
    features = torch.tensor(
        [[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]]
    )
    attention = torch.tensor([[0.2, 0.1, 0.6, 0.1]])
    _, indices = attention_diversity_select(features, attention, 2)

    normalized = features.float() / features.float().norm(
        p=2, dim=-1, keepdim=True
    )
    distances = 1 - torch.bmm(normalized, normalized.transpose(1, 2))
    distances *= (attention.float() * 1e6).unsqueeze(1)
    reference = torch.zeros(1, 2, dtype=torch.long)
    nearest = torch.topk(distances, k=2, dim=1, largest=False).values[:, 1]
    reference[:, 0] = nearest.argmax(dim=-1)
    selected = torch.gather(
        distances, 1, reference[:, :1].unsqueeze(-1).expand(-1, -1, 4)
    )
    minimum = selected.min(dim=1).values
    minimum.scatter_(1, reference[:, :1], -1)
    reference[:, 1] = minimum.argmax(dim=-1)
    assert torch.equal(indices, reference.sort(dim=-1).values)


def test_tstm_merges_identical_adjacent_tokens():
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    available = torch.ones(2, 2, dtype=torch.bool)
    merged, indices = temporal_tree_merge(features, available, threshold=0.8)
    assert [len(item) for item in merged] == [2, 0]
    assert torch.allclose(merged[0], features[0])
    assert torch.equal(indices[0], torch.tensor([0, 1]))


def test_dpc_knn_is_deterministic_and_centers_own_themselves():
    features = torch.tensor(
        [[[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]]]
    )
    assignment1, centers1 = dpc_knn(features, num_clusters=2, k=2)
    assignment2, centers2 = dpc_knn(features, num_clusters=2, k=2)
    assert torch.equal(assignment1, assignment2)
    assert torch.equal(centers1, centers2)
    assert torch.equal(
        assignment1[0, centers1[0]], torch.arange(2, dtype=torch.long)
    )


@pytest.mark.parametrize("ratio", [1.0, 0.5, 0.25, 0.1])
def test_compressor_hits_exact_ratio_budget(ratio):
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(8, 20, 16, generator=generator)
    attention = torch.rand(8, 20, generator=generator)
    compressor = FlashVIDCompressor(FlashVIDConfig(retention_ratio=ratio))
    result = compressor(features, attention)
    expected = math.ceil(features.shape[0] * features.shape[1] * ratio)
    assert result.retained_tokens == expected
    assert result.anchor_indices.unique().numel() == expected
    assert torch.equal(result.anchor_indices, result.anchor_indices.sort().values)


def test_ratio_one_is_exact_bypass():
    features = torch.randn(3, 5, 4)
    attention = torch.rand(3, 5)
    result = FlashVIDCompressor(FlashVIDConfig(retention_ratio=1.0))(
        features, attention
    )
    assert torch.equal(result.features, features.flatten(0, 1))
    assert torch.equal(result.anchor_indices, torch.arange(15))

