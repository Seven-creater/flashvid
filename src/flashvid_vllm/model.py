"""Out-of-tree vLLM model for Qwen3.5 with FlashVID compression."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from flashvid import FlashVIDCompressor, FlashVIDConfig
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config

logger = logging.getLogger(__name__)


def _ratio_from_environment() -> float:
    raw = os.environ.get("FLASHVID_VISION_RETENTION_RATIO", "0.1")
    ratio = float(raw)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("FLASHVID_VISION_RETENTION_RATIO must be in (0, 1]")
    return ratio


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class FlashVIDQwen3_5ForConditionalGeneration(
    Qwen3_5ForConditionalGeneration
):
    """Qwen3.5 with before-LLM ADTS + TSTM vision token compression."""

    supports_multimodal_pruning = True

    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vision_retention_ratio = _ratio_from_environment()
        expected_pruning = 1.0 - self.vision_retention_ratio
        configured_pruning = (
            vllm_config.model_config.multimodal_config.video_pruning_rate or 0.0
        )
        if abs(configured_pruning - expected_pruning) > 1e-6:
            raise ValueError(
                "Use flashvid-serve so --vision-retention-ratio and vLLM "
                "placeholder pruning remain synchronized."
            )

        self.video_pruning_rate = configured_pruning
        self.is_multimodal_pruning_enabled = self.vision_retention_ratio < 1.0
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        deepstack_indexes = getattr(
            self.config.vision_config, "deepstack_visual_indexes", []
        )
        self.use_deepstack = bool(deepstack_indexes)
        self.deepstack_num_level = len(deepstack_indexes)
        self.visual_dim = self.config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        self._flashvid = FlashVIDCompressor(
            FlashVIDConfig(retention_ratio=self.vision_retention_ratio)
        )
        self._flashvid_last_qkv: torch.Tensor | None = None
        self._flashvid_attention: tuple[torch.Tensor, ...] | None = None
        logger.info(
            "FlashVID vision compression enabled: retention_ratio=%.4f",
            self.vision_retention_ratio,
        )

    def _capture_last_qkv(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        projected = output[0] if isinstance(output, tuple) else output
        self._flashvid_last_qkv = projected.detach()

    @torch.no_grad()
    def _attention_from_last_qkv(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        projected = self._flashvid_last_qkv
        self._flashvid_last_qkv = None
        if projected is None:
            raise RuntimeError("last vision-layer QKV was not captured")

        attention_module = self.visual.blocks[-1].attn
        sequence, batch, _ = projected.shape
        if batch != 1:
            raise RuntimeError("Qwen3.5 vision encoder batch dimension must be 1")
        heads = attention_module.num_attention_heads_per_partition
        head_dim = attention_module.hidden_size_per_attention_head
        qkv = projected.view(sequence, batch, 3, heads, head_dim).permute(
            1, 0, 2, 3, 4
        )
        qk = qkv[:, :, :2].permute(2, 0, 1, 3, 4).reshape(
            2 * batch, sequence, heads, head_dim
        )
        cosine, sine = self.visual.rot_pos_emb(grid_thw.tolist())
        qk = attention_module.apply_rotary_emb(
            qk.contiguous(), cosine, sine
        ).view(2, batch, sequence, heads, head_dim)
        query, key = qk.unbind(dim=0)
        query = query[0]
        key = key[0]

        outputs: list[torch.Tensor] = []
        offset = 0
        merge_unit = self.visual.spatial_merge_unit
        chunk = self._flashvid.config.attention_query_chunk
        scale = head_dim**-0.5
        for temporal, height, width in grid_thw.tolist():
            patches_per_frame = height * width
            video_length = temporal * patches_per_frame
            video_query = (
                query[offset : offset + video_length]
                .view(temporal, patches_per_frame, heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            video_key = (
                key[offset : offset + video_length]
                .view(temporal, patches_per_frame, heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            received = torch.zeros(
                temporal,
                patches_per_frame,
                device=query.device,
                dtype=torch.float32,
            )
            transposed_key = video_key.float().transpose(-1, -2)
            for start in range(0, patches_per_frame, chunk):
                logits = torch.matmul(
                    video_query[:, :, start : start + chunk].float(),
                    transposed_key,
                )
                probabilities = torch.softmax(logits * scale, dim=-1)
                received += probabilities.sum(dim=(1, 2))
            received /= heads * patches_per_frame
            outputs.append(
                received.view(temporal, -1, merge_unit).mean(dim=-1)
            )
            offset += video_length
        if offset != sequence:
            raise RuntimeError("QKV length does not match video grids")
        return tuple(outputs)

    def _process_video_input(self, video_input):
        hook = None
        if self.is_multimodal_pruning_enabled:
            if video_input["type"] == "video_embeds":
                raise ValueError(
                    "FlashVID requires pixel_values_videos to compute last-layer "
                    "vision attention; precomputed video_embeds are unsupported."
                )
            hook = self.visual.blocks[-1].attn.qkv.register_forward_hook(
                self._capture_last_qkv
            )
        try:
            embeddings = super()._process_video_input(video_input)
        finally:
            if hook is not None:
                hook.remove()
        if self.is_multimodal_pruning_enabled:
            self._flashvid_attention = self._attention_from_last_qkv(
                video_input["video_grid_thw"]
            )
        return embeddings

    def _postprocess_video_embeds_evs(
        self,
        video_embeds_split: tuple[torch.Tensor, ...],
        video_input,
    ) -> tuple[torch.Tensor, ...]:
        if not self.is_multimodal_pruning_enabled:
            return video_embeds_split
        attentions = self._flashvid_attention
        self._flashvid_attention = None
        if attentions is None or len(attentions) != len(video_embeds_split):
            raise RuntimeError("missing FlashVID attention for video batch")

        grid_list = video_input["video_grid_thw"].tolist()
        merge_size = self.visual.spatial_merge_size
        output: list[torch.Tensor] = []
        for index, (embedding, attention, size) in enumerate(
            zip(video_embeds_split, attentions, grid_list)
        ):
            temporal, height, width = size
            tokens_per_frame = (height // merge_size) * (width // merge_size)
            target = max(
                tokens_per_frame,
                int(
                    temporal
                    * tokens_per_frame
                    * (1.0 - self.video_pruning_rate)
                ),
            )
            started = time.perf_counter()
            result = self._flashvid(
                embedding.view(temporal, tokens_per_frame, -1),
                attention,
                target_tokens=target,
            )
            retention_mask = torch.zeros(
                temporal * tokens_per_frame,
                dtype=torch.bool,
                device=embedding.device,
            )
            retention_mask[result.anchor_indices] = True
            per_frame = (
                retention_mask.view(temporal, tokens_per_frame)
                .sum(dim=-1)
                .long()
                .tolist()
            )
            final = self._create_final_video_embeddings(
                video_embeddings=result.features,
                num_tokens_per_frame=per_frame,
                timestamps=video_input.timestamps[index],
                video_grid_thw=size,
                retention_mask=retention_mask,
            )
            output.append(final)
            logger.info(
                "FlashVID compressed video %d: %d -> %d tokens (%.2f ms)",
                index,
                result.original_tokens,
                result.retained_tokens,
                (time.perf_counter() - started) * 1000,
            )
        return tuple(output)

    def recompute_mrope_positions(
        self,
        input_ids,
        multimodal_embeddings,
        mrope_positions,
        num_computed_tokens,
    ):
        return self._recompute_mrope_positions(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            mrope_positions=mrope_positions,
            num_computed_tokens=num_computed_tokens,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )
