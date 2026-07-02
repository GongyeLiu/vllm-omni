from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from vllm_omni.diffusion.models.sd3.sd3_transformer import SD3Transformer2DModel
from vllm_omni.diffusion.models.sd3_reward.config import SD3RewardModelConfig
from vllm_omni.diffusion.models.sd3_reward.lora_training import apply_trainable_lora_backbone
from vllm_omni.diffusion.models.sd3_reward.reward_head import RewardHead


def _as_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


class SD3RewardBackbone(nn.Module):
    """Feature-tap wrapper around vLLM-Omni's SD3 transformer."""

    def __init__(self, transformer: SD3Transformer2DModel, config: SD3RewardModelConfig):
        super().__init__()
        if config.num_transformer_layers > len(transformer.transformer_blocks):
            raise ValueError(
                "num_transformer_layers exceeds the loaded SD3 transformer depth: "
                f"{config.num_transformer_layers} > {len(transformer.transformer_blocks)}"
            )
        self.pos_embed = transformer.pos_embed
        self.time_text_embed = transformer.time_text_embed
        self.context_embedder = transformer.context_embedder
        self.transformer_blocks = nn.ModuleList(transformer.transformer_blocks[: config.num_transformer_layers])
        self.visual_head_idx = list(config.visual_head_idx)
        self.text_head_idx = list(config.text_head_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.LongTensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = _as_tensor(self.context_embedder(encoder_hidden_states))

        visual_features = [hidden_states] if self.visual_head_idx and self.visual_head_idx[0] == 0 else []
        text_features = [encoder_hidden_states] if self.text_head_idx and self.text_head_idx[0] == 0 else []

        for block_index, block in enumerate(self.transformer_blocks, start=1):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )
            if block_index in self.visual_head_idx:
                visual_features.append(hidden_states)
            if block_index in self.text_head_idx:
                if encoder_hidden_states is None:
                    raise ValueError(f"text_head_idx={block_index} points to a context-pre-only SD3 block")
                text_features.append(encoder_hidden_states)

        return temb, visual_features, text_features


class SD3RewardModel(nn.Module):
    """SD3.5-Medium latent reward model using vLLM-Omni SD3 components."""

    def __init__(self, transformer: SD3Transformer2DModel, config: SD3RewardModelConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.backbone = SD3RewardBackbone(transformer, config)
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
        elif config.use_lora:
            # LoRA is injected after base SD3 weights are loaded. Wrapping the
            # vLLM-Omni linear layers here would change parameter names before
            # the SD3 loader has a chance to map diffusers weights.
            self.backbone.requires_grad_(False)
        else:
            self.backbone.requires_grad_(True)

        backbone_dim = transformer.inner_dim
        self.reward_head = RewardHead(
            token_dim=backbone_dim,
            n_visual_heads=len(config.visual_head_idx),
            n_text_heads=len(config.text_head_idx),
            patch_size=transformer.patch_size,
            t_embed_dim=backbone_dim,
            use_t_embed=config.use_t_embed,
            **config.reward_head,
        ).to(dtype=dtype)

        self.use_logistic = config.use_logistic
        if self.use_logistic:
            self.eta1 = 2.0
            self.eta2 = -2.0
            self.eta3 = nn.Parameter(torch.tensor(0.0))
            self.eta4 = nn.Parameter(torch.tensor(0.15))

    def _logistic(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_logistic:
            return x
        exp_pow = -1 * (x - self.eta3) / (torch.abs(self.eta4) + 1e-6)
        return (self.eta1 - self.eta2) / (1 + torch.exp(exp_pow)) + self.eta2

    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: Optional[torch.Tensor],
        timesteps: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, _, height, width = latents.shape
        if pooled_projections is None:
            pooled_projections = torch.zeros(
                (bsz, self.backbone.time_text_embed.text_embedder.linear_1.in_features),
                device=latents.device,
                dtype=latents.dtype,
            )
        temb, visual_features, text_features = self.backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps,
        )
        reward = self.reward_head(
            visual_features=visual_features,
            text_features=text_features,
            t_embed=temb,
            hw=(height, width),
        )
        return self._logistic(reward)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def enable_trainable_lora(self, dtype: torch.dtype) -> None:
        if not self.config.use_lora or self.config.freeze_backbone:
            return
        apply_trainable_lora_backbone(self.backbone, self.config, dtype=dtype)
