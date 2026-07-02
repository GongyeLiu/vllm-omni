from __future__ import annotations

from contextlib import nullcontext

import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.models.sd3_reward.checkpoint import load_reward_head
from vllm_omni.diffusion.models.sd3_reward.config import sd3_reward_config_from_dict
from vllm_omni.diffusion.models.sd3_reward.model import SD3RewardModel
from vllm_omni.diffusion.request import OmniDiffusionRequest


def get_sd3_reward_post_process_func(od_config: OmniDiffusionConfig):
    return lambda scores: scores


class StableDiffusion3RewardPipeline(StableDiffusion3Pipeline):
    """Reward scoring pipeline for SD3.5-Medium latents.

    Request-level controls live in ``sampling_params.extra_args``:
    ``latents`` (required unless ``sampling_params.latents`` is set), ``u``,
    ``add_noise``, and optional ``reward_checkpoint``.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        cfg_dict = dict(od_config.additional_config.get("reward_model", {}))
        self.reward_config = sd3_reward_config_from_dict(cfg_dict)
        self.reward_model = SD3RewardModel(self.transformer, self.reward_config, dtype=od_config.dtype).to(self.device)
        checkpoint = self.reward_config.reward_checkpoint
        if checkpoint:
            load_reward_head(self.reward_model, checkpoint, map_location=self.device)

    def _reward_forward_context(self):
        vllm_config = getattr(self, "vllm_config", None) or getattr(self, "_sd3_reward_vllm_config", None)
        if vllm_config is None:
            return nullcontext()
        return set_forward_context(vllm_config=vllm_config, omni_diffusion_config=self.od_config)

    @staticmethod
    def get_timesteps_from_sigma(noise_scheduler, sigma_target, n_dim=4):
        sigmas = noise_scheduler.sigmas.to(sigma_target.device)
        idx = torch.argmin((sigmas[None, :] - sigma_target[:, None]).abs(), dim=1)
        timesteps = noise_scheduler.timesteps.to(sigma_target.device)[idx]
        sigma = sigmas[idx]
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma, timesteps

    def reward(
        self,
        prompts: str | list[str],
        latents: torch.Tensor,
        *,
        u: float = 0.1,
        add_noise: bool | None = None,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 256,
    ) -> torch.Tensor:
        with self._reward_forward_context():
            self.reward_model.eval()
            latents = latents.to(device=self.device, dtype=self.od_config.dtype)
            if prompt_embeds is None:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                    prompt=prompts,
                    prompt_2="",
                    prompt_3="",
                    max_sequence_length=max_sequence_length,
                )
            else:
                prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.od_config.dtype)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.od_config.dtype)

            u_tensor = torch.full((latents.shape[0],), float(u), device=self.device)
            sigmas, timesteps = self.get_timesteps_from_sigma(self.scheduler, u_tensor, n_dim=latents.dim())
            should_add_noise = self.reward_config.add_noise if add_noise is None else add_noise
            if should_add_noise:
                latent_sigmas = sigmas.to(dtype=self.od_config.dtype)
                latents = ((1.0 - latent_sigmas) * latents + latent_sigmas * torch.randn_like(latents)).to(
                    dtype=self.od_config.dtype
                )

            with torch.no_grad():
                return self.reward_model(
                    latents=latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timesteps=timesteps,
                )

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra_args = dict(req.sampling_params.extra_args or {})
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]
        latents = extra_args.pop("latents", None)
        if latents is None:
            latents = req.sampling_params.latents
        if latents is None:
            raise ValueError("StableDiffusion3RewardPipeline requires latents in sampling_params.latents or extra_args")
        checkpoint = extra_args.pop("reward_checkpoint", None)
        if checkpoint:
            load_reward_head(self.reward_model, checkpoint, map_location=self.device)
        scores = self.reward(
            prompts=prompt,
            latents=latents,
            u=float(extra_args.pop("u", 0.1)),
            add_noise=extra_args.pop("add_noise", None),
            max_sequence_length=req.sampling_params.max_sequence_length or 256,
        )
        return DiffusionOutput(output=scores, custom_output={"scores": scores})
