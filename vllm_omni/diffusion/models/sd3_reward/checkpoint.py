from __future__ import annotations

import os
from collections.abc import Mapping

import torch

from vllm_omni.diffusion.models.sd3_reward.lora_training import save_backbone_lora_peft


def load_reward_head(reward_model, checkpoint_dir: str, map_location="cpu") -> None:
    rm_head_path = os.path.join(checkpoint_dir, "rm_head.pt")
    if not os.path.exists(rm_head_path):
        raise FileNotFoundError(f"Reward head checkpoint not found: {rm_head_path}")
    reward_model.reward_head.load_state_dict(torch.load(rm_head_path, map_location=map_location))


def diffusers_lora_key_to_vllm_omni(key: str) -> str:
    """Map SD3 diffusers/PEFT LoRA keys to vLLM-Omni SD3 fused projection names."""
    replacements = {
        ".attn.to_q.": ".attn.to_qkv.",
        ".attn.to_k.": ".attn.to_qkv.",
        ".attn.to_v.": ".attn.to_qkv.",
        ".attn.add_q_proj.": ".attn.add_kv_proj.",
        ".attn.add_k_proj.": ".attn.add_kv_proj.",
        ".attn.add_v_proj.": ".attn.add_kv_proj.",
        ".attn.to_out.0.": ".attn.to_out.0.",
        ".attn.to_add_out.": ".attn.to_add_out.",
    }
    for src, dst in replacements.items():
        if src in key:
            return key.replace(src, dst)
    return key


def summarize_lora_key_mapping(state_dict: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {key: diffusers_lora_key_to_vllm_omni(key) for key in state_dict}

