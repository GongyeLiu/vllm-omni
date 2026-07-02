from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Iterable, get_args

import torch
import torch.nn as nn
from safetensors.torch import save_file
from vllm.config.lora import MaxLoRARanks
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.utils import replace_submodule
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

from vllm_omni.config.lora import LoRAConfig
from vllm_omni.diffusion.lora.utils import _match_target_modules, from_layer_diffusion
from vllm_omni.diffusion.models.sd3_reward.config import SD3RewardModelConfig

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.sd3_reward.model import SD3RewardBackbone

_VALID_MAX_RANKS: list[int] = sorted(get_args(MaxLoRARanks))


def _smallest_valid_max_rank(min_rank: int) -> int:
    allowed = [rank for rank in _VALID_MAX_RANKS if rank >= min_rank]
    if not allowed:
        raise ValueError(f"LoRA rank {min_rank} exceeds max allowed rank {max(_VALID_MAX_RANKS)}")
    return min(allowed)


def _packed_modules_list(module: nn.Module) -> list[str]:
    if isinstance(module, QKVParallelLinear):
        return ["q", "k", "v"]
    if isinstance(module, MergedColumnParallelLinear):
        return ["0", "1"]
    return []


def _should_exclude_module(module_name: str, config: SD3RewardModelConfig) -> bool:
    if config.use_text_features and config.text_head_idx and config.text_head_idx[-1] == config.num_transformer_layers:
        return False
    last = config.num_transformer_layers - 1
    excluded = {
        f"transformer_blocks.{last}.attn.add_kv_proj",
        f"transformer_blocks.{last}.attn.to_add_out",
    }
    return module_name in excluded


def _init_gaussian_lora(layer: BaseLayerWithLoRA, rank: int) -> None:
    lora_a_params = []
    lora_b_params = []
    for a_tensor, b_tensor in zip(layer.lora_a_stacked, layer.lora_b_stacked):
        nn.init.normal_(a_tensor[0, 0], std=1 / math.sqrt(rank))
        nn.init.zeros_(b_tensor[0, 0])
        lora_a_params.append(nn.Parameter(a_tensor))
        lora_b_params.append(nn.Parameter(b_tensor))
    layer.lora_a_stacked = nn.ParameterList(lora_a_params)
    layer.lora_b_stacked = nn.ParameterList(lora_b_params)
    layer._diffusion_lora_active_slices = (True,) * len(lora_a_params)


def apply_trainable_lora_backbone(
    backbone: SD3RewardBackbone,
    config: SD3RewardModelConfig,
    dtype: torch.dtype,
) -> dict[str, BaseLayerWithLoRA]:
    """Inject trainable vLLM-Omni diffusion LoRA layers into the reward backbone."""
    rank = _smallest_valid_max_rank(config.lora_config.r)
    lora_config = LoRAConfig(
        max_lora_rank=rank,
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype=dtype,
        fully_sharded_loras=False,
    )
    target_modules = ["to_qkv", "to_out.0", "add_kv_proj", "to_add_out"]
    replaced: dict[str, BaseLayerWithLoRA] = {}

    pending: list[tuple[str, nn.Module, list[str]]] = []
    for module_name, module in backbone.named_modules():
        if isinstance(module, BaseLayerWithLoRA) or "base_layer" in module_name.split("."):
            continue
        if _should_exclude_module(module_name, config):
            continue
        if not _match_target_modules(module_name, target_modules):
            continue
        pending.append((module_name, module, _packed_modules_list(module)))

    for module_name, module, packed_modules_list in pending:
        lora_layer = from_layer_diffusion(
            layer=module,
            max_loras=1,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
        )
        if lora_layer is module or not isinstance(lora_layer, BaseLayerWithLoRA):
            continue
        replace_submodule(backbone, module_name, lora_layer)
        _init_gaussian_lora(lora_layer, rank)
        replaced[module_name] = lora_layer

    if not replaced:
        raise RuntimeError("No LoRA layers were injected into SD3RewardBackbone")

    for name, param in backbone.named_parameters():
        param.requires_grad = "lora_" in name
    return replaced


def iter_trainable_lora_state_dict(backbone: SD3RewardBackbone) -> Iterable[tuple[str, torch.Tensor]]:
    """Export LoRA tensors using diffusers/PEFT-style key names for compatibility."""
    qkv_slices = ("to_q", "to_k", "to_v")
    add_slices = ("add_q_proj", "add_k_proj", "add_v_proj")
    for module_name, module in backbone.named_modules():
        if not isinstance(module, BaseLayerWithLoRA):
            continue
        rel = module_name
        for slice_idx, (a_tensor, b_tensor) in enumerate(zip(module.lora_a_stacked, module.lora_b_stacked)):
            a = a_tensor[0, 0].detach().cpu()
            b = b_tensor[0, 0].detach().cpu()
            if ".to_qkv" in rel:
                base = rel.replace(".to_qkv", f".{qkv_slices[slice_idx]}")
            elif ".add_kv_proj" in rel:
                base = rel.replace(".add_kv_proj", f".{add_slices[slice_idx]}")
            else:
                base = rel
            yield f"{base}.lora_A.default.weight", a
            yield f"{base}.lora_B.default.weight", b


def save_backbone_lora_peft(backbone: SD3RewardBackbone, save_dir: str, config: SD3RewardModelConfig) -> None:
    os.makedirs(save_dir, exist_ok=True)
    state = dict(iter_trainable_lora_state_dict(backbone))
    adapter_model_path = os.path.join(save_dir, "adapter_model.safetensors")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_file(state, tmp_path)
        shutil.copyfile(tmp_path, adapter_model_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    adapter_config = {
        "r": config.lora_config.r,
        "lora_alpha": config.lora_config.lora_alpha,
        "init_lora_weights": config.lora_config.init_lora_weights,
        "target_modules": [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ],
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
