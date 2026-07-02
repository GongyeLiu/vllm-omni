import torch

from vllm_omni.diffusion.models.sd3_reward.checkpoint import (
    diffusers_lora_key_to_vllm_omni,
    summarize_lora_key_mapping,
)


def test_diffusers_self_attention_lora_keys_map_to_fused_qkv():
    key = "transformer_blocks.0.attn.to_q.lora_A.default.weight"

    assert diffusers_lora_key_to_vllm_omni(key) == "transformer_blocks.0.attn.to_qkv.lora_A.default.weight"


def test_diffusers_added_attention_lora_keys_map_to_fused_added_qkv():
    key = "transformer_blocks.0.attn.add_v_proj.lora_B.default.weight"

    assert diffusers_lora_key_to_vllm_omni(key) == "transformer_blocks.0.attn.add_kv_proj.lora_B.default.weight"


def test_summarize_lora_key_mapping_preserves_unpacked_output_keys():
    state = {
        "transformer_blocks.0.attn.to_out.0.lora_A.default.weight": torch.empty(1),
        "transformer_blocks.0.attn.to_add_out.lora_A.default.weight": torch.empty(1),
    }

    assert summarize_lora_key_mapping(state) == {
        "transformer_blocks.0.attn.to_out.0.lora_A.default.weight": "transformer_blocks.0.attn.to_out.0.lora_A.default.weight",
        "transformer_blocks.0.attn.to_add_out.lora_A.default.weight": "transformer_blocks.0.attn.to_add_out.lora_A.default.weight",
    }

