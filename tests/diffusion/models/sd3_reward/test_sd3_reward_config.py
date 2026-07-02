from vllm_omni.diffusion.models.sd3_reward.config import sd3_reward_config_from_dict


def test_sd3_reward_config_defaults_to_reward_head_training():
    config = sd3_reward_config_from_dict(None)

    assert config.num_transformer_layers == 12
    assert config.freeze_backbone is True
    assert config.use_lora is False
    assert config.visual_head_idx == [4, 8, 12]
    assert config.text_head_idx == [4, 8, 12]
    assert config.reward_head["num_queries"] == 4


def test_sd3_reward_config_overrides_nested_lora_config():
    config = sd3_reward_config_from_dict(
        {
            "freeze_backbone": False,
            "use_lora": True,
            "lora_config": {"r": 8, "lora_alpha": 16},
            "reward_head": {"num_queries": 2},
        }
    )

    assert config.freeze_backbone is False
    assert config.use_lora is True
    assert config.lora_config.r == 8
    assert config.lora_config.lora_alpha == 16
    assert config.reward_head["num_queries"] == 2
