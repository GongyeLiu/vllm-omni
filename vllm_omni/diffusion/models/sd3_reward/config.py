from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SD3RewardLoRAConfig:
    r: int = 64
    lora_alpha: int = 64
    init_lora_weights: str = "gaussian"


@dataclass
class SD3RewardModelConfig:
    model_type: str = "thurstone"
    num_transformer_layers: int = 12
    freeze_backbone: bool = True
    use_lora: bool = False
    use_ema: bool = False
    ema_decay: float = 0.995
    use_text_features: bool = True
    lora_config: SD3RewardLoRAConfig = field(default_factory=SD3RewardLoRAConfig)
    use_logistic: bool = False
    visual_head_idx: list[int] = field(default_factory=lambda: [4, 8, 12])
    text_head_idx: list[int] = field(default_factory=lambda: [4, 8, 12])
    use_t_embed: bool = True
    reward_head: dict[str, Any] = field(
        default_factory=lambda: {
            "use_proj_in": False,
            "width": -1,
            "out_dim": 1,
            "num_queries": 4,
            "num_attn_heads": 8,
            "dropout": 0.0,
        }
    )
    reward_checkpoint: str | None = None
    add_noise: bool = True


def _merge_dict_into_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def sd3_reward_config_from_dict(values: dict[str, Any] | None) -> SD3RewardModelConfig:
    config = SD3RewardModelConfig()
    if values:
        _merge_dict_into_dataclass(config, deepcopy(values))
    return config
