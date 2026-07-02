from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
from contextlib import nullcontext
from typing import Any

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.sd3_reward.data.bucket_dataset import create_bucket_dataloader
from vllm_omni.diffusion.models.sd3_reward.data.simple_dataset import create_simple_dataloader
from vllm_omni.diffusion.models.sd3_reward.checkpoint import save_backbone_lora_peft

logger = get_logger(__name__)


class RewardEMAManager:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.ema_state: dict[str, torch.Tensor] = {}
        self.backup_state: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.detach().clone()

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_state:
                self.ema_state[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def swap_in(self, model: torch.nn.Module):
        self.backup_state.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_state:
                self.backup_state[name] = param.detach().clone()
                param.data.copy_(self.ema_state[name].data)

    def swap_out(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup_state:
                param.data.copy_(self.backup_state[name].data)
        self.backup_state.clear()

    def save(self, path: str):
        torch.save(self.ema_state, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD3.5-Medium diffusion reward head on vLLM-Omni")
    parser.add_argument("--config", required=True, help="Path to a reward training YAML config")
    return parser.parse_args()


def load_config(path: str):
    with open(path) as f:
        return OmegaConf.create(yaml.safe_load(f))


def build_pipeline(config, accelerator: Accelerator, dtype: torch.dtype):
    od_config = OmniDiffusionConfig.from_kwargs(
        model=config.model.backbone_model_id,
        model_class_name="StableDiffusion3RewardPipeline",
        dtype=dtype,
        output_type="latent",
        additional_config={"reward_model": OmegaConf.to_container(config.model, resolve=True)},
    )
    od_config.enrich_config()
    vllm_config = create_standalone_vllm_config(accelerator.device, od_config, accelerator.num_processes)
    with diffusion_forward_context(vllm_config, od_config):
        ensure_diffusion_model_parallel(od_config, accelerator)
        loader = DiffusersPipelineLoader(LoadConfig(load_format="auto"), od_config)
        pipeline = loader.load_model(load_device=accelerator.device.type, device=accelerator.device)
        if config.model.use_lora and not config.model.freeze_backbone:
            pipeline.reward_model.enable_trainable_lora(dtype)
    pipeline._sd3_reward_vllm_config = vllm_config
    pipeline.to(accelerator.device)
    return pipeline


def create_standalone_vllm_config(device: torch.device, od_config: OmniDiffusionConfig, data_parallel_size: int):
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    from vllm_omni.diffusion.worker.diffusion_worker import (
        _make_diffusion_vllm_model_config,
        _resolve_ir_op_priority,
    )

    config_kwargs: dict[str, Any] = {
        "compilation_config": CompilationConfig(),
        "device_config": DeviceConfig(device=device),
    }
    if od_config.additional_config:
        config_kwargs["additional_config"] = od_config.additional_config
    vllm_config = VllmConfig(**config_kwargs)
    parallel_config = od_config.parallel_config
    vllm_config.parallel_config.tensor_parallel_size = parallel_config.tensor_parallel_size
    vllm_config.parallel_config.data_parallel_size = max(int(data_parallel_size), 1)
    vllm_config.parallel_config.enable_expert_parallel = parallel_config.enable_expert_parallel
    vllm_config.profiler_config = od_config.profiler_config
    vllm_config.model_config = _make_diffusion_vllm_model_config(od_config)  # type: ignore[assignment]
    vllm_config.quant_config = od_config.quantization_config
    vllm_config.kernel_config.ir_op_priority = _resolve_ir_op_priority(od_config, vllm_config)
    return vllm_config

def diffusion_forward_context(vllm_config, od_config: OmniDiffusionConfig):
    from vllm_omni.diffusion.forward_context import set_forward_context

    if vllm_config is None:
        return nullcontext()
    return set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config)


def ensure_diffusion_model_parallel(od_config: OmniDiffusionConfig, accelerator: Accelerator):
    """Initialize the vLLM-Omni groups required by SD3 parallel linear layers."""
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    init_distributed_environment(
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
        distributed_init_method=f"tcp://127.0.0.1:{od_config.master_port}",
        local_rank=accelerator.local_process_index,
    )
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            data_parallel_size=accelerator.num_processes,
            cfg_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            fully_shard_degree=1,
            hsdp_replicate_size=1,
        )


def cleanup_diffusion_distributed():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    from vllm_omni.diffusion.distributed.parallel_state import destroy_distributed_env

    destroy_distributed_env()


def timestep_sampling(weighting_scheme, weighting_scheme_param, batch_size, device):
    if weighting_scheme == "uniform":
        return torch.rand(size=(batch_size,), device=device)
    if weighting_scheme == "constant":
        return torch.full(size=(batch_size,), fill_value=float(weighting_scheme_param), device=device)
    if weighting_scheme == "power":
        return torch.rand(size=(batch_size,), device=device) ** float(weighting_scheme_param)
    if weighting_scheme == "logit_normal":
        mean, std = [float(x) for x in str(weighting_scheme_param).split("_")]
        return torch.sigmoid(torch.normal(mean=mean, std=std, size=(batch_size,), device=device))
    if weighting_scheme == "mode":
        mode_scale = float(weighting_scheme_param)
        u = torch.rand(size=(batch_size,), device=device)
        return 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    raise ValueError(f"Unknown timestep weighting scheme: {weighting_scheme}")


def thurstone_loss(score_chosen, score_reject, sigma, eps=1e-6):
    sigma = sigma.reshape(-1, 1).repeat(1, score_chosen.shape[1])
    var = 2.0 * sigma**2 + 0.05
    p = torch.distributions.Normal(0, 1).cdf((score_chosen - score_reject) / ((var + var + eps) ** 0.5))
    return (1.0 - torch.sqrt(p + eps)).mean()


def get_sigmas_and_timesteps(pipeline, u, n_dim):
    return pipeline.get_timesteps_from_sigma(pipeline.scheduler, u, n_dim=n_dim)


def encode_prompt(pipeline, prompts, max_sequence_length=128, uncond_prob=0.0, device=None):
    if uncond_prob > 0.0 and device is not None:
        mask = torch.rand(len(prompts), device=device) < uncond_prob
        prompts = ["" if mask[i].item() else prompts[i] for i in range(len(prompts))]
    with torch.no_grad():
        prompt_embeds, pooled = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2="",
            prompt_3="",
            max_sequence_length=max_sequence_length,
        )
    return {
        "encoder_hidden_states": prompt_embeds,
        "pooled_projections": pooled,
    }


def score_pair(pipeline, batch, config, accelerator, u=None, noise=None):
    with diffusion_forward_context(getattr(pipeline, "_sd3_reward_vllm_config", None), pipeline.od_config):
        device = accelerator.device
        text_conds = encode_prompt(
            pipeline,
            batch["prompt"],
            config.model.get("max_sequence_length", 128),
            float(config.training.get("uncond_prob", 0.0)),
            device=device,
        )
        chosen = batch["latent_chosen"].to(device=device, dtype=pipeline.od_config.dtype)
        reject = batch["latent_reject"].to(device=device, dtype=pipeline.od_config.dtype)
        if noise is None:
            noise = torch.randn_like(chosen)
        if u is None:
            u = timestep_sampling(
                config.training.t_weighting_scheme,
                config.training.t_weighting_scheme_param,
                chosen.shape[0],
                device,
            )
        elif not torch.is_tensor(u):
            u = torch.full(size=(chosen.shape[0],), fill_value=float(u), device=device)
        sigmas, timesteps = get_sigmas_and_timesteps(pipeline, u, chosen.dim())
        if config.training.add_noise:
            latent_sigmas = sigmas.to(dtype=pipeline.od_config.dtype)
            chosen = ((1.0 - latent_sigmas) * chosen + latent_sigmas * noise).to(dtype=pipeline.od_config.dtype)
            reject = ((1.0 - latent_sigmas) * reject + latent_sigmas * noise).to(dtype=pipeline.od_config.dtype)
        scores_chosen = pipeline.reward_model(latents=chosen, timesteps=timesteps, **text_conds)
        scores_reject = pipeline.reward_model(latents=reject, timesteps=timesteps, **text_conds)
        return scores_chosen, scores_reject, sigmas


@torch.no_grad()
def validate(pipeline, dataloaders: dict[str, Any], config, accelerator, step: int, ema_manager=None):
    was_training = pipeline.reward_model.training
    if ema_manager is not None:
        ema_manager.swap_in(accelerator.unwrap_model(pipeline.reward_model))
    try:
        return _validate_impl(pipeline, dataloaders, config, accelerator, step)
    finally:
        if ema_manager is not None:
            ema_manager.swap_out(accelerator.unwrap_model(pipeline.reward_model))
        pipeline.reward_model.train(was_training)


@torch.no_grad()
def _validate_impl(pipeline, dataloaders: dict[str, Any], config, accelerator, step: int):
    pipeline.reward_model.eval()
    metrics = {}
    validation_u = list(config.logging.get("validation_u", [0.3, 0.4, 0.5]))
    for split, dataloader in dataloaders.items():
        total_correct = {u: torch.tensor(0.0, device=accelerator.device) for u in validation_u}
        total = torch.tensor(0.0, device=accelerator.device)
        total_loss = {u: torch.tensor(0.0, device=accelerator.device) for u in validation_u}
        all_scores_chosen = {u: [] for u in validation_u}
        all_scores_reject = {u: [] for u in validation_u}
        for batch in tqdm(dataloader, desc=f"Validating {split}", disable=not accelerator.is_local_main_process):
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(accelerator.device)
            noise = torch.randn_like(batch["latent_chosen"])
            bsz = batch["latent_chosen"].shape[0]
            total += bsz
            for u in validation_u:
                scores_chosen, scores_reject, sigmas = score_pair(
                    pipeline, batch, config, accelerator, u=u, noise=noise
                )
                loss = thurstone_loss(scores_chosen, scores_reject, sigmas)
                total_loss[u] += loss.detach() * bsz
                total_correct[u] += ((scores_chosen - scores_reject) > 0.0).float().sum()
                all_scores_chosen[u].extend(scores_chosen.detach().float().cpu().view(-1).tolist())
                all_scores_reject[u].extend(scores_reject.detach().float().cpu().view(-1).tolist())
        for u in validation_u:
            reduced = accelerator.reduce(
                {"loss": total_loss[u], "correct": total_correct[u], "total": total},
                reduction="sum",
            )
            denom = max(reduced["total"].item(), 1.0)
            metrics[f"{split}_val_{u}/loss"] = reduced["loss"].item() / denom
            metrics[f"{split}_val_{u}/accuracy"] = reduced["correct"].item() / denom
            if accelerator.is_main_process and all_scores_chosen[u]:
                chosen_tensor = torch.tensor(all_scores_chosen[u])
                reject_tensor = torch.tensor(all_scores_reject[u])
                metrics[f"{split}_val_{u}/scores_chosen_mean"] = chosen_tensor.mean().item()
                metrics[f"{split}_val_{u}/scores_reject_mean"] = reject_tensor.mean().item()
                metrics[f"{split}_val_{u}/scores_chosen_std"] = chosen_tensor.std(unbiased=False).item()
                metrics[f"{split}_val_{u}/scores_reject_std"] = reject_tensor.std(unbiased=False).item()
    accelerator.log(metrics, step=step)
    return metrics


def save_checkpoint(pipeline, accelerator, config, output_dir, step, ema_manager=None):
    if not accelerator.is_main_process:
        return
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{step:05d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    reward_model = accelerator.unwrap_model(pipeline.reward_model)
    reward_head = accelerator.unwrap_model(reward_model.reward_head)
    torch.save(reward_head.state_dict(), os.path.join(ckpt_dir, "rm_head.pt"))
    if config.model.use_lora and not config.model.freeze_backbone:
        backbone = accelerator.unwrap_model(reward_model.backbone)
        save_backbone_lora_peft(backbone, os.path.join(ckpt_dir, "backbone_lora"), reward_model.config)
    elif not config.model.freeze_backbone:
        torch.save(reward_model.state_dict(), os.path.join(ckpt_dir, "full_model.pt"))
    if ema_manager is not None:
        ema_manager.save(os.path.join(ckpt_dir, "ema_state.pt"))


def build_lr_scheduler(optimizer, config, max_steps: int):
    warmup_steps = config.training.get("warmup_steps", 0)
    if warmup_steps is None:
        warmup_steps = 0
    elif isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)
    else:
        warmup_steps = int(warmup_steps)

    lr_scheduler_name = str(config.training.get("lr_scheduler", "constant"))
    if lr_scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    return None


def resolve_training_steps(config, train_dataloader, accelerator: Accelerator) -> tuple[int, int]:
    num_epochs = int(config.training.get("num_epochs", 1))
    explicit_max_steps = config.training.get("max_steps", None)
    if explicit_max_steps is not None:
        return int(explicit_max_steps), num_epochs

    num_training_steps = (
        len(train_dataloader)
        * num_epochs
        // accelerator.num_processes
        // int(config.training.gradient_accumulation_steps)
    )
    return int(num_training_steps), num_epochs


def main():
    args = parse_args()
    config = load_config(args.config)
    run_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_name = f"{config.paths.run_name}_{run_id}" if config.paths.run_name else run_id
    output_dir = os.path.join(config.paths.save_dir, run_name)
    os.environ.setdefault("WANDB_DIR", os.path.join(output_dir, "wandb"))
    wandb_mode = str(config.logging.get("wandb_mode", os.environ.get("WANDB_MODE", "online")))
    os.environ["WANDB_MODE"] = wandb_mode

    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=output_dir,
    )
    if config.get("system", {}).get("seed") is not None:
        torch.manual_seed(int(config.system.seed))
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(config, resolve=True), f, indent=2)
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True)

    dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        dtype = torch.bfloat16

    pipeline = build_pipeline(config, accelerator, dtype)
    train_lora_backbone = config.model.use_lora and not config.model.freeze_backbone
    if not config.model.freeze_backbone and not config.model.use_lora:
        raise NotImplementedError("Full unfrozen backbone training without LoRA is not supported.")

    train_dataloader = create_bucket_dataloader(
        world_size=accelerator.num_processes,
        global_rank=accelerator.process_index,
        **config.data.train,
    )
    eval_dataloaders = {split: create_simple_dataloader(**cfg) for split, cfg in config.data.eval.items()}
    optimizer = torch.optim.AdamW(
        (p for p in pipeline.reward_model.parameters() if p.requires_grad),
        lr=float(config.training.learning_rate),
        weight_decay=config.training.weight_decay,
    )
    max_steps, num_epochs = resolve_training_steps(config, train_dataloader, accelerator)
    lr_scheduler = build_lr_scheduler(optimizer, config, max_steps)
    if train_lora_backbone:
        pipeline.reward_model.backbone, pipeline.reward_model.reward_head, optimizer, train_dataloader = accelerator.prepare(
            pipeline.reward_model.backbone,
            pipeline.reward_model.reward_head,
            optimizer,
            train_dataloader,
        )
    else:
        pipeline.reward_model.reward_head, optimizer, train_dataloader = accelerator.prepare(
            pipeline.reward_model.reward_head,
            optimizer,
            train_dataloader,
        )
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)
    for split in eval_dataloaders:
        eval_dataloaders[split] = accelerator.prepare(eval_dataloaders[split])

    use_ema = bool(config.model.get("use_ema", False))
    ema_manager = None
    if use_ema:
        ema_manager = RewardEMAManager(accelerator.unwrap_model(pipeline.reward_model), float(config.model.get("ema_decay", 0.995)))

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.logging.get("wandb_project", "diffusion-rm"),
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": run_name}},
        )
    global_step = 0
    progress = tqdm(total=max_steps, desc="Training", disable=not accelerator.is_local_main_process)
    pipeline.reward_model.train()

    for epoch in range(num_epochs):
        if hasattr(train_dataloader, "start_epoch"):
            train_dataloader.start_epoch()
        for batch in train_dataloader:
            if global_step >= max_steps:
                break
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(accelerator.device)
            with accelerator.accumulate(pipeline.reward_model):
                with accelerator.autocast():
                    scores_chosen, scores_reject, sigmas = score_pair(pipeline, batch, config, accelerator)
                    loss = thurstone_loss(scores_chosen, scores_reject, sigmas)
                accelerator.backward(loss)
                grad_norm = None
                if config.training.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(
                        pipeline.reward_model.parameters(),
                        config.training.max_grad_norm,
                    )
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                if ema_manager is not None and accelerator.sync_gradients:
                    ema_manager.update(accelerator.unwrap_model(pipeline.reward_model))

            if accelerator.sync_gradients:
                acc = ((scores_chosen - scores_reject) > 0.0).float().mean()
                train_metrics = {
                    "train/loss": loss.detach(),
                    "train/accuracy": acc.detach(),
                    "train/scores_chosen_mean": scores_chosen.detach().mean(),
                    "train/scores_reject_mean": scores_reject.detach().mean(),
                    "train/lr": torch.tensor(optimizer.param_groups[0]["lr"], device=accelerator.device),
                    "epoch": torch.tensor(epoch + 1, device=accelerator.device),
                }
                if grad_norm is not None:
                    if not torch.is_tensor(grad_norm):
                        grad_norm = torch.tensor(grad_norm, device=accelerator.device)
                    train_metrics["train/grad_norm"] = grad_norm.detach()
                accelerator.log(train_metrics, step=global_step)
                if global_step and global_step % config.logging.eval_frac == 0:
                    validate(pipeline, eval_dataloaders, config, accelerator, global_step, ema_manager=ema_manager)
                if global_step and global_step % config.logging.save_frac == 0:
                    save_checkpoint(pipeline, accelerator, config, output_dir, global_step, ema_manager=ema_manager)
                global_step += 1
                progress.update(1)
        if global_step >= max_steps:
            break
    save_checkpoint(pipeline, accelerator, config, output_dir, global_step, ema_manager=ema_manager)
    accelerator.end_training()
    cleanup_diffusion_distributed()


if __name__ == "__main__":
    main()
