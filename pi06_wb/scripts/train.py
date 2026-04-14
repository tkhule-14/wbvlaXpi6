"""Training script for Pi0.6-WB on AgiBot World 2026.

Usage:
    python scripts/train.py --config configs/g2.yaml

Key features:
    - Dual loss: flow_matching (manipulation) + MSE (locomotion)
    - 2K-step warmup: freeze VLM+ActionExpert, train only new layers
    - Per-module gradient norm tracking for lambda_loco tuning
    - BF16 mixed precision
    - Cosine LR decay
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pi06_wb.models.pi0_wb import Pi0WB, Pi0WBConfig
from pi06_wb.data.agibot_world import AgiBotWorldDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_model_config(cfg: dict) -> Pi0WBConfig:
    mc = cfg["model"]
    return Pi0WBConfig(
        state_dim=mc["state_dim"],
        manip_action_dim=mc["manip_action_dim"],
        loco_action_dim=mc["loco_action_dim"],
        action_horizon=mc["action_horizon"],
        paligemma_variant=mc["paligemma_variant"],
        action_expert_variant=mc["action_expert_variant"],
        dtype=mc["dtype"],
        image_keys=tuple(mc["image_keys"]),
    )


def make_observation(batch: dict, device: torch.device):
    """Convert dataloader batch dict → observation object for Pi0WB."""
    class Observation:
        pass

    obs = Observation()
    obs.state = batch["state"].to(device)
    obs.images = {k: v.to(device) for k, v in batch["images"].items()}
    obs.image_masks = {k: v.to(device) for k, v in batch["image_masks"].items()}
    obs.tokenized_prompt = batch["tokenized_prompt"].to(device)
    obs.tokenized_prompt_mask = batch["tokenized_prompt_mask"].to(device)
    # Required by preprocessing_pytorch
    obs.token_ar_mask = torch.zeros_like(batch["tokenized_prompt_mask"], dtype=torch.bool).to(device)
    obs.token_loss_mask = batch["tokenized_prompt_mask"].to(device)
    return obs


def get_warmup_params(model: Pi0WB, freeze_modules: list[str]) -> tuple[list, list]:
    """Split parameters into: (frozen during warmup, always trainable)."""
    frozen_params = []
    trainable_params = []
    freeze_prefixes = tuple(freeze_modules)

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in freeze_prefixes):
            frozen_params.append((name, param))
        else:
            trainable_params.append((name, param))

    return frozen_params, trainable_params


def compute_grad_norms(model: Pi0WB, tracked_modules: list[str]) -> dict[str, float]:
    """Compute per-module gradient norms for monitoring."""
    norms = {}
    for module_name in tracked_modules:
        module = dict(model.named_modules()).get(module_name)
        if module is None:
            continue
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        norms[module_name] = total_norm ** 0.5
    return norms


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model_config = make_model_config(cfg)
    ckpt_cfg = cfg.get("checkpoint", {})
    openpie_ckpt = ckpt_cfg.get("openpie_checkpoint")

    if openpie_ckpt and Path(openpie_ckpt).exists():
        logger.info(f"Loading from openpie-0.6 checkpoint with weight surgery: {openpie_ckpt}")
        model = Pi0WB.from_openpie_checkpoint(openpie_ckpt, model_config)
    else:
        logger.warning("No openpie checkpoint found — initializing from scratch (for testing)")
        model = Pi0WB(model_config)

    model = model.to(device)

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}. Using None (tokens will be zeros).")
        tokenizer = None

    data_cfg = cfg["data"]
    norm_stats = None
    if Path(data_cfg.get("norm_stats_path", "")).exists():
        with open(data_cfg["norm_stats_path"]) as f:
            norm_stats = json.load(f)

    dataset = AgiBotWorldDataset(
        dataset_root=data_cfg["dataset_root"],
        tokenizer=tokenizer,
        action_horizon=data_cfg["action_horizon"],
        max_token_len=data_cfg["max_token_len"],
        use_subtask_split=data_cfg.get("use_subtask_split", True),
        norm_stats=norm_stats,
        train=True,
    )
    logger.info(f"Dataset size: {len(dataset)} frames")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"] // max(1, torch.cuda.device_count()),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        pin_memory=True,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    # Optimizer + LR scheduler
    # ----------------------------------------------------------------
    train_cfg = cfg["training"]

    # Enable gradient checkpointing for memory savings (recommended for <=24GB GPUs)
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["total_steps"])

    lambda_loco = train_cfg.get("lambda_loco", 0.1)
    warmup_steps = train_cfg.get("warmup_steps", 2000)
    freeze_modules = train_cfg.get("warmup_freeze_modules", ["paligemma_with_expert"])
    tracked_modules = train_cfg.get("track_grad_norms", ["locomotion_head", "action_in_proj"])

    # ----------------------------------------------------------------
    # Warmup: freeze VLM + ActionExpert
    # ----------------------------------------------------------------
    frozen_params, _ = get_warmup_params(model, freeze_modules)
    logger.info(f"Warmup: freezing {len(frozen_params)} parameter groups for {warmup_steps} steps")
    for _, param in frozen_params:
        param.requires_grad_(False)

    # ----------------------------------------------------------------
    # Mixed precision (bf16 — no GradScaler needed, same dynamic range as fp32)
    # ----------------------------------------------------------------
    use_bf16 = train_cfg.get("bf16", True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # ----------------------------------------------------------------
    # Output dir
    # ----------------------------------------------------------------
    output_dir = Path(ckpt_cfg.get("output_dir", "checkpoints/pi06_wb_g2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    global_step = 0
    total_steps = train_cfg["total_steps"]
    log_every = train_cfg.get("log_every", 50)
    save_every = train_cfg.get("save_every", 2000)

    model.train()
    data_iter = iter(dataloader)

    logger.info("Starting training loop...")
    t0 = time.time()

    while global_step < total_steps:
        # Refill iterator if exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Unfreeze after warmup
        if global_step == warmup_steps:
            logger.info(f"Step {global_step}: warmup complete — unfreezing all parameters")
            for _, param in frozen_params:
                param.requires_grad_(True)

        observation = make_observation(batch, device)
        manip_actions = batch["manip_actions"].to(device)   # [B, T, 20]
        loco_actions  = batch["loco_action"].to(device)     # [B, 3]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            manip_loss, loco_loss = model(observation, manip_actions, loco_actions)
            total_loss = manip_loss + lambda_loco * loco_loss

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clip", 1.0))

        # Track per-module grad norms (every log_every steps)
        grad_norms = {}
        if global_step % log_every == 0:
            grad_norms = compute_grad_norms(model, tracked_modules)

        optimizer.step()
        scheduler.step()

        global_step += 1

        # Logging
        if global_step % log_every == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"step={global_step:6d} | "
                f"manip_loss={manip_loss.item():.4f} | "
                f"loco_loss={loco_loss.item():.4f} | "
                f"total_loss={total_loss.item():.4f} | "
                f"lr={lr:.2e} | "
                f"elapsed={elapsed:.0f}s"
            )
            if grad_norms:
                norm_str = " | ".join(f"{k}={v:.3f}" for k, v in grad_norms.items())
                logger.info(f"  grad_norms: {norm_str}")

            # Lambda tuning hint: if loco head grad_norm >> action_expert, reduce lambda_loco
            if "locomotion_head" in grad_norms and "action_in_proj" in grad_norms:
                loco_norm = grad_norms.get("locomotion_head", 0)
                manip_norm = grad_norms.get("action_in_proj", 1)
                ratio = loco_norm / (manip_norm + 1e-8)
                if ratio > 10.0:
                    logger.warning(
                        f"  [WARN] loco/manip grad norm ratio={ratio:.1f} > 10. "
                        f"Consider reducing lambda_loco from {lambda_loco}"
                    )

        # Save checkpoint
        if global_step % save_every == 0 or global_step == total_steps:
            ckpt_path = output_dir / f"step_{global_step:06d}.pt"
            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "manip_loss": manip_loss.item(),
                "loco_loss": loco_loss.item(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train Pi0.6-WB on AgiBot World 2026")
    parser.add_argument("--config", type=str, default="configs/g2.yaml")
    parser.add_argument("--lambda-loco", type=float, default=None,
                        help="Override lambda_loco from config")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total_steps from config")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset_root from config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.lambda_loco is not None:
        cfg["training"]["lambda_loco"] = args.lambda_loco
        logger.info(f"Overriding lambda_loco = {args.lambda_loco}")
    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps
    if args.data_root is not None:
        cfg["data"]["dataset_root"] = args.data_root

    train(cfg)


if __name__ == "__main__":
    main()
