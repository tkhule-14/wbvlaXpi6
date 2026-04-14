"""Pi0.6-WB: Pi0.6 extended with WholeBodyVLA-style hierarchical whole-body control.

Architecture:
    - PaliGemma VLM backbone (SigLIP-400M + Gemma 2B) — unchanged from pi0.6
    - Action Expert (Gemma 2B, flow matching) — adapted for G2's 20D manipulation action space
    - Locomotion Head (MLP) — new; predicts 3D velocity command [vx, vy, omega_z] per VLA step

Key differences from openpi/pi0_pytorch.py:
    - state_dim and action_dim are decoupled (G2: state=25D, manip_action=20D)
    - state_proj: Linear(state_dim, expert_width)  [was: action_dim]
    - action_in_proj: Linear(manip_action_dim, expert_width)  [was: action_dim]
    - action_out_proj: Linear(expert_width, manip_action_dim)  [was: action_dim]
    - locomotion_head: MLP on mean-pooled VLM hidden states → 3D loco command
    - Locomotion head is called ONCE per inference step (not inside denoising loop)

Weight surgery from openpie-0.6 (ALOHA 14D) → G2 (20D):
    - action_in_proj: copy [:, :14] from checkpoint, zero-init [:, 14:]
    - action_out_proj: copy [:14, :] from checkpoint, zero-init [14:, :]
    - state_proj: reinitialize (G2 state semantics differ from ALOHA)
    - locomotion_head: initialize from scratch
    - Everything else: load from checkpoint unchanged
"""

import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../exla-fla/src'))

import fla.models.gemma as _gemma
from fla.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import fla.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger(__name__)


def get_safe_dtype(target_dtype, device_type):
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
    return target_dtype


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be shape (batch_size,)")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    dist = torch.distributions.Beta(
        torch.as_tensor(alpha, dtype=torch.float32, device=device),
        torch.as_tensor(beta, dtype=torch.float32, device=device),
    )
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class Pi0WB(nn.Module):
    """Pi0.6-WB: whole-body VLA for the AgiBot G2 platform.

    Action space (G2):
        Manipulation (20D): 14D arm joints + 2D grippers + 2D head + 2D waist
        Locomotion  ( 3D): [vx, vy, omega_z] base velocity commands

    State space (G2, ~25D):
        14D arm joint positions + 2D gripper positions + 3D base position + 6D base velocity
    """

    def __init__(self, config: "Pi0WBConfig"):
        super().__init__()
        self.config = config

        paligemma_cfg = _gemma.get_config(config.paligemma_variant)
        action_expert_cfg = _gemma.get_config(config.action_expert_variant)
        expert_width = action_expert_cfg.width  # 2048 for gemma_300m

        # PaliGemma + Action Expert (unchanged architecture)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_cfg,
            action_expert_cfg,
            use_adarms=[False, False],  # pi0 style (not pi0.5)
            precision=config.dtype,
        )

        # State projection: state → Action Expert token (pi0 style, not pi0.5)
        # G2 state_dim decoupled from manip_action_dim
        self.state_proj = nn.Linear(config.state_dim, expert_width)

        # Action Expert I/O: flow matching over manipulation actions
        self.action_in_proj = nn.Linear(config.manip_action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.manip_action_dim)

        # Timestep MLP (fuses noised action + time embedding)
        self.action_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
        self.action_time_mlp_out = nn.Linear(expert_width, expert_width)

        # Locomotion head: MLP on mean-pooled VLM hidden states → 3D velocity command
        # Called ONCE per VLA inference step, NOT inside denoising loop
        paligemma_width = paligemma_cfg.width  # 2048 for gemma_2b
        self.locomotion_head = nn.Sequential(
            nn.Linear(paligemma_width, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, config.loco_action_dim),  # [vx, vy, omega_z]
        )

        # Verify transformers_replace patches are installed
        try:
            from transformers.models.siglip import check
            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(
                    "transformers_replace not installed. Run: "
                    "cp -r ./exla-fla/src/fla/models_pytorch/transformers_replace/* "
                    "$(python -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/"
                )
        except ImportError:
            logger.warning("Could not verify transformers_replace — may cause incorrect SigLIP behavior")

        torch.set_float32_matmul_precision("high")

        # Gradient checkpointing (disabled by default)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing")

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Weight surgery: adapt openpie-0.6 (ALOHA 14D) → G2 (20D)
    # ------------------------------------------------------------------

    @classmethod
    def from_openpie_checkpoint(cls, checkpoint_path: str, config: "Pi0WBConfig") -> "Pi0WB":
        """Load openpie-0.6 weights with weight surgery for G2 action dims.

        Surgery rules:
            action_in_proj  : copy cols [:14] from checkpoint, zero-init cols [14:]
            action_out_proj : copy rows [:14] from checkpoint, zero-init rows [14:]
            state_proj      : reinitialize (G2 state semantics differ from ALOHA)
            locomotion_head : initialize from scratch
            everything else : load unchanged
        """
        from safetensors.torch import load_file

        model = cls(config)
        ckpt = load_file(checkpoint_path)

        # Remap checkpoint keys from openpie naming → our naming
        # openpie uses "paligemma_with_expert.*", "action_in_proj.*", "action_out_proj.*"
        state_dict = model.state_dict()
        surgery_log = []

        for key in list(state_dict.keys()):
            ckpt_key = key  # names should match except for new/resized layers

            if key == "action_in_proj.weight":
                # [expert_width, manip_action_dim] — cols are input dims
                if ckpt_key in ckpt:
                    old_w = ckpt[ckpt_key]  # [expert_width, 14]
                    new_w = torch.zeros_like(state_dict[key])  # [expert_width, 20]
                    new_w[:, :old_w.shape[1]] = old_w
                    state_dict[key] = new_w
                    surgery_log.append(f"action_in_proj.weight: copied [:, :14], zero-init [:, 14:]")

            elif key == "action_in_proj.bias":
                if ckpt_key in ckpt:
                    state_dict[key] = ckpt[ckpt_key]

            elif key == "action_out_proj.weight":
                # [manip_action_dim, expert_width] — rows are output dims
                if ckpt_key in ckpt:
                    old_w = ckpt[ckpt_key]  # [14, expert_width]
                    new_w = torch.zeros_like(state_dict[key])  # [20, expert_width]
                    new_w[:old_w.shape[0], :] = old_w
                    state_dict[key] = new_w
                    surgery_log.append(f"action_out_proj.weight: copied [:14, :], zero-init [14:, :]")

            elif key == "action_out_proj.bias":
                if ckpt_key in ckpt:
                    old_b = ckpt[ckpt_key]  # [14]
                    new_b = torch.zeros_like(state_dict[key])  # [20]
                    new_b[:old_b.shape[0]] = old_b
                    state_dict[key] = new_b

            elif key.startswith("state_proj."):
                # Reinitialize — G2 state semantics differ from ALOHA
                surgery_log.append(f"{key}: reinitializing (G2 state != ALOHA state)")

            elif key.startswith("locomotion_head."):
                # New module — initialize from scratch
                surgery_log.append(f"{key}: new module, initializing from scratch")

            elif ckpt_key in ckpt:
                state_dict[key] = ckpt[ckpt_key]

            else:
                logger.warning(f"Key {key} not found in checkpoint — keeping random init")

        model.load_state_dict(state_dict, strict=True)

        for msg in surgery_log:
            logger.info(f"[weight surgery] {msg}")

        return model

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(
            observation,
            train=train,
            image_keys=self.config.image_keys,
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        return (sample_beta(1.5, 1.0, bsize, device) * 0.999 + 0.001).to(dtype=torch.float32)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images + language for PaliGemma prefix."""
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        bsize = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state + noisy actions + timestep for Action Expert suffix."""
        embs, pad_masks, att_masks = [], [], []

        # State token (pi0 style — not pi0.5)
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=state_emb.device))
        att_masks += [1]  # image/lang do not attend to state

        # Timestep sinusoidal embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0, device=timestep.device,
        ).to(dtype=timestep.dtype)

        # Action + time fusion MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
        action_time_emb = F.silu(self.action_time_mlp_in(action_time_emb))
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)
        bsize, T = action_time_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, T, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + [0] * (self.config.action_horizon - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, observation, manip_actions, loco_actions, noise=None, time=None):
        """Training forward: returns (manip_flow_loss, loco_mse_loss).

        Args:
            observation: Observation dataclass with .state, .images, .tokenized_prompt, etc.
            manip_actions: [B, action_horizon, manip_action_dim] manipulation action targets
            loco_actions:  [B, loco_action_dim] locomotion velocity command targets
            noise: optional pre-sampled noise for flow matching
            time:  optional pre-sampled time for flow matching

        Returns:
            manip_loss: scalar, flow matching MSE over manipulation actions
            loco_loss:  scalar, MSE over locomotion velocity commands
            prefix_hidden: [B, prefix_len, paligemma_width] VLM hidden states (for loco head)
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(manip_actions.shape, manip_actions.device)
        if time is None:
            time = self.sample_time(manip_actions.shape[0], manip_actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * manip_actions
        u_t = noise - manip_actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None],
        )

        # Manipulation flow matching loss
        # Note: reference returns reduction="none" for per-dim/per-step weighting flexibility.
        # We reduce to scalar here since our training loop expects scalar losses.
        suffix_out_manip = suffix_out[:, -self.config.action_horizon:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out_manip)
        manip_loss = F.mse_loss(u_t, v_t)

        # Locomotion loss — mean-pool prefix (VLM) hidden states
        # prefix_out: [B, prefix_len, paligemma_width]
        prefix_out_fp32 = prefix_out.to(dtype=torch.float32)
        # Mask-weighted mean pooling over valid prefix tokens
        valid = prefix_pad_masks.unsqueeze(-1).float()  # [B, prefix_len, 1]
        prefix_pooled = (prefix_out_fp32 * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # [B, hidden]
        loco_pred = self.locomotion_head(prefix_pooled)  # [B, loco_action_dim]
        loco_loss = F.mse_loss(loco_pred, loco_actions.to(dtype=torch.float32))

        return manip_loss, loco_loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        """Inference: returns (manip_actions, loco_command).

        manip_actions: [B, action_horizon, manip_action_dim]
        loco_command:  [B, loco_action_dim]
        """
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.action_horizon, self.config.manip_action_dim), device
            )

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Locomotion: predict ONCE from VLM prefix hidden states
        prefix_out_fp32 = prefix_out.to(dtype=torch.float32)
        valid = prefix_pad_masks.unsqueeze(-1).float()
        prefix_pooled = (prefix_out_fp32 * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        loco_command = self.locomotion_head(prefix_pooled)  # [B, 3]

        # Manipulation: flow matching denoising loop
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, expanded_time)

            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d)

            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=position_ids,
                past_key_values=past_kv,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None],
            )

            suffix_out = suffix_out[:, -self.config.action_horizon:].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t, loco_command


class Pi0WBConfig:
    """Configuration for Pi0.6-WB.

    G2 defaults:
        state_dim       = 25   (14D arm joints + 2D grippers + 3D base pos + 6D base vel)
        manip_action_dim= 20   (14D arms + 2D grippers + 2D head + 2D waist)
        loco_action_dim = 3    ([vx, vy, omega_z])
        action_horizon  = 50   (reduce to 30 if latency > 100ms)
    """

    def __init__(
        self,
        state_dim: int = 25,
        manip_action_dim: int = 20,
        loco_action_dim: int = 3,
        action_horizon: int = 50,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: str = "bfloat16",
        image_keys: tuple = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    ):
        self.state_dim = state_dim
        self.manip_action_dim = manip_action_dim
        self.loco_action_dim = loco_action_dim
        self.action_horizon = action_horizon
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.dtype = dtype
        self.image_keys = image_keys
        # For compatibility with openpi code
        self.pi05 = False
        self.action_dim = manip_action_dim  # used by some openpi internals
