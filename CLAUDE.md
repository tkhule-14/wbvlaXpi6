# Pi0.6-WB — Project Context

## What this is

Hybrid VLA combining pi0.6's flow matching backbone with WholeBodyVLA's hierarchical whole-body decomposition. Target: AgiBot World Challenge 2026 (deadline: April 20, 2026). Train on AgiBot World 2026 (G2 embodiment), evaluate against GR00T N1.5 (51% on Humanoid Everyday).

## Architecture (3.5B params)

```
Input: 3x RGB (top_head + hand_left + hand_right) + language + proprioceptive state (25D)
  |
  PaliGemma VLM (SigLIP-400M + Gemma 2B)
  |-- state_proj: Linear(25, 2048) — state → VLM token
  |
  ├── Action Expert (Gemma 300M, cross-attn to VLM, flow matching)
  |   |-- action_in_proj: Linear(20, 2048)
  |   |-- action_time_mlp_in/out: fuse action + sinusoidal time embedding
  |   |-- action_out_proj: Linear(2048, 20)
  |   |-- 10-step Euler denoising at inference
  |   └── Output: 20D manipulation chunk (horizon=50 @ 10-20Hz)
  |
  └── Locomotion Head (MLP on mean-pooled VLM hidden)
      |-- 2048 → 512 → 128 → 3
      |-- Called ONCE per step (NOT inside denoising loop)
      └── Output: [vx, vy, omega_z] @ 10Hz
```

G2 action: 14D arms + 2D grippers + 2D head + 2D waist = 20D manip, 3D loco
G2 state: 14D arm joints + 2D grippers + 3D base pos + 6D base vel = 25D

## Files

- `pi06_wb/models/pi0_wb.py` — Model (Pi0WB + Pi0WBConfig). Imports from exla-fla/src.
- `pi06_wb/data/agibot_world.py` — AgiBot World dataloader. **UNTESTED against real data — verify field names first.**
- `pi06_wb/scripts/train.py` — Training: dual loss, bf16, gradient checkpointing, warmup freeze.
- `pi06_wb/configs/g2.yaml` — All hyperparameters.
- `pi06_wb/tests/test_forward_pass.py` — Architecture validation tests (all pass).
- `setup.sh` — Clone exla-fla, install deps, apply transformers patches.

## Critical dependencies

### exla-fla (exla-ai/fla)
Cloned as `exla-fla/`. Model imports:
- `fla.models.gemma` — config (get_config for gemma_2b, gemma_300m)
- `fla.models_pytorch.gemma_pytorch` — PaliGemmaWithExpertModel
- `fla.models_pytorch.preprocessing_pytorch` — preprocess_observation_pytorch

Despite claiming "No JAX", it requires jax[cpu] + flax for config utilities.

### transformers_replace patches
Stock HuggingFace transformers won't work. Must copy patched files from exla-fla into transformers install (SigLIP get_image_features returns raw tensor, not BaseModelOutputWithPooling). setup.sh handles this.

## Weight surgery (openpie-0.6 ALOHA 14D → G2 20D)

Via `Pi0WB.from_openpie_checkpoint()`:
- action_in_proj: copy cols [:, :14], zero-init [:, 14:]
- action_out_proj: copy rows [:14, :], zero-init [14:, :]
- state_proj: reinitialize (different state semantics)
- locomotion_head: from scratch
- Everything else: load unchanged

## Training

- Loss: `total = manip_flow_matching_mse + 0.1 * loco_mse`
- Warmup 0-2K steps: freeze VLM + Action Expert, train only new layers
- Full 2K+: unfreeze all, cosine LR 1e-4
- bf16, grad clip 1.0, gradient checkpointing available
- Track per-module grad norms to tune lambda_loco

## Server setup steps

```bash
bash setup.sh
huggingface-cli download exla-ai/openpie-0.6 --local-dir checkpoints/openpie-0.6
# Download AgiBot World task_3777 subset (~7GB)
# Compute norm stats before training
python pi06_wb/scripts/train.py --config pi06_wb/configs/g2.yaml
```

## Known issues

- Data loader field names unverified against real AgiBot World data — check first
- Norm stats must be computed before training starts
- lambda_loco=0.1 needs tuning via grad norms (watch for loco head saturating VLM gradients)
- Loss uses reduction="mean" (reference uses "none") — intentional
- LD_LIBRARY_PATH may need setting for cuDNN on some systems
- User environment uses uv, not pip/conda

## Compute

- 8x H200 cluster for training
- batch_size=128 in config, consider 256 on H200s
- Deadline: April 20, 2026 — prioritize getting training running over polish
