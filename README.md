# Pi0.6-WB

**Pi0.6 + WholeBodyVLA**: A hybrid Vision-Language-Action model combining [pi0.6](https://github.com/exla-ai/fla)'s flow matching backbone with [WholeBodyVLA](https://arxiv.org/abs/2502.07792)-style hierarchical whole-body control for humanoid robots.

Built for the [AgiBot World Challenge 2026](https://agibot-world.com/) (G2 embodiment).

## Architecture

```
Input: 3x RGB (head + wrist L/R) + language instruction + proprioceptive state
                            |
               PaliGemma VLM (SigLIP-400M + Gemma 2B)
                   /                       \
        Action Expert (Gemma 300M)      Locomotion Head (MLP)
        cross-attn to VLM               mean-pool VLM hidden
        + flow matching                  2048 -> 512 -> 128 -> 3
                |                               |
        20D manipulation chunk          3D velocity command
        @ 10-20 Hz                      [vx, vy, wz] @ 10 Hz
```

**Manipulation (20D):** 14D arm joints + 2D grippers + 2D head + 2D waist  
**Locomotion (3D):** `[vx, vy, omega_z]` base velocity commands -> downstream RL policy at 50-100 Hz

### Key design choices

- **Flow matching** for manipulation (continuous, not VQ-VAE discretization)
- **Hierarchical decomposition**: separate manipulation (Action Expert, chunked) and locomotion (MLP, single command) heads
- **Single VLM, multi-rate control**: PaliGemma produces both 10 Hz manipulation chunks and 10 Hz loco commands
- **Weight surgery** from openpie-0.6 (ALOHA 14D) to G2 (20D): zero-init expansion of action projection layers

## Setup

```bash
# Clone this repo
git clone https://github.com/<your-org>/pi06-wb.git
cd pi06-wb

# Create environment (Python 3.10+)
uv venv && source .venv/bin/activate

# Run setup (clones exla-fla, installs deps, patches transformers)
bash setup.sh
```

### Manual setup

```bash
# 1. Clone exla-ai/fla (openpie-0.6 codebase)
git clone https://github.com/exla-ai/fla.git exla-fla

# 2. Install dependencies
pip install -r requirements.txt

# 3. Apply transformers_replace patches (required for SigLIP/PaliGemma)
TRANSFORMERS_DIR=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py "$TRANSFORMERS_DIR/models/gemma/modeling_gemma.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py "$TRANSFORMERS_DIR/models/gemma/configuration_gemma.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py "$TRANSFORMERS_DIR/models/paligemma/modeling_paligemma.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py "$TRANSFORMERS_DIR/models/siglip/modeling_siglip.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/siglip/check.py "$TRANSFORMERS_DIR/models/siglip/check.py"

# 4. Download openpie-0.6 weights (for weight surgery)
mkdir -p checkpoints/openpie-0.6
huggingface-cli download exla-ai/openpie-0.6 --local-dir checkpoints/openpie-0.6
```

## Usage

### Validate architecture

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  python pi06_wb/tests/test_forward_pass.py
```

Expected output:
```
Creating model on cuda...
Model created. Parameters: 3,502,455,559
Testing projection shapes... PASS
Testing training forward pass... PASS
Testing inference shapes... PASS
All tests passed!
```

### Train on AgiBot World 2026

```bash
# Download dataset (start with task_3777 subset, ~7GB)
# See: https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

# Train
python pi06_wb/scripts/train.py --config pi06_wb/configs/g2.yaml
```

### Configuration

See [`pi06_wb/configs/g2.yaml`](pi06_wb/configs/g2.yaml) for all training hyperparameters.

Key settings:
| Parameter | Default | Notes |
|-----------|---------|-------|
| `model.action_horizon` | 50 | Reduce to 30 if inference > 100ms |
| `model.num_denoising_steps` | 10 | Reduce to 4 for faster inference |
| `training.lambda_loco` | 0.1 | Locomotion loss weight (tune via grad norms) |
| `training.warmup_steps` | 2000 | Freeze VLM+ActionExpert, train only new layers |
| `training.total_steps` | 20000 | For task_3777 subset; 200K for full dataset |

## Project structure

```
pi06_wb/
  models/
    pi0_wb.py          # Pi0.6-WB model (PaliGemma + Action Expert + Loco Head)
  data/
    agibot_world.py    # AgiBot World 2026 dataloader
  scripts/
    train.py           # Training script (flow matching + loco MSE)
  tests/
    test_forward_pass.py  # Architecture validation tests
  configs/
    g2.yaml            # G2 training configuration
```

## Loss function

```
total_loss = manip_loss + lambda_loco * loco_loss
```

- **manip_loss**: Flow matching MSE between predicted and target velocity fields over 20D manipulation actions
- **loco_loss**: MSE between predicted and target 3D base velocity commands
- Track per-module gradient norms to tune `lambda_loco`

## Weight surgery (ALOHA 14D -> G2 20D)

| Layer | Surgery |
|-------|---------|
| `action_in_proj` | Copy cols `[:, :14]`, zero-init `[:, 14:]` |
| `action_out_proj` | Copy rows `[:14, :]`, zero-init `[14:, :]` |
| `state_proj` | Reinitialize (G2 state semantics differ) |
| `locomotion_head` | Initialize from scratch |
| Everything else | Load unchanged from openpie-0.6 |

## Training strategy

1. **Warmup (0-2K steps):** Freeze VLM + Action Expert. Train only `state_proj`, new action projection rows, and `locomotion_head`.
2. **Full training (2K+ steps):** Unfreeze all parameters. Cosine LR decay from 1e-4.

## Planned ablations

| ID | Variant | Question |
|----|---------|----------|
| A1 | Pi0.6-WB (flow match + MLP loco head) | Full proposed architecture |
| A4 | Flat: single flow match over 23D [manip+loco] | Does hierarchy help? |
| A3 | VQ-VAE instead of flow match | Flow match > VQ-VAE for gripper? |
| A5 | Separate arm + gripper flow match heads | Gripper expert improves dexterity? |
| A6 | Cross-attn loco head (vs. MLP) | Does attending over tokens help coordination? |

## References

- [OpenPIE-0.6](https://github.com/exla-ai/fla) (exla-ai) — base pi0.6 implementation
- [pi_0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) (Physical Intelligence, 2024)
- [WholeBodyVLA: Toward Whole-Body Control for VLA Models](https://arxiv.org/abs/2502.07792) (2025)
- [AgiBot World](https://agibot-world.com/) — dataset and challenge
- [GR00T N1.5](https://arxiv.org/abs/2503.14734) (NVIDIA, 2025) — evaluation baseline

## License

MIT
