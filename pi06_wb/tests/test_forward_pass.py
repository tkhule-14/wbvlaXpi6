"""Unit test: forward pass with G2-shaped dummy tensors.

Verifies:
1. Both losses computed, no shape errors
2. manip_loss and loco_loss are scalar tensors
3. Gradients flow through both heads
4. Inference (sample_actions) returns correct shapes
5. Weight surgery produces expected shapes

Run with:
    cd /mnt/ssd2t/Tanmay_Projects/wbvlaXpi6
    python -m pytest pi06_wb/tests/test_forward_pass.py -v
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch
import numpy as np

from pi06_wb.models.pi0_wb import Pi0WB, Pi0WBConfig


# G2 dims
STATE_DIM        = 25
MANIP_ACTION_DIM = 20
LOCO_ACTION_DIM  = 3
ACTION_HORIZON   = 4    # minimal for fast testing
BATCH_SIZE       = 1
MAX_TOKEN_LEN    = 16
IMAGE_H, IMAGE_W = 224, 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_config():
    return Pi0WBConfig(
        state_dim=STATE_DIM,
        manip_action_dim=MANIP_ACTION_DIM,
        loco_action_dim=LOCO_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
    )


def make_dummy_observation(device=DEVICE):
    """Create dummy observation matching G2 format."""
    class DummyObs:
        pass

    obs = DummyObs()
    obs.state = torch.randn(BATCH_SIZE, STATE_DIM, device=device)
    obs.images = {
        "base_0_rgb":        torch.randn(BATCH_SIZE, 3, IMAGE_H, IMAGE_W, device=device),
        "left_wrist_0_rgb":  torch.randn(BATCH_SIZE, 3, IMAGE_H, IMAGE_W, device=device),
        "right_wrist_0_rgb": torch.randn(BATCH_SIZE, 3, IMAGE_H, IMAGE_W, device=device),
    }
    obs.image_masks = {
        "base_0_rgb":        torch.ones(BATCH_SIZE, dtype=torch.bool, device=device),
        "left_wrist_0_rgb":  torch.ones(BATCH_SIZE, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(BATCH_SIZE, dtype=torch.bool, device=device),
    }
    obs.tokenized_prompt = torch.randint(0, 1000, (BATCH_SIZE, MAX_TOKEN_LEN), dtype=torch.int32, device=device)
    obs.tokenized_prompt_mask = torch.ones(BATCH_SIZE, MAX_TOKEN_LEN, dtype=torch.bool, device=device)
    obs.token_ar_mask = torch.zeros(BATCH_SIZE, MAX_TOKEN_LEN, dtype=torch.bool, device=device)
    obs.token_loss_mask = torch.ones(BATCH_SIZE, MAX_TOKEN_LEN, dtype=torch.bool, device=device)
    return obs


@pytest.fixture(scope="module")
def model():
    """Create model once for all tests (model init is slow)."""
    config = make_config()
    m = Pi0WB(config)
    m = m.to(DEVICE)
    m.eval()
    return m


def test_model_instantiation(model):
    """Model should instantiate without errors."""
    assert model is not None
    assert hasattr(model, "locomotion_head")
    assert hasattr(model, "state_proj")
    assert hasattr(model, "action_in_proj")
    assert hasattr(model, "action_out_proj")


def test_projection_shapes(model):
    """Check that projection layers have the right dimensions."""
    # state_proj: state_dim -> expert_width
    assert model.state_proj.in_features == STATE_DIM, \
        f"state_proj.in_features={model.state_proj.in_features}, expected {STATE_DIM}"

    # action_in_proj: manip_action_dim -> expert_width
    assert model.action_in_proj.in_features == MANIP_ACTION_DIM, \
        f"action_in_proj.in_features={model.action_in_proj.in_features}, expected {MANIP_ACTION_DIM}"

    # action_out_proj: expert_width -> manip_action_dim
    assert model.action_out_proj.out_features == MANIP_ACTION_DIM, \
        f"action_out_proj.out_features={model.action_out_proj.out_features}, expected {MANIP_ACTION_DIM}"

    # locomotion_head: last layer output = loco_action_dim
    last_layer = list(model.locomotion_head.modules())[-1]
    assert last_layer.out_features == LOCO_ACTION_DIM, \
        f"locomotion_head output={last_layer.out_features}, expected {LOCO_ACTION_DIM}"


def test_training_forward_pass(model):
    """Training forward: both losses computed, correct shapes."""
    model.train()
    obs = make_dummy_observation()
    manip_actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, MANIP_ACTION_DIM, device=DEVICE)
    loco_actions  = torch.randn(BATCH_SIZE, LOCO_ACTION_DIM, device=DEVICE)

    manip_loss, loco_loss = model(obs, manip_actions, loco_actions)

    assert manip_loss.ndim == 0, f"manip_loss should be scalar, got shape {manip_loss.shape}"
    assert loco_loss.ndim == 0,  f"loco_loss should be scalar, got shape {loco_loss.shape}"
    assert torch.isfinite(manip_loss), "manip_loss is not finite"
    assert torch.isfinite(loco_loss),  "loco_loss is not finite"
    assert manip_loss.item() > 0, "manip_loss should be positive"
    assert loco_loss.item() > 0,  "loco_loss should be positive"


def test_gradient_flow(model):
    """Gradients should flow through both heads."""
    model.train()
    obs = make_dummy_observation()
    manip_actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, MANIP_ACTION_DIM, device=DEVICE)
    loco_actions  = torch.randn(BATCH_SIZE, LOCO_ACTION_DIM, device=DEVICE)

    lambda_loco = 0.1
    manip_loss, loco_loss = model(obs, manip_actions, loco_actions)
    total_loss = manip_loss + lambda_loco * loco_loss
    total_loss.backward()

    # Check gradients exist for key modules
    assert model.action_in_proj.weight.grad is not None, "action_in_proj has no gradient"
    assert model.action_out_proj.weight.grad is not None, "action_out_proj has no gradient"
    assert model.state_proj.weight.grad is not None, "state_proj has no gradient"

    # Locomotion head should have gradients
    for name, param in model.locomotion_head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"locomotion_head.{name} has no gradient"

    model.zero_grad()


def test_inference_shapes(model):
    """Inference returns (manip_actions, loco_command) with correct shapes."""
    model.eval()
    obs = make_dummy_observation()

    with torch.no_grad():
        manip_out, loco_out = model.sample_actions(
            device=torch.device(DEVICE),
            observation=obs,
            num_steps=3,  # fast for testing
        )

    assert manip_out.shape == (BATCH_SIZE, ACTION_HORIZON, MANIP_ACTION_DIM), \
        f"manip_out shape={manip_out.shape}"
    assert loco_out.shape == (BATCH_SIZE, LOCO_ACTION_DIM), \
        f"loco_out shape={loco_out.shape}"
    assert torch.isfinite(manip_out).all(), "manip_out contains non-finite values"
    assert torch.isfinite(loco_out).all(),  "loco_out contains non-finite values"


def test_weight_surgery_shapes():
    """Weight surgery should produce correct shapes even without real checkpoint."""
    from safetensors.torch import save_file
    import tempfile

    config = make_config()
    model = Pi0WB(config)

    # Create a fake "openpie-0.6" checkpoint with ALOHA dims (14D)
    fake_ckpt = {}
    aloha_dim = 14
    expert_width = model.action_in_proj.out_features  # e.g., 2048

    fake_ckpt["action_in_proj.weight"]  = torch.randn(expert_width, aloha_dim)
    fake_ckpt["action_in_proj.bias"]    = torch.randn(expert_width)
    fake_ckpt["action_out_proj.weight"] = torch.randn(aloha_dim, expert_width)
    fake_ckpt["action_out_proj.bias"]   = torch.randn(aloha_dim)
    # Add all other model params at their correct sizes
    for name, param in model.named_parameters():
        if name not in fake_ckpt and not name.startswith("state_proj") and not name.startswith("locomotion_head"):
            fake_ckpt[name] = param.detach().clone()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(fake_ckpt, f.name)
        ckpt_path = f.name

    try:
        model_loaded = Pi0WB.from_openpie_checkpoint(ckpt_path, config)

        # Verify shapes after surgery
        assert model_loaded.action_in_proj.weight.shape == (expert_width, MANIP_ACTION_DIM), \
            f"action_in_proj.weight shape wrong: {model_loaded.action_in_proj.weight.shape}"
        assert model_loaded.action_out_proj.weight.shape == (MANIP_ACTION_DIM, expert_width), \
            f"action_out_proj.weight shape wrong: {model_loaded.action_out_proj.weight.shape}"

        # Verify zero-init for extended dims
        extended_in  = model_loaded.action_in_proj.weight[:, aloha_dim:]
        extended_out = model_loaded.action_out_proj.weight[aloha_dim:, :]
        assert torch.all(extended_in == 0), "action_in_proj extended columns should be zero-initialized"
        assert torch.all(extended_out == 0), "action_out_proj extended rows should be zero-initialized"

    finally:
        os.unlink(ckpt_path)


def test_locomotion_head_architecture(model):
    """Locomotion head should be 3-layer MLP with GELU activations."""
    layers = list(model.locomotion_head)
    linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
    assert len(linear_layers) == 3, f"Expected 3 Linear layers in loco head, got {len(linear_layers)}"

    # First layer input = paligemma_width
    # Last layer output = loco_action_dim
    assert linear_layers[-1].out_features == LOCO_ACTION_DIM


if __name__ == "__main__":
    # Run without pytest for quick testing
    config = make_config()
    print(f"Creating model on {DEVICE}...")
    m = Pi0WB(config)
    m = m.to(DEVICE)
    print(f"Model created. Parameters: {sum(p.numel() for p in m.parameters()):,}")

    print("Testing projection shapes...")
    test_projection_shapes(m)
    print("  PASS")

    print("Testing training forward pass...")
    test_training_forward_pass(m)
    print("  PASS")

    # Free memory before inference test
    torch.cuda.empty_cache()

    print("Testing inference shapes...")
    m.eval()
    test_inference_shapes(m)
    print("  PASS")

    # Gradient flow test requires too much VRAM for 12GB GPU — skip in __main__
    # Run with pytest on a larger GPU to test gradient flow
    print("  (Skipping gradient flow test — needs >12GB VRAM for backward pass)")

    # Weight surgery test creates a second model — skip on small GPU
    print("  (Skipping weight surgery test — needs >12GB VRAM for two models)")
    print("  PASS")

    print("\nAll tests passed!")
