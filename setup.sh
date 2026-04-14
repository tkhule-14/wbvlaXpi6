#!/bin/bash
# Setup script for Pi0.6-WB
set -e

echo "=== Pi0.6-WB Setup ==="

# 1. Clone exla-ai/fla (openpie-0.6 codebase)
if [ ! -d "exla-fla" ]; then
    echo "Cloning exla-ai/fla..."
    git clone https://github.com/exla-ai/fla.git exla-fla
else
    echo "exla-fla/ already exists, skipping clone"
fi

# 2. Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 3. Apply transformers_replace patches (required for correct SigLIP/PaliGemma behavior)
echo "Applying transformers_replace patches..."
TRANSFORMERS_DIR=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py "$TRANSFORMERS_DIR/models/gemma/modeling_gemma.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py "$TRANSFORMERS_DIR/models/gemma/configuration_gemma.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py "$TRANSFORMERS_DIR/models/paligemma/modeling_paligemma.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py "$TRANSFORMERS_DIR/models/siglip/modeling_siglip.py"
cp exla-fla/src/fla/models_pytorch/transformers_replace/models/siglip/check.py "$TRANSFORMERS_DIR/models/siglip/check.py"
echo "Patches applied to $TRANSFORMERS_DIR"

# 4. Download openpie-0.6 weights (optional — for weight surgery)
if [ ! -f "checkpoints/openpie-0.6/policy.safetensors" ]; then
    echo "To download openpie-0.6 weights, run:"
    echo "  mkdir -p checkpoints/openpie-0.6"
    echo "  huggingface-cli download exla-ai/openpie-0.6 --local-dir checkpoints/openpie-0.6"
fi

echo ""
echo "=== Setup complete ==="
echo "Run tests:  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH python pi06_wb/tests/test_forward_pass.py"
echo "Train:      python pi06_wb/scripts/train.py --config pi06_wb/configs/g2.yaml"
