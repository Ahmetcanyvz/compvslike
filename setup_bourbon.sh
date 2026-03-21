#!/usr/bin/env bash
set -euo pipefail

# Setup script for bourbon (Blackwell GPUs, sm_120)
# Installs everything with stable torch first, then replaces with nightly cu128

echo "=== Step 1: Install all dependencies with stable torch ==="
uv sync

echo ""
echo "=== Step 2: Replace torch with nightly cu126 (Blackwell support) ==="
uv pip install --force-reinstall --no-deps \
    torch --pre --index-url https://download.pytorch.org/whl/nightly/cu126

echo ""
echo "=== Step 3: Verify ==="
uv run --no-sync python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Arch list: {torch.cuda.get_arch_list()}')
sm120 = any('sm_120' in a or 'sm_12' in a for a in torch.cuda.get_arch_list())
print(f'Blackwell (sm_120) supported: {sm120}')
"

echo ""
echo "=== Done! Use 'uv run --no-sync' instead of 'uv run' to avoid torch being reverted ==="
