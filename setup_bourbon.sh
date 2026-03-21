#!/usr/bin/env bash
set -euo pipefail

# Setup for bourbon (Blackwell GPUs sm_120)

echo "=== Setting up bourbon environment ==="

# Use bourbon-specific pyproject
cp pyproject.bourbon.toml pyproject.toml
rm -f uv.lock

# Sync
uv sync

# Set LD_LIBRARY_PATH to include torch's bundled libs (includes nvshmem)
TORCH_LIB=$(uv run --no-sync python -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent / 'lib')")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# Verify
echo ""
echo "=== Verifying ==="
uv run --no-sync python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Archs: {torch.cuda.get_arch_list()}')
t = torch.zeros(1).cuda()
print(f'CUDA test: OK')
"

echo ""
echo "=== Done! ==="
echo ""
echo "Before training, always run:"
echo "  export LD_LIBRARY_PATH=${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"
