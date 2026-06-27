#!/bin/bash
# Setup script for SONIC stub training on Huawei Ascend NPU server.
# This script prepares the environment for training without Isaac Sim.
#
# Prerequisites:
#   - Huawei CANN toolkit installed
#   - torch_npu installed and working (torch.npu.is_available() == True)
#   - Python 3.10+
#
# Usage:
#   bash scripts/setup_npu_training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== SONIC NPU Training Setup ==="
echo "Repo root: $REPO_ROOT"

# 1. Check Python version
echo ""
echo "--- Checking Python version ---"
python3 -c "
import sys
assert sys.version_info >= (3, 10), f'Python 3.10+ required, got {sys.version}'
print(f'Python {sys.version} OK')
"

# 2. Check torch + NPU
echo ""
echo "--- Checking PyTorch + NPU ---"
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
try:
    import torch_npu
    print(f'torch_npu imported OK')
    if torch.npu.is_available():
        print(f'NPU available: {torch.npu.get_device_name(0)}')
        print(f'NPU count: {torch.npu.device_count()}')
    else:
        print('WARNING: torch.npu.is_available() = False')
        print('Training will fall back to CPU')
except ImportError:
    print('WARNING: torch_npu not installed')
    print('Training will fall back to CPU')
"

# 3. Install gear_sonic with training dependencies
echo ""
echo "--- Installing gear_sonic[training] ---"
cd "$REPO_ROOT"

# Install training deps (skip smpl_sim if it fails — optional for stub mode)
pip install -e "gear_sonic[training]" || {
    echo "Full training install failed, trying core deps only..."
    pip install -e "gear_sonic"
    pip install hydra-core==1.3.2 trl==0.28.0 "transformers>=4.56.2" "accelerate>=1.3.0" tensorboard wandb
}

# Install additional deps that may be missing
pip install gymnasium filelock pyyaml rich kornia 2>/dev/null || true

# 4. Download sample motion data
echo ""
echo "--- Downloading motion data ---"
if [ -d "data/motion_lib_bones_seed/robot_filtered" ]; then
    echo "Motion data already exists, skipping download"
else
    echo "Downloading sample motion data..."
    python3 download_from_hf.py --sample --output-dir "$REPO_ROOT" || {
        echo "WARNING: Failed to download sample data"
        echo "You may need to download manually:"
        echo "  python3 download_from_hf.py --sample"
    }
fi

# 5. Download training checkpoint (optional)
echo ""
echo "--- Downloading training checkpoint ---"
if [ -f "sonic_release/last.pt" ]; then
    echo "Training checkpoint already exists, skipping download"
else
    echo "Downloading training checkpoint..."
    python3 download_from_hf.py --training --no-smpl --output-dir "$REPO_ROOT" 2>/dev/null || {
        echo "WARNING: Failed to download training checkpoint"
        echo "Training will start from scratch (no pretrained weights)"
    }
fi

# 6. Verify setup
echo ""
echo "--- Verifying setup ---"
python3 -c "
import torch
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot
from gear_sonic.trl.utils.torch_transform import quat_mul, quat_inv, quat_apply
from gear_sonic.isaac_utils import rotations
print('All core imports OK')

# Check motion data
import os
motion_path = 'data/motion_lib_bones_seed/robot_filtered'
if os.path.exists(motion_path):
    import glob
    files = glob.glob(os.path.join(motion_path, '**/*.pkl'), recursive=True)
    print(f'Motion data: {len(files)} PKL files found')
else:
    print(f'WARNING: Motion data not found at {motion_path}')
    print('Download with: python3 download_from_hf.py --sample')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start stub training:"
echo "  SONIC_STUB_ENV=1 python gear_sonic/train_agent_trl.py +exp=stub_train num_envs=64 headless=True use_wandb=False"
echo ""
echo "To start with fewer envs (for testing):"
echo "  SONIC_STUB_ENV=1 python gear_sonic/train_agent_trl.py +exp=stub_train num_envs=4 headless=True use_wandb=False"
