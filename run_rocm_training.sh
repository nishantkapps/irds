#!/bin/bash

echo "=== ROCm Training Script ==="

# Set ROCm environment for all GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PLATFORM=amd

echo "Environment variables set:"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "ROCM_PATH: $ROCM_PATH"

# Check ROCm status
echo "=== ROCm Status ==="
rocm-smi

# Test PyTorch GPU detection
echo "=== PyTorch GPU Detection ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        try:
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        except Exception as e:
            print(f'GPU {i}: ERROR - {e}')
else:
    print('No GPUs available')
"

# Run training with environment variables
echo "=== Starting Training ==="
python train_rocm_fixed.py --config config_high_accuracy.yaml

echo "Training completed!"
