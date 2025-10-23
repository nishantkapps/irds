#!/bin/bash

echo "=== GPU Training Script ==="

# Set ROCm environment variables
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PLATFORM=amd

echo "Environment variables set:"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "ROCM_PATH: $ROCM_PATH"

# Test GPU first
echo "Testing GPU..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    a = torch.randn(100, 100).cuda()
    print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
    print('✅ GPU test: SUCCESS')
else:
    print('❌ GPU test: FAILED')
"

if [ $? -eq 0 ]; then
    echo "GPU test passed, starting training..."
    python train_with_config.py --config config_very_fast.yaml
else
    echo "GPU test failed, check ROCm installation"
    exit 1
fi
