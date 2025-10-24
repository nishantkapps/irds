#!/bin/bash

echo "=== Multi-GPU Training Script ==="

# Set ROCm environment
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PLATFORM=amd

echo "Environment variables set:"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "ROCM_PATH: $ROCM_PATH"

# Check available GPUs
echo "=== Available GPUs ==="
rocm-smi

# Test each GPU
echo "=== Testing GPUs ==="
for gpu_id in 0 1 2 3; do
    echo "Testing GPU $gpu_id..."
    python -c "
import torch
import os
os.environ['HIP_VISIBLE_DEVICES'] = '$gpu_id'
if torch.cuda.is_available():
    print(f'GPU $gpu_id: {torch.cuda.get_device_name(0)}')
    try:
        test_tensor = torch.randn(100, 100).cuda()
        print(f'GPU $gpu_id: SUCCESS')
    except Exception as e:
        print(f'GPU $gpu_id: FAILED - {e}')
else:
    print(f'GPU $gpu_id: Not available')
"
done

# Run training on specific GPU
echo "=== Starting Training ==="
echo "Choose GPU to use:"
echo "1. GPU 0"
echo "2. GPU 1" 
echo "3. GPU 2"
echo "4. GPU 3"
echo "5. All GPUs"

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "Using GPU 0"
        python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --gpu 0
        ;;
    2)
        echo "Using GPU 1"
        python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --gpu 1
        ;;
    3)
        echo "Using GPU 2"
        python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --gpu 2
        ;;
    4)
        echo "Using GPU 3"
        python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --gpu 3
        ;;
    5)
        echo "Using all GPUs"
        python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --all-gpus
        ;;
    *)
        echo "Invalid choice, using GPU 0"
        python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --gpu 0
        ;;
esac

echo "Training completed!"
