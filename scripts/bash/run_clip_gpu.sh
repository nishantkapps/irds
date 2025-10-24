#!/bin/bash

echo "=== CLIP Gesture Model Training with AMD GPU ==="
echo "Start time: $(date)"

# Set ROCm environment variables
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PLATFORM=amd

# Check ROCm availability
echo "=== Checking ROCm Setup ==="
echo "ROCm version:"
rocm-smi --version

echo "GPU status:"
rocm-smi

echo "PyTorch ROCm test:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        print('ROCm test: SUCCESS - GPU tensor creation works')
        print(f'GPU name: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    except Exception as e:
        print(f'ROCm test: FAILED - {e}')
else:
    print('ROCm test: FAILED - torch.cuda.is_available() returned False')
"

echo ""
echo "=== Starting CLIP Model Training ==="
echo "Training start time: $(date)"

# Run the CLIP gesture model
python clip_gesture_model.py

echo ""
echo "=== Training Completed ==="
echo "End time: $(date)"

# Check output files
echo "=== Generated Files ==="
ls -la clip_*.pth clip_*.pkl clip_*.png clip_*.json 2>/dev/null || echo "No output files found"

echo "=== File Sizes ==="
du -h clip_gesture_model.pth 2>/dev/null || echo "Model file not found"
du -h clip_gesture_scaler.pkl 2>/dev/null || echo "Scaler file not found"

echo "=== Training Complete ==="
