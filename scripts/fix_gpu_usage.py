#!/usr/bin/env python3
"""
Fix GPU usage issues for CLIP gesture model training
"""

import torch
import os
import subprocess

def check_gpu_availability():
    """Check if GPU is properly available"""
    print("=== GPU Availability Check ===")
    
    # Check PyTorch CUDA availability
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU tensor creation
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            print("✅ GPU tensor creation: SUCCESS")
            return True
        except Exception as e:
            print(f"❌ GPU tensor creation: FAILED - {e}")
            return False
    else:
        print("❌ GPU not available")
        return False

def check_environment_variables():
    """Check ROCm environment variables"""
    print("\n=== Environment Variables ===")
    
    env_vars = [
        'HIP_VISIBLE_DEVICES',
        'ROCM_PATH', 
        'HSA_OVERRIDE_GFX_VERSION',
        'HIP_PLATFORM'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def set_environment_variables():
    """Set proper ROCm environment variables"""
    print("\n=== Setting Environment Variables ===")
    
    env_commands = [
        "export HIP_VISIBLE_DEVICES=0",
        "export ROCM_PATH=/opt/rocm", 
        "export HSA_OVERRIDE_GFX_VERSION=10.3.0",
        "export HIP_PLATFORM=amd"
    ]
    
    for cmd in env_commands:
        print(f"Running: {cmd}")
        os.system(cmd)
    
    print("Environment variables set")

def test_gpu_after_env_setup():
    """Test GPU after setting environment variables"""
    print("\n=== Testing GPU After Environment Setup ===")
    
    # Reload torch to pick up new environment
    import importlib
    importlib.reload(torch)
    
    if torch.cuda.is_available():
        print("✅ GPU available after environment setup")
        
        # Test GPU operations
        try:
            # Create tensors on GPU
            a = torch.randn(100, 100).cuda()
            b = torch.randn(100, 100).cuda()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            print("✅ GPU operations: SUCCESS")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            return True
        except Exception as e:
            print(f"❌ GPU operations: FAILED - {e}")
            return False
    else:
        print("❌ GPU still not available")
        return False

def create_gpu_test_script():
    """Create a script to test GPU usage"""
    gpu_test_script = """#!/usr/bin/env python3
import torch
import time

print("=== GPU Test Script ===")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test GPU operations
    print("Testing GPU operations...")
    start_time = time.time()
    
    # Create large tensors
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    gpu_time = time.time() - start_time
    print(f"GPU Matrix multiplication time: {gpu_time:.4f} seconds")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("✅ GPU test: SUCCESS")
else:
    print("❌ GPU test: FAILED - No GPU available")
"""
    
    with open('gpu_test.py', 'w') as f:
        f.write(gpu_test_script)
    
    print(f"\n=== Created GPU Test Script ===")
    print("File: gpu_test.py")
    print("Run: python gpu_test.py")

def create_gpu_training_script():
    """Create a script to run training with proper GPU setup"""
    training_script = """#!/bin/bash

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
python gpu_test.py

if [ $? -eq 0 ]; then
    echo "GPU test passed, starting training..."
    python model/train/train_with_config.py --config config/config_very_fast.yaml
else
    echo "GPU test failed, check ROCm installation"
    exit 1
fi
"""
    
    with open('run_gpu_training.sh', 'w') as f:
        f.write(training_script)
    
    print(f"\n=== Created GPU Training Script ===")
    print("File: run_gpu_training.sh")
    print("Run: chmod +x run_gpu_training.sh && ./run_gpu_training.sh")

def main():
    print("=== GPU Usage Fix ===")
    
    # Check current GPU status
    gpu_available = check_gpu_availability()
    
    # Check environment variables
    check_environment_variables()
    
    # Set environment variables
    set_environment_variables()
    
    # Test GPU after environment setup
    gpu_working = test_gpu_after_env_setup()
    
    if gpu_working:
        print("\n✅ GPU is now working properly!")
        print("You can now run training with GPU acceleration")
    else:
        print("\n❌ GPU is still not working")
        print("Possible issues:")
        print("  - ROCm not properly installed")
        print("  - Wrong ROCm version")
        print("  - GPU not accessible")
        print("  - Need to restart Python session")
    
    # Create helper scripts
    create_gpu_test_script()
    create_gpu_training_script()
    
    print("\n=== Next Steps ===")
    print("1. Run: python gpu_test.py")
    print("2. If successful, run: ./run_gpu_training.sh")
    print("3. Monitor GPU usage: rocm-smi -l 1")

if __name__ == "__main__":
    main()
