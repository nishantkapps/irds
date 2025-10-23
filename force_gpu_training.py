#!/usr/bin/env python3
"""
Force GPU usage for CLIP gesture model training
"""

import os
import sys
import torch

def force_gpu_environment():
    """Force GPU environment variables"""
    print("=== Setting GPU Environment ===")
    
    # Set environment variables
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['ROCM_PATH'] = '/opt/rocm'
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['HIP_PLATFORM'] = 'amd'
    
    print("Environment variables set:")
    print(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"ROCM_PATH: {os.environ.get('ROCM_PATH')}")

def test_gpu_force():
    """Force GPU test"""
    print("\n=== Forcing GPU Test ===")
    
    # Force CUDA availability check
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Force GPU operations
        try:
            # Create tensors on GPU
            a = torch.randn(100, 100).cuda()
            b = torch.randn(100, 100).cuda()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            print("✅ GPU operations: SUCCESS")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            return True
        except Exception as e:
            print(f"❌ GPU operations: FAILED - {e}")
            return False
    else:
        print("❌ GPU not available")
        return False

def run_training_with_gpu():
    """Run training with forced GPU usage"""
    print("\n=== Running Training with GPU ===")
    
    # Set environment
    force_gpu_environment()
    
    # Test GPU
    if test_gpu_force():
        print("✅ GPU is working, starting training...")
        
        # Run training with specific config
        import subprocess
        import sys
        
        print("Starting training with GPU...")
        try:
            # Run the training with fast config
            result = subprocess.run([
                sys.executable, 'train_with_config.py', 
                '--config', 'config_high_accuracy.yaml'
            ], check=True)
            print("✅ Training completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except FileNotFoundError:
            print("❌ train_with_config.py not found")
            print("Run: python train_with_config.py --config config_very_fast.yaml")
    else:
        print("❌ GPU not working, training will be slow on CPU")

if __name__ == "__main__":
    run_training_with_gpu()
