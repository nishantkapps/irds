#!/usr/bin/env python3
"""
Check GPU usage and training efficiency
"""

import torch
import time
import psutil
import os

def check_gpu_usage():
    """Check if GPU is being used effectively"""
    print("=== GPU Usage Check ===")
    
    # Check if CUDA/ROCm is available
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU tensor operations
        print("\n=== GPU Performance Test ===")
        start_time = time.time()
        
        # Create large tensors on GPU
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        
        # Matrix multiplication
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        
        gpu_time = time.time() - start_time
        print(f"GPU Matrix multiplication (1000x1000): {gpu_time:.4f} seconds")
        
        # Check GPU memory usage
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        return True
    else:
        print("❌ GPU not available - using CPU")
        return False

def check_system_resources():
    """Check system resources"""
    print("\n=== System Resources ===")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # Check if running in SLURM
    if 'SLURM_JOB_ID' in os.environ:
        print(f"SLURM Job ID: {os.environ['SLURM_JOB_ID']}")
        print(f"SLURM Node: {os.environ.get('SLURM_NODELIST', 'Unknown')}")
    
    # Check ROCm environment
    print(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'Not set')}")
    print(f"ROCM_PATH: {os.environ.get('ROCM_PATH', 'Not set')}")

def suggest_optimizations():
    """Suggest optimizations for faster training"""
    print("\n=== Optimization Suggestions ===")
    
    if torch.cuda.is_available():
        print("✅ GPU is available - good!")
        print("💡 Suggestions:")
        print("  - Use smaller batch size (8-16) for better GPU utilization")
        print("  - Use mixed precision training if available")
        print("  - Reduce sequence length if possible")
        print("  - Use fewer files for faster experiments")
    else:
        print("❌ GPU not available - training will be slow on CPU")
        print("💡 Suggestions:")
        print("  - Check ROCm installation")
        print("  - Set HIP_VISIBLE_DEVICES=0")
        print("  - Use SLURM with GPU allocation")
        print("  - Reduce data size for faster experiments")

def estimate_training_time():
    """Estimate training time based on current setup"""
    print("\n=== Training Time Estimation ===")
    
    if torch.cuda.is_available():
        print("With GPU (ROCm):")
        print("  - 100 epochs: ~2-4 hours")
        print("  - 150 epochs: ~3-6 hours")
        print("  - 200 epochs: ~4-8 hours")
    else:
        print("With CPU only:")
        print("  - 100 epochs: ~8-16 hours")
        print("  - 150 epochs: ~12-24 hours")
        print("  - 200 epochs: ~16-32 hours")

if __name__ == "__main__":
    print("=== GPU Usage and Training Efficiency Check ===")
    
    check_system_resources()
    gpu_available = check_gpu_usage()
    suggest_optimizations()
    estimate_training_time()
    
    if not gpu_available:
        print("\n⚠️  WARNING: GPU not detected!")
        print("Training will be very slow on CPU.")
        print("Check ROCm installation and environment variables.")
