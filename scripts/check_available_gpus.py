#!/usr/bin/env python3
"""
Check which GPUs are actually available and working
"""

import torch
import os

def check_available_gpus():
    """Check which GPUs are available and working"""
    print("=== GPU Availability Check ===")
    
    if not torch.cuda.is_available():
        print("❌ No GPUs available")
        return []
    
    available_gpus = []
    total_gpus = torch.cuda.device_count()
    
    print(f"Total GPUs detected: {total_gpus}")
    
    for gpu_id in range(total_gpus):
        try:
            print(f"\nTesting GPU {gpu_id}...")
            
            # Test GPU
            test_tensor = torch.randn(10, 10).cuda(gpu_id)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            
            print(f"✅ GPU {gpu_id}: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            
            available_gpus.append(gpu_id)
            
        except Exception as e:
            print(f"❌ GPU {gpu_id}: FAILED - {e}")
    
    return available_gpus

def test_gpu_operations(gpu_id):
    """Test GPU operations on specific GPU"""
    print(f"\n=== Testing GPU {gpu_id} Operations ===")
    
    try:
        # Set environment for specific GPU
        os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Test basic operations
        a = torch.randn(100, 100).cuda(gpu_id)
        b = torch.randn(100, 100).cuda(gpu_id)
        c = torch.matmul(a, b)
        
        print(f"✅ GPU {gpu_id}: Basic operations SUCCESS")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id}: Operations FAILED - {e}")
        return False

def main():
    print("=== GPU Availability and Functionality Check ===")
    
    # Check available GPUs
    available_gpus = check_available_gpus()
    
    if not available_gpus:
        print("\n❌ No working GPUs found")
        print("Training will use CPU")
        return
    
    print(f"\n✅ Available GPUs: {available_gpus}")
    
    # Test operations on each GPU
    working_gpus = []
    for gpu_id in available_gpus:
        if test_gpu_operations(gpu_id):
            working_gpus.append(gpu_id)
    
    print(f"\n✅ Working GPUs: {working_gpus}")
    
    if working_gpus:
        print(f"\n=== Recommendations ===")
        print(f"Use GPU {working_gpus[0]} for training:")
        print(f"python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --gpu {working_gpus[0]}")
        
        if len(working_gpus) > 1:
            print(f"\nOr use all working GPUs:")
            gpu_list = ",".join(map(str, working_gpus))
            print(f"HIP_VISIBLE_DEVICES={gpu_list} python train_pytorch_multi_gpu.py --config config_high_accuracy.yaml --all-gpus")
    else:
        print("\n❌ No working GPUs found")
        print("Check ROCm installation and driver compatibility")

if __name__ == "__main__":
    main()
