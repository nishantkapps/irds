#!/usr/bin/env python3
"""
Debug GPU usage in training and force GPU allocation
"""

import torch
import os
import sys

def debug_gpu_status():
    """Debug current GPU status"""
    print("=== GPU Status Debug ===")
    
    # Check environment variables
    print("Environment variables:")
    print(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'Not set')}")
    print(f"ROCM_PATH: {os.environ.get('ROCM_PATH', 'Not set')}")
    print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}")
    print(f"HIP_PLATFORM: {os.environ.get('HIP_PLATFORM', 'Not set')}")
    
    # Check PyTorch CUDA
    print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU allocation
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            return True
        except Exception as e:
            print(f"GPU allocation failed: {e}")
            return False
    else:
        print("❌ GPU not available")
        return False

def force_gpu_environment():
    """Force GPU environment variables"""
    print("\n=== Forcing GPU Environment ===")
    
    # Set environment variables
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['ROCM_PATH'] = '/opt/rocm'
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['HIP_PLATFORM'] = 'amd'
    
    print("Environment variables set")
    
    # Reload torch to pick up new environment
    if 'torch' in sys.modules:
        del sys.modules['torch']
    
    import torch
    print(f"PyTorch reloaded, CUDA available: {torch.cuda.is_available()}")

def test_gpu_allocation():
    """Test GPU memory allocation"""
    print("\n=== Testing GPU Allocation ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ GPU not available")
            return False
        
        # Create tensors on GPU
        print("Creating tensors on GPU...")
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Perform operations
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        print(f"After operations - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print("✅ GPU allocation test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ GPU allocation test failed: {e}")
        return False

def create_gpu_forced_training():
    """Create training script that forces GPU usage"""
    training_script = '''#!/usr/bin/env python3
"""
Force GPU usage in CLIP gesture model training
"""

import os
import sys
import torch
import torch.nn as nn

# Force GPU environment
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_PLATFORM'] = 'amd'

print("=== Forced GPU Training ===")
print(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES')}")

# Force device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test GPU allocation
    test_tensor = torch.randn(100, 100).cuda()
    print(f"Test GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Create a simple model to test GPU usage
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel().to(device)
    print(f"Model moved to device: {next(model.parameters()).device}")
    
    # Test model on GPU
    x = torch.randn(10, 100).to(device)
    y = model(x)
    print(f"Model output device: {y.device}")
    print(f"GPU memory after model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("✅ GPU training test: SUCCESS")
else:
    print("❌ GPU not available - training will be slow on CPU")

# Now run actual training
print("\\n=== Starting Actual Training ===")
try:
    from train_with_config import main
    main()
except Exception as e:
    print(f"Training failed: {e}")
    print("Try running: python train_with_config.py --config config_very_fast.yaml")
'''
    
    with open('gpu_forced_training.py', 'w') as f:
        f.write(training_script)
    
    print(f"\n=== Created GPU Forced Training Script ===")
    print("File: gpu_forced_training.py")
    print("Run: python gpu_forced_training.py")

def create_simple_gpu_test():
    """Create simple GPU test"""
    test_script = '''#!/usr/bin/env python3
import torch
import os

# Set environment
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCM_PATH'] = '/opt/rocm'

print("=== Simple GPU Test ===")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Force GPU allocation
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print("✅ GPU test: SUCCESS")
else:
    print("❌ GPU test: FAILED")
'''
    
    with open('simple_gpu_test.py', 'w') as f:
        f.write(test_script)
    
    print(f"\n=== Created Simple GPU Test ===")
    print("File: simple_gpu_test.py")
    print("Run: python simple_gpu_test.py")

def main():
    print("=== GPU Training Debug ===")
    
    # Debug current status
    debug_gpu_status()
    
    # Force GPU environment
    force_gpu_environment()
    
    # Test GPU allocation
    gpu_working = test_gpu_allocation()
    
    if gpu_working:
        print("\n✅ GPU is working - creating forced training script")
        create_gpu_forced_training()
        create_simple_gpu_test()
        
        print("\n=== Next Steps ===")
        print("1. Test: python simple_gpu_test.py")
        print("2. Train: python gpu_forced_training.py")
    else:
        print("\n❌ GPU still not working")
        print("Possible issues:")
        print("  - ROCm not properly installed")
        print("  - Wrong ROCm version")
        print("  - Need to restart Python session")
        print("  - Try: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")

if __name__ == "__main__":
    main()
