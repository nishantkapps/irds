#!/usr/bin/env python3
"""
Fix GPU driver issues for Radeon RX 7900 XTX
"""

import subprocess
import sys
import os

def check_rocm_installation():
    """Check ROCm installation and compatibility"""
    print("=== ROCm Installation Check ===")
    
    try:
        # Check ROCm version
        result = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"ROCm version: {result.stdout}")
        else:
            print("❌ ROCm not found or not working")
            return False
        
        # Check GPU info
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"GPU info: {result.stdout}")
        else:
            print("❌ rocm-smi failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ ROCm check failed: {e}")
        return False

def check_pytorch_rocm_compatibility():
    """Check PyTorch ROCm compatibility"""
    print("\n=== PyTorch ROCm Compatibility ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"ROCm version: {torch.version.hip}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        return False

def fix_pytorch_installation():
    """Fix PyTorch installation for Radeon RX 7900 XTX"""
    print("\n=== Fixing PyTorch Installation ===")
    
    print("1. Uninstalling current PyTorch...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'], check=False)
    
    print("2. Clearing pip cache...")
    subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], check=False)
    
    print("3. Installing PyTorch with ROCm 5.5...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/rocm5.5.1',
            '--no-cache-dir'
        ], check=True)
        print("✅ PyTorch installation successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False

def test_gpu_operations():
    """Test GPU operations safely"""
    print("\n=== Testing GPU Operations ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Test 1: Basic tensor creation
        print("Test 1: Basic tensor creation...")
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            print("✅ Basic tensor creation: SUCCESS")
        except Exception as e:
            print(f"❌ Basic tensor creation failed: {e}")
            return False
        
        # Test 2: Small matrix operation
        print("Test 2: Small matrix operation...")
        try:
            a = torch.randn(10, 10).cuda()
            b = torch.randn(10, 10).cuda()
            c = torch.matmul(a, b)
            print("✅ Small matrix operation: SUCCESS")
        except Exception as e:
            print(f"❌ Small matrix operation failed: {e}")
            return False
        
        # Test 3: Memory allocation
        print("Test 3: Memory allocation...")
        try:
            large_tensor = torch.randn(1000, 1000).cuda()
            print(f"✅ Memory allocation: SUCCESS")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            return True
        except Exception as e:
            print(f"❌ Memory allocation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def create_gpu_working_script():
    """Create script that ensures GPU works"""
    script_content = '''#!/usr/bin/env python3
"""
Ensure GPU works for training
"""

import os
import torch

# Set environment variables
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_PLATFORM'] = 'amd'

print("=== GPU Training Setup ===")

# Test GPU
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    
    # Test GPU operations
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        print("✅ GPU tensor creation: SUCCESS")
        
        # Test larger operation
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = torch.matmul(a, b)
        print("✅ GPU matrix multiplication: SUCCESS")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        print("✅ GPU is working, starting training...")
        
        # Run training
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, 'train_with_config.py', 
            '--config', 'config_high_accuracy.yaml'
        ])
        
        if result.returncode == 0:
            print("✅ Training completed successfully")
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        print("Need to fix GPU driver issues")
else:
    print("❌ GPU not available")
'''
    
    with open('gpu_working_training.py', 'w') as f:
        f.write(script_content)
    
    print(f"\n=== Created GPU Working Script ===")
    print("File: gpu_working_training.py")
    print("Run: python gpu_working_training.py")

def main():
    print("=== GPU Driver Fix for Radeon RX 7900 XTX ===")
    
    # Check ROCm
    rocm_ok = check_rocm_installation()
    
    # Check PyTorch
    pytorch_ok = check_pytorch_rocm_compatibility()
    
    if not rocm_ok or not pytorch_ok:
        print("\n=== Fixing Installation ===")
        fix_pytorch_installation()
    
    # Test GPU operations
    gpu_ok = test_gpu_operations()
    
    if gpu_ok:
        print("\n✅ GPU is working!")
        create_gpu_working_script()
        print("\n=== Next Steps ===")
        print("Run: python gpu_working_training.py")
    else:
        print("\n❌ GPU still not working")
        print("Possible solutions:")
        print("1. Try different PyTorch version")
        print("2. Use conda instead of pip")
        print("3. Check ROCm driver compatibility")
        print("4. Use SLURM with proper GPU allocation")

if __name__ == "__main__":
    main()
