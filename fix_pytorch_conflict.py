#!/usr/bin/env python3
"""
Fix PyTorch library conflicts and ROCm issues
"""

import subprocess
import sys
import os

def check_pytorch_installation():
    """Check current PyTorch installation"""
    print("=== PyTorch Installation Check ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"PyTorch ROCm version: {torch.version.hip}")
        
        # Check installation path
        print(f"PyTorch installation path: {torch.__file__}")
        
        return True
    except Exception as e:
        print(f"Error checking PyTorch: {e}")
        return False

def check_pip_packages():
    """Check installed packages"""
    print("\n=== Installed Packages ===")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        packages = result.stdout
        
        # Check for PyTorch related packages
        pytorch_packages = [line for line in packages.split('\n') if 'torch' in line.lower()]
        print("PyTorch related packages:")
        for pkg in pytorch_packages:
            print(f"  {pkg}")
            
        return pytorch_packages
    except Exception as e:
        print(f"Error checking packages: {e}")
        return []

def fix_pytorch_conflict():
    """Fix PyTorch library conflicts"""
    print("\n=== Fixing PyTorch Conflicts ===")
    
    print("1. Uninstalling all PyTorch packages...")
    uninstall_commands = [
        "pip uninstall torch torchvision torchaudio -y",
        "pip uninstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 -y",
        "pip uninstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -y"
    ]
    
    for cmd in uninstall_commands:
        print(f"Running: {cmd}")
        try:
            subprocess.run(cmd.split(), check=False)
        except:
            pass
    
    print("2. Clearing pip cache...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], check=False)
    except:
        pass
    
    print("3. Installing clean PyTorch with ROCm...")
    install_cmd = [
        sys.executable, '-m', 'pip', 'install', 
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/rocm5.6',
        '--no-cache-dir'
    ]
    
    print(f"Running: {' '.join(install_cmd)}")
    try:
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ PyTorch installation successful")
        else:
            print(f"❌ PyTorch installation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Installation error: {e}")

def test_clean_pytorch():
    """Test PyTorch after clean installation"""
    print("\n=== Testing Clean PyTorch ===")
    
    try:
        # Clear any cached modules
        if 'torch' in sys.modules:
            del sys.modules['torch']
        
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            
            # Test GPU operations
            try:
                a = torch.randn(100, 100).cuda()
                b = torch.randn(100, 100).cuda()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                print("✅ GPU operations: SUCCESS")
                return True
            except Exception as e:
                print(f"❌ GPU operations failed: {e}")
                return False
        else:
            print("❌ GPU not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def create_clean_environment_script():
    """Create script to set up clean environment"""
    script_content = """#!/bin/bash

echo "=== Clean PyTorch Environment Setup ==="

# Set ROCm environment
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PLATFORM=amd

echo "Environment variables set"

# Uninstall all PyTorch packages
echo "Uninstalling PyTorch packages..."
pip uninstall torch torchvision torchaudio -y

# Clear pip cache
echo "Clearing pip cache..."
pip cache purge

# Install clean PyTorch with ROCm
echo "Installing PyTorch with ROCm..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 --no-cache-dir

# Test installation
echo "Testing PyTorch..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    a = torch.randn(100, 100).cuda()
    b = torch.randn(100, 100).cuda()
    c = torch.matmul(a, b)
    print('GPU operations: SUCCESS')
else:
    print('GPU not available')
"

echo "Setup complete!"
"""
    
    with open('setup_clean_pytorch.sh', 'w') as f:
        f.write(script_content)
    
    print(f"\n=== Created Clean Environment Script ===")
    print("File: setup_clean_pytorch.sh")
    print("Run: chmod +x setup_clean_pytorch.sh && ./setup_clean_pytorch.sh")

def main():
    print("=== PyTorch Conflict Fix ===")
    
    # Check current installation
    check_pytorch_installation()
    check_pip_packages()
    
    # Fix conflicts
    fix_pytorch_conflict()
    
    # Test clean installation
    test_clean_pytorch()
    
    # Create setup script
    create_clean_environment_script()
    
    print("\n=== Next Steps ===")
    print("1. Run: chmod +x setup_clean_pytorch.sh")
    print("2. Run: ./setup_clean_pytorch.sh")
    print("3. Test: python gpu_test.py")
    print("4. Train: python train_with_config.py --config config_very_fast.yaml")

if __name__ == "__main__":
    main()
