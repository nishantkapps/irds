#!/usr/bin/env python3
"""
Test PyTorch installation without conflicts
"""

import sys
import os

def test_pytorch_import():
    """Test PyTorch import without conflicts"""
    print("=== Testing PyTorch Import ===")
    
    try:
        # Clear any cached modules
        modules_to_clear = [mod for mod in sys.modules.keys() if 'torch' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        print("Cleared cached PyTorch modules")
        
        # Fresh import
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"Version: {torch.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\n=== Testing GPU Availability ===")
    
    try:
        import torch
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test GPU operations
            try:
                a = torch.randn(100, 100).cuda()
                b = torch.randn(100, 100).cuda()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                print("✅ GPU operations: SUCCESS")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                return True
            except Exception as e:
                print(f"❌ GPU operations failed: {e}")
                return False
        else:
            print("❌ GPU not available")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_training_imports():
    """Test training module imports"""
    print("\n=== Testing Training Module Imports ===")
    
    try:
        # Test basic imports
        import torch
        import pandas as pd
        import sklearn
        import matplotlib
        import joblib
        
        print("✅ Basic imports: SUCCESS")
        
        # Test PyTorch imports
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        
        print("✅ PyTorch imports: SUCCESS")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    print("=== PyTorch Clean Test ===")
    
    # Test PyTorch import
    if not test_pytorch_import():
        print("❌ PyTorch import failed - need to fix installation")
        return
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    # Test training imports
    imports_ok = test_training_imports()
    
    print(f"\n=== Test Results ===")
    print(f"PyTorch import: {'✅' if test_pytorch_import() else '❌'}")
    print(f"GPU available: {'✅' if gpu_available else '❌'}")
    print(f"Training imports: {'✅' if imports_ok else '❌'}")
    
    if gpu_available and imports_ok:
        print("\n✅ All tests passed! Ready for training.")
        print("Run: python train_with_config.py --config config_very_fast.yaml")
    else:
        print("\n❌ Some tests failed. Need to fix issues.")
        print("Run: ./setup_clean_pytorch.sh")

if __name__ == "__main__":
    main()
