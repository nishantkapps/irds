#!/usr/bin/env python3
"""
ROCm-fixed training that handles GPU device ID issues
"""

import yaml
import argparse
import os
import torch
from clip_gesture_model_pytorch import prepare_clip_gesture_data_pytorch, train_clip_gesture_model_pytorch

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def safe_gpu_setup():
    """Safely set up GPU without device ID issues"""
    print("=== Safe GPU Setup ===")
    
    # Set ROCm environment first
    os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['ROCM_PATH'] = '/opt/rocm'
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['HIP_PLATFORM'] = 'amd'
    
    print("ROCm environment set")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 'cpu'
    
    # Get device count
    device_count = torch.cuda.device_count()
    print(f"PyTorch reports {device_count} GPUs")
    
    # Test each GPU safely
    working_gpu = None
    for gpu_id in range(device_count):
        try:
            print(f"Testing GPU {gpu_id}...")
            
            # Test basic operations
            test_tensor = torch.randn(10, 10).cuda(gpu_id)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            
            print(f"✅ GPU {gpu_id}: {gpu_name}")
            working_gpu = gpu_id
            break
            
        except Exception as e:
            print(f"❌ GPU {gpu_id}: FAILED - {e}")
    
    if working_gpu is not None:
        device = f'cuda:{working_gpu}'
        print(f"✅ Using GPU {working_gpu}")
        return device
    else:
        print("❌ No working GPUs found")
        return 'cpu'

def main():
    parser = argparse.ArgumentParser(description='ROCm-fixed GPU Training')
    parser.add_argument('--config', type=str, default='config_high_accuracy.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== ROCm-fixed GPU Training ===")
    print(f"Config file: {args.config}")
    
    # Safe GPU setup
    device = safe_gpu_setup()
    
    print(f"Using device: {device}")
    
    # Test device before training
    if device != 'cpu':
        try:
            test_tensor = torch.randn(100, 100).to(device)
            print(f"✅ Device test: SUCCESS")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        except Exception as e:
            print(f"❌ Device test failed: {e}")
            device = 'cpu'
            print("Falling back to CPU")
    
    # Prepare data
    print("\n=== Preparing Data ===")
    data_config = config.get('data', {})
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data_pytorch(
        folder_path=data_config.get('folder_path', 'data'),
        max_files=data_config.get('max_files', 200),
        sequence_length=data_config.get('sequence_length', 15)
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(gesture_names)}")
    
    # Train model
    print("\n=== Training Model ===")
    training_config = config.get('training', {})
    
    model, scaler, gesture_names, gesture_descriptions = train_clip_gesture_model_pytorch(
        X, y, gesture_names, gesture_descriptions,
        sequence_length=data_config.get('sequence_length', 15),
        batch_size=training_config.get('batch_size', 16),
        num_epochs=training_config.get('num_epochs', 100),
        learning_rate=training_config.get('learning_rate', 0.0005),
        device=device
    )
    
    print("\n=== Training Completed ===")
    print(f"Model saved as: clip_gesture_model_pytorch.pth")

if __name__ == "__main__":
    main()
