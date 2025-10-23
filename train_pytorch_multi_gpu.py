#!/usr/bin/env python3
"""
Train CLIP Gesture Model - Multi-GPU Support
Allows specifying which GPU to use
"""

import yaml
import argparse
import os
import torch
from clip_gesture_model_pytorch import prepare_clip_gesture_data_pytorch, train_clip_gesture_model_pytorch, check_rocm_availability

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_gpu_environment(gpu_id: int = 0):
    """Set GPU environment variables"""
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['ROCM_PATH'] = '/opt/rocm'
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['HIP_PLATFORM'] = 'amd'
    
    print(f"Set HIP_VISIBLE_DEVICES={gpu_id}")

def main():
    parser = argparse.ArgumentParser(description='Train CLIP Gesture Model - Multi-GPU Support')
    parser.add_argument('--config', type=str, default='config_high_accuracy.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (0, 1, 2, 3)')
    parser.add_argument('--all-gpus', action='store_true',
                       help='Use all available GPUs')
    args = parser.parse_args()
    
    # Set GPU environment
    if args.all_gpus:
        print("Using all available GPUs")
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
    else:
        print(f"Using GPU {args.gpu}")
        set_gpu_environment(args.gpu)
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== CLIP Gesture Model Training - Multi-GPU ===")
    print(f"Config file: {args.config}")
    print(f"GPU ID: {args.gpu}")
    
    # Set device with specific GPU
    if torch.cuda.is_available():
        if args.all_gpus:
            device = 'cuda'
            print(f"Using all GPUs: {torch.cuda.device_count()} devices")
        else:
            device = f'cuda:{args.gpu}'
            print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.1f} GB")
        
        # Test GPU allocation
        try:
            if args.all_gpus:
                test_tensor = torch.randn(100, 100).cuda()
            else:
                test_tensor = torch.randn(100, 100).to(device)
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print("✅ GPU allocation test: SUCCESS")
        except Exception as e:
            print(f"❌ GPU allocation test failed: {e}")
            device = 'cpu'
            print("Falling back to CPU")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Prepare data with PyTorch only
    print("\n=== Preparing Data (PyTorch Only) ===")
    data_config = config.get('data', {})
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data_pytorch(
        folder_path=data_config.get('folder_path', 'data'),
        max_files=data_config.get('max_files', 200),
        sequence_length=data_config.get('sequence_length', 15)
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(gesture_names)}")
    print(f"Gesture names: {gesture_names}")
    
    # Train model with specific GPU
    print("\n=== Training Model (Multi-GPU) ===")
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
    print(f"Scaler saved as: clip_gesture_scaler_pytorch.json")
    print(f"Info saved as: clip_gesture_info_pytorch.json")

if __name__ == "__main__":
    main()
