#!/usr/bin/env python3
"""
Safe GPU training that only uses available GPUs
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

def find_working_gpu():
    """Find the first working GPU"""
    if not torch.cuda.is_available():
        return None
    
    total_gpus = torch.cuda.device_count()
    print(f"Checking {total_gpus} GPUs...")
    
    for gpu_id in range(total_gpus):
        try:
            print(f"Testing GPU {gpu_id}...")
            test_tensor = torch.randn(10, 10).cuda(gpu_id)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            print(f"✅ GPU {gpu_id}: {gpu_name}")
            return gpu_id
        except Exception as e:
            print(f"❌ GPU {gpu_id}: FAILED - {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Safe GPU Training')
    parser.add_argument('--config', type=str, default='config_high_accuracy.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Safe GPU Training ===")
    print(f"Config file: {args.config}")
    
    # Find working GPU
    working_gpu = find_working_gpu()
    
    if working_gpu is not None:
        device = f'cuda:{working_gpu}'
        print(f"✅ Using GPU {working_gpu}: {torch.cuda.get_device_name(working_gpu)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(working_gpu).total_memory / 1024**3:.1f} GB")
        
        # Set environment for this GPU
        os.environ['HIP_VISIBLE_DEVICES'] = str(working_gpu)
        os.environ['ROCM_PATH'] = '/opt/rocm'
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        os.environ['HIP_PLATFORM'] = 'amd'
        
        print(f"Set HIP_VISIBLE_DEVICES={working_gpu}")
        
        # Test GPU allocation
        try:
            test_tensor = torch.randn(100, 100).to(device)
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(working_gpu) / 1024**3:.2f} GB")
            print("✅ GPU allocation test: SUCCESS")
        except Exception as e:
            print(f"❌ GPU allocation test failed: {e}")
            device = 'cpu'
            print("Falling back to CPU")
    else:
        device = 'cpu'
        print("❌ No working GPUs found, using CPU")
    
    print(f"Using device: {device}")
    
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
