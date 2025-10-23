#!/usr/bin/env python3
"""
Train CLIP Gesture Model with YAML Configuration
Allows easy experimentation with different hyperparameters
"""

import yaml
import argparse
import os
from clip_gesture_model import prepare_clip_gesture_data, train_clip_gesture_model, check_rocm_availability

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train CLIP Gesture Model with Configuration')
    parser.add_argument('--config', type=str, default='training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== CLIP Gesture Model Training with Configuration ===")
    print(f"Config file: {args.config}")
    print(f"Experiment: {args.experiment or config['experiment']['name']}")
    
    # Set device
    hardware_config = config.get('hardware', {})
    device_setting = hardware_config.get('device', 'auto')
    
    if device_setting == 'auto':
        if check_rocm_availability():
            device = 'cuda'
            print(f"Using AMD GPU with ROCm: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("ROCm not available, using CPU")
    else:
        device = device_setting
    
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n=== Preparing Data ===")
    data_config = config.get('data', {})
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data(
        folder_path=data_config.get('folder_path', 'data'),
        max_files=data_config.get('max_files', 200),
        sequence_length=data_config.get('sequence_length', 15)
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(gesture_names)}")
    print(f"Gesture names: {gesture_names}")
    
    # Train model
    print("\n=== Training Model ===")
    training_config = config.get('training', {})
    
    model, scaler, gesture_names, gesture_descriptions = train_clip_gesture_model(
        X, y, gesture_names, gesture_descriptions,
        sequence_length=data_config.get('sequence_length', 15),
        batch_size=training_config.get('batch_size', 16),
        num_epochs=training_config.get('num_epochs', 100),
        learning_rate=training_config.get('learning_rate', 0.0005),
        device=device
    )
    
    print("\n=== Training Completed ===")
    output_config = config.get('output', {})
    print(f"Model saved as: {output_config.get('model_save_path', 'clip_gesture_model.pth')}")
    print(f"Scaler saved as: {output_config.get('scaler_save_path', 'clip_gesture_scaler.pkl')}")
    print(f"Info saved as: {output_config.get('info_save_path', 'clip_gesture_info.json')}")

if __name__ == "__main__":
    import torch
    main()
