#!/usr/bin/env python3
"""
Train CLIP Gesture Model - Multi-GPU Support
Allows specifying which GPU to use
"""

# CRITICAL: Import torch first and warm up GPU immediately
# This prevents segmentation faults on ROCm/AMD GPU systems
import torch

# GPU warm-up: Initialize GPU context immediately after torch import
if torch.cuda.is_available():
    # This early GPU operation prevents segmentation faults later
    _ = torch.tensor([1.0]).cuda()

import yaml, argparse, os, sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.clip_gesture_model_pytorch import prepare_clip_gesture_data_pytorch, train_clip_gesture_model_pytorch
from utils import get_logger, setup_logger, get_data_path

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_gpu_environment(gpu_id: int = 0):
    """Set GPU environment variables for ROCm/AMD GPUs"""
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    logger = get_logger()
    logger.info(f"Set HIP_VISIBLE_DEVICES={gpu_id}")

def main():
    parser = argparse.ArgumentParser(description='Train CLIP Gesture Model - Multi-GPU Support')
    parser.add_argument('--config', type=str, default='../../config/config_high_accuracy.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (0, 1, 2, 3)')
    parser.add_argument('--all-gpus', action='store_true',
                       help='Use all available GPUs (experimental)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logger from config
    logger = setup_logger(config)
    
    logger.info("=== CLIP Gesture Model Training - Multi-GPU ===")
    logger.info(f"Config file: {args.config}")
    logger.info(f"GPU ID: {args.gpu}")
    
    # Set GPU environment
    if args.all_gpus:
        logger.info("Using all available GPUs")
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
    else:
        logger.info(f"Using GPU {args.gpu}")
        set_gpu_environment(args.gpu)
    
    # Set device with specific GPU
    if torch.cuda.is_available():
        if args.all_gpus:
            device = 'cuda'
            logger.info(f"Using all GPUs: {torch.cuda.device_count()} devices")
        else:
            device = f'cuda:{args.gpu}'
            logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
            gpu_memory = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        device = 'cpu'
        logger.info("CUDA not available, using CPU")
    
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("=== Preparing Data ===")
    data_config = config.get('data', {})
    folder_path = data_config.get('folder_path')
    if folder_path is None:
        folder_path = get_data_path()
    
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data_pytorch(
        folder_path=folder_path,
        max_files=data_config.get('max_files'),
        sequence_length=data_config.get('sequence_length', 15)
    )
    
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Number of classes: {len(gesture_names)}")
    logger.info(f"Gesture names: {gesture_names}")
    
    # Train model with specific GPU
    logger.info("=== Training Model ===")
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    model_architecture = model_config.get('architecture', 'medium')
    
    logger.info(f"Model architecture: {model_architecture}")
    logger.info(f"Sequence length: {data_config.get('sequence_length', 15)}")
    logger.info(f"Batch size: {training_config.get('batch_size', 16)}")
    logger.info(f"Epochs: {training_config.get('num_epochs', 100)}")
    logger.info(f"Learning rate: {training_config.get('learning_rate', 0.0005)}")
    
    model, scaler, gesture_names, gesture_descriptions = train_clip_gesture_model_pytorch(
        X, y, gesture_names, gesture_descriptions,
        sequence_length=data_config.get('sequence_length', 15),
        batch_size=training_config.get('batch_size', 16),
        num_epochs=training_config.get('num_epochs', 100),
        learning_rate=training_config.get('learning_rate', 0.0005),
        device=device,
        model_architecture=model_architecture
    )
    logger.info("Training completed successfully")
    
    logger.info("=== Training Completed ===")
    logger.info("Model saved as: outputs/clip_gesture_model_pytorch.pth")
    logger.info("Scaler saved as: outputs/clip_gesture_scaler_pytorch.json")
    logger.info("Info saved as: outputs/clip_gesture_info_pytorch.json")

if __name__ == "__main__":
    main()
