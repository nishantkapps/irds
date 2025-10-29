#!/usr/bin/env python3
"""
Train CLIP Gesture Model - PyTorch Only Version
Avoids NumPy operations that can cause GPU conflicts

IMPORTANT: GPU context must be initialized immediately after importing torch
to avoid segmentation faults on ROCm systems. Do not move the GPU warm-up code.
"""

# CRITICAL: Import torch first and warm up GPU immediately
# This prevents segmentation faults on ROCm/AMD GPU systems
import torch

# GPU warm-up: Initialize GPU context immediately after torch import
if torch.cuda.is_available():
    # This early GPU operation prevents segmentation faults later
    _ = torch.tensor([1.0]).to('cuda:1')

# Now import everything else
import yaml, argparse, sys, os
from pathlib import Path

# Add project paths to system path
path_to_model = Path(__file__).parent.parent
sys.path.insert(0, str(path_to_model))
path_to_irds = Path(__file__).parent.parent.parent
sys.path.insert(0, str(path_to_irds))

# Import project modules
from clip_gesture_model_pytorch import prepare_clip_gesture_data_pytorch, train_clip_gesture_model_pytorch
from utils import get_data_path, get_config_path, setup_logger
from utils.benchmark import GPUBenchmark

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    
    # If path doesn't exist, try to find it using utility
    if not os.path.exists(config_path):
        config_file = get_config_path() / 'config_high_accuracy.yaml'
        if config_file.exists():
            config_path = str(config_file)
        else:
            # Fallback to default config
            config_file = get_config_path() / 'config.yaml'
            if config_file.exists():
                config_path = str(config_file)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train CLIP Gesture Model - PyTorch Only')
    parser.add_argument('--config', type=str, default='../../config/config_high_accuracy.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logger from config
    logger = setup_logger(config)
    
    # Initialize benchmark
    experiment_name = config.get('experiment', {}).get('name', 'clip_gesture_training')
    benchmark = GPUBenchmark(experiment_name)
    
    logger.info("=== CLIP Gesture Model Training - PyTorch Only ===")
    logger.info(f"Config file: {args.config}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Experiment: {experiment_name}")
    
    # Set ROCm environment variables
    os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['ROCM_PATH'] = '/opt/rocm'
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['HIP_PLATFORM'] = 'amd'
    
    # Set device
    if torch.cuda.is_available():
        device = 'cuda:1'
        logger.debug(f"Using GPU: {torch.cuda.get_device_name(1)}")
        logger.debug(f"GPU Memory: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        logger.debug("CUDA not available, using CPU")
    
    logger.debug(f"Device: {device}")
    
    # Record initial memory state
    if device.startswith('cuda'):
        benchmark.record_memory('initial', device)
    
    # Prepare data with PyTorch only
    logger.info("=== Preparing Data ===")
    benchmark.start_timer('data_loading')
    data_config = config.get('data', {})
    logger.info(f"Folder path: {data_config.get('folder_path', 'data')}")
    logger.info(f"Max files: {data_config.get('max_files', 'all')}")
    logger.info(f"Sequence length: {data_config.get('sequence_length', 15)}")
    
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data_pytorch(
        folder_path=data_config.get('folder_path', str(get_data_path())),
        max_files=data_config.get('max_files', None),  # Use all files by default
        sequence_length=data_config.get('sequence_length', 15)
    )
    benchmark.stop_timer('data_loading')
    logger.info("Data preparation completed")
    
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Number of classes: {len(gesture_names)}")
    logger.info(f"Gesture names: {gesture_names}")
    
    # Train model with PyTorch only
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
        model_architecture=model_architecture,
        benchmark=benchmark
    )
    logger.info("Training completed successfully")
    
    # Finalize and save benchmark
    benchmark.finalize()
    benchmark_path = benchmark.save_report()
    logger.info(f"Benchmark report saved to: {benchmark_path}")
    
    # Print benchmark summary and save text report
    text_report_path = benchmark.print_summary()
    
    # Generate benchmark image for presentation
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        from generate_benchmark_image import create_benchmark_image
        image_path = create_benchmark_image(benchmark_path)
        logger.info(f"Benchmark image saved to: {image_path}")
    except Exception as e:
        logger.info(f"Could not generate benchmark image: {e}")
        logger.info("Install pillow to generate images: pip install pillow")
    
    logger.info("=== Training Completed ===")
    logger.info("Model saved as: outputs/clip_gesture_model_pytorch.pth")
    logger.info("Scaler saved as: outputs/clip_gesture_scaler_pytorch.json")
    logger.info("Info saved as: outputs/clip_gesture_info_pytorch.json")

if __name__ == "__main__":
    main()
