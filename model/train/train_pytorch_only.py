#!/usr/bin/env python3
"""
Train CLIP Gesture Model - PyTorch Only Version
Avoids NumPy operations that can cause GPU conflicts
"""

print("=== SCRIPT STARTING ===")

import yaml
print("+ yaml imported")

import argparse
print("+ argparse imported")

import os
print("+ os imported")

import torch
print("+ torch imported")

print("About to import from clip_gesture_model_pytorch...")
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from clip_gesture_model_pytorch import prepare_clip_gesture_data_pytorch, train_clip_gesture_model_pytorch, check_rocm_availability
    print("+ clip_gesture_model_pytorch imported successfully")
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    print("+ outputs directory ensured")
except Exception as e:
    print(f"ERROR: Import error: {e}")
    exit(1)

# Import project utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_path, get_config_path, setup_logger, get_logger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    print(config_path)
    
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
    print("=== ENTERING main() ===")
    
    parser = argparse.ArgumentParser(description='Train CLIP Gesture Model - PyTorch Only')
    parser.add_argument('--config', type=str, default='../../config/config_high_accuracy.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (0, 1, 2, 3). If not specified, will use GPU 0')
    
    print("Parsing arguments...")
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logger from config
    logger = setup_logger(config)
    
    logger.info("=== CLIP Gesture Model Training - PyTorch Only ===")
    logger.info(f"Config file: {args.config}")
    
    # Set ROCm environment first
    os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['ROCM_PATH'] = '/opt/rocm'
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['HIP_PLATFORM'] = 'amd'
    
    # Set device with PyTorch-only approach
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Available GPUs: {device_count}")
        
        if args.gpu is not None:
            # Use specified GPU directly
            gpu_id = args.gpu
            logger.info(f"Using specified GPU {gpu_id}")
            device = f'cuda:{gpu_id}'
            logger.info(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
        else:
            # Use first available GPU
            device = f'cuda:0'
            logger.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        logger.info("CUDA not available, using CPU")
    
    logger.info(f"Using device: {device}")
    
    # Proceed directly to training
    if device.startswith('cuda'):
        logger.info(f"Using GPU: {device}")
    
    # Prepare data with PyTorch only
    logger.info("=== Preparing Data (PyTorch Only) ===")
    logger.debug("About to call prepare_clip_gesture_data_pytorch...")
    data_config = config.get('data', {})
    logger.debug(f"Data config: {data_config}")
    logger.debug(f"Folder path: {data_config.get('folder_path', 'data')}")
    logger.debug(f"Max files: {data_config.get('max_files', 200)}")
    logger.debug(f"Sequence length: {data_config.get('sequence_length', 15)}")
    
    logger.info("Calling prepare_clip_gesture_data_pytorch now...")
    try:
        # Use threading-based timeout for cross-platform compatibility
        import threading
        import time
        
        result = [None]
        exception = [None]
        
        def run_data_prep():
            try:
                result[0] = prepare_clip_gesture_data_pytorch(
                    folder_path=data_config.get('folder_path', str(get_data_path())),
                    max_files=data_config.get('max_files', 200),
                    sequence_length=data_config.get('sequence_length', 15)
                )
            except Exception as e:
                exception[0] = e
        
        # Start data preparation in a separate thread
        thread = threading.Thread(target=run_data_prep)
        thread.daemon = True
        thread.start()
        
        # Wait for completion with timeout
        thread.join(timeout=60)  # 60 second timeout
        
        if thread.is_alive():
            print("ERROR: Data preparation timed out after 60 seconds")
            return
        
        if exception[0]:
            raise exception[0]
        
        if result[0] is None:
            print("ERROR: Data preparation failed - no result returned")
            return
            
        X, y, gesture_names, gesture_descriptions = result[0]
        print("prepare_clip_gesture_data_pytorch completed!")
    except Exception as e:
        print(f"ERROR: prepare_clip_gesture_data_pytorch failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(gesture_names)}")
    print(f"Gesture names: {gesture_names}")
    
    # Train model with PyTorch only
    print("\n=== Training Model (PyTorch Only) ===")
    training_config = config.get('training', {})
    
    print(f"Training config: {training_config}")
    print(f"Sequence length: {data_config.get('sequence_length', 15)}")
    print(f"Batch size: {training_config.get('batch_size', 16)}")
    print(f"Epochs: {training_config.get('num_epochs', 100)}")
    print(f"Learning rate: {training_config.get('learning_rate', 0.0005)}")
    print(f"Device: {device}")
    
    print("Calling train_clip_gesture_model_pytorch...")
    try:
        model, scaler, gesture_names, gesture_descriptions = train_clip_gesture_model_pytorch(
            X, y, gesture_names, gesture_descriptions,
            sequence_length=data_config.get('sequence_length', 15),
            batch_size=training_config.get('batch_size', 16),
            num_epochs=training_config.get('num_epochs', 100),
            learning_rate=training_config.get('learning_rate', 0.0005),
            device=device
        )
        print("train_clip_gesture_model_pytorch completed!")
    except Exception as e:
        print(f"ERROR: train_clip_gesture_model_pytorch failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Training Completed ===")
    print(f"Model saved as: clip_gesture_model_pytorch.pth")
    print(f"Scaler saved as: clip_gesture_scaler_pytorch.json")
    print(f"Info saved as: clip_gesture_info_pytorch.json")

if __name__ == "__main__":
    print("=== SCRIPT EXECUTION STARTING ===")
    try:
        main()
        print("=== SCRIPT COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        print(f"=== SCRIPT FAILED WITH ERROR: {e} ===")
        import traceback
        traceback.print_exc()
