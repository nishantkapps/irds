#!/usr/bin/env python3
"""
Monitor training progress and estimate completion time
"""

import time
import os
import psutil
import torch

def monitor_training_progress():
    """Monitor current training progress"""
    print("=== Training Progress Monitor ===")
    
    # Check if training is running
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'])
                if 'clip_gesture_model.py' in cmdline or 'train_with_config.py' in cmdline or 'model/train/train_with_config.py' in cmdline:
                    training_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if training_processes:
        print(f"Found {len(training_processes)} training processes")
        for proc in training_processes:
            print(f"PID: {proc.pid}, CPU: {proc.cpu_percent()}%, Memory: {proc.memory_info().rss / 1024**3:.1f}GB")
    else:
        print("No training processes found")
    
    # Check system resources
    print(f"\n=== System Resources ===")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # Check GPU usage if available
    if torch.cuda.is_available():
        print(f"\n=== GPU Status ===")
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("\n=== GPU Status ===")
        print("GPU not available - training on CPU (very slow)")

def estimate_training_time():
    """Estimate remaining training time"""
    print("\n=== Training Time Estimation ===")
    
    # Check if we can find training logs
    log_files = [f for f in os.listdir('.') if f.endswith('.out') and 'training' in f]
    
    if log_files:
        print(f"Found log files: {log_files}")
        # Try to extract epoch information from logs
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if 'Epoch' in content:
                        print(f"Training logs found in {log_file}")
                        # Extract epoch numbers
                        import re
                        epochs = re.findall(r'Epoch (\d+)', content)
                        if epochs:
                            current_epoch = int(epochs[-1])
                            print(f"Current epoch: {current_epoch}")
            except:
                pass
    
    # Estimate based on configuration
    print("\n=== Time Estimates ===")
    
    if torch.cuda.is_available():
        print("With GPU (ROCm):")
        print("  - 100 epochs: ~2-4 hours")
        print("  - 150 epochs: ~3-6 hours") 
        print("  - 200 epochs: ~4-8 hours")
        print("  - 300 epochs: ~6-12 hours")
    else:
        print("With CPU only:")
        print("  - 100 epochs: ~8-16 hours")
        print("  - 150 epochs: ~12-24 hours")
        print("  - 200 epochs: ~16-32 hours")
        print("  - 300 epochs: ~24-48 hours")

def suggest_optimizations():
    """Suggest ways to speed up training"""
    print("\n=== Speed Optimization Suggestions ===")
    
    if not torch.cuda.is_available():
        print("❌ CRITICAL: GPU not available - training is very slow on CPU")
        print("💡 Solutions:")
        print("  - Check ROCm installation")
        print("  - Set HIP_VISIBLE_DEVICES=0")
        print("  - Use SLURM with GPU allocation")
        print("  - Reduce data size for faster experiments")
        return
    
    print("✅ GPU is available")
    print("💡 Speed optimizations:")
    print("  - Reduce max_files (200 → 100)")
    print("  - Reduce sequence_length (15 → 10)")
    print("  - Increase batch_size (16 → 32)")
    print("  - Reduce num_epochs (100 → 50)")
    print("  - Use mixed precision training")

def create_fast_config():
    """Create a fast configuration for quick testing"""
    fast_config = """
# Fast Training Configuration
data:
  folder_path: "data"
  max_files: 50                     # Reduced from 200
  sequence_length: 10               # Reduced from 15
  test_size: 0.2
  random_state: 42

model:
  skeleton_encoder:
    hidden_dim: 128                 # Reduced from 256
    num_layers: 2
    dropout: 0.2
    attention_heads: 4              # Reduced from 8
    
  text_encoder:
    embedding_dim: 64               # Reduced from 128
    hidden_dim: 128
    num_layers: 2
    
  clip_model:
    embedding_dim: 128              # Reduced from 256
    logit_scale: 0.07

training:
  num_epochs: 30                    # Reduced from 100
  batch_size: 32                    # Increased from 16
  learning_rate: 0.001              # Increased from 0.0005
  
  optimizer: "Adam"
  weight_decay: 0.01
  
  contrastive_weight: 0.5
  classification_weight: 0.5
  temperature: 0.07

augmentation:
  enabled: false                    # Disabled for speed

regularization:
  dropout: 0.2
  weight_decay: 0.01
  gradient_clip: 1.0
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001

output:
  model_save_path: "clip_gesture_model_fast.pth"
  scaler_save_path: "clip_gesture_scaler_fast.pkl"
  info_save_path: "clip_gesture_info_fast.json"
  plots_save_path: "clip_training_curves_fast.png"
  confusion_matrix_path: "clip_confusion_matrix_fast.png"

experiment:
  name: "clip_gesture_fast"
  tags: ["clip", "gesture", "fast", "rocm"]
  notes: "Fast training for quick results"
"""
    
    with open('config_very_fast.yaml', 'w') as f:
        f.write(fast_config)
    
    print(f"\n=== Created Fast Configuration ===")
    print("File: config_very_fast.yaml")
    print("Estimated time: 30-60 minutes (with GPU)")
    print("Usage: python model/train/train_with_config.py --config config/config_very_fast.yaml")

if __name__ == "__main__":
    monitor_training_progress()
    estimate_training_time()
    suggest_optimizations()
    create_fast_config()
