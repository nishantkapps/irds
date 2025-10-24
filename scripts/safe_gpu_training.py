#!/usr/bin/env python3
"""
Safe GPU training with error handling and fallback to CPU
"""

import os
import sys
import torch
import subprocess

def safe_gpu_test():
    """Safely test GPU without causing segfaults"""
    print("=== Safe GPU Test ===")
    
    try:
        # Set environment variables
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        os.environ['ROCM_PATH'] = '/opt/rocm'
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        os.environ['HIP_PLATFORM'] = 'amd'
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Very simple GPU test to avoid segfault
            try:
                # Create small tensor first
                test_tensor = torch.tensor([1.0]).cuda()
                print(f"✅ Basic GPU tensor creation: SUCCESS")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                return True
            except Exception as e:
                print(f"❌ GPU tensor creation failed: {e}")
                return False
        else:
            print("❌ GPU not available")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def run_training_safe():
    """Run training with safe GPU handling"""
    print("\n=== Safe Training ===")
    
    # Test GPU safely
    gpu_available = safe_gpu_test()
    
    if gpu_available:
        print("✅ GPU is available, starting training with GPU...")
        
        # Run training with GPU
        try:
            result = subprocess.run([
                sys.executable, 'model/train/train_with_config.py', 
                '--config', 'config_high_accuracy.yaml'
            ], check=True)
            print("✅ Training completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except FileNotFoundError:
            print("❌ model/train/train_with_config.py not found")
    else:
        print("❌ GPU not available, training will be slow on CPU")
        print("Consider using CPU training or fixing GPU issues")

def run_cpu_training():
    """Run training on CPU as fallback"""
    print("\n=== CPU Training Fallback ===")
    
    # Create CPU-only config
    cpu_config = """
data:
  folder_path: "data"
  max_files: 20                     # Very small for CPU
  sequence_length: 5                # Very short
  test_size: 0.2
  random_state: 42

model:
  skeleton_encoder:
    hidden_dim: 64
    num_layers: 1
    dropout: 0.1
    attention_heads: 2
    
  text_encoder:
    embedding_dim: 32
    hidden_dim: 64
    num_layers: 1
    
  clip_model:
    embedding_dim: 64
    logit_scale: 0.07

training:
  num_epochs: 10                    # Very few epochs for CPU
  batch_size: 4
  learning_rate: 0.001
  
  optimizer: "Adam"
  weight_decay: 0.01
  
  contrastive_weight: 0.5
  classification_weight: 0.5
  temperature: 0.07

augmentation:
  enabled: false

regularization:
  dropout: 0.1
  weight_decay: 0.01
  gradient_clip: 1.0
  early_stopping:
    enabled: false

output:
  model_save_path: "clip_gesture_model_cpu.pth"
  scaler_save_path: "clip_gesture_scaler_cpu.pkl"
  info_save_path: "clip_gesture_info_cpu.json"
  plots_save_path: "clip_training_curves_cpu.png"
  confusion_matrix_path: "clip_confusion_matrix_cpu.png"

experiment:
  name: "clip_gesture_cpu"
  tags: ["clip", "gesture", "cpu"]
  notes: "CPU training due to GPU issues"
"""
    
    with open('config_cpu.yaml', 'w') as f:
        f.write(cpu_config)
    
    print("Created CPU-only config: config_cpu.yaml")
    print("Run: python model/train/train_with_config.py --config config/config_cpu.yaml")

def main():
    print("=== Safe GPU Training ===")
    
    # Try GPU training first
    run_training_safe()
    
    # If GPU fails, offer CPU fallback
    print("\n=== Options ===")
    print("1. If GPU training failed, try CPU training:")
    print("   python model/train/train_with_config.py --config config/config_cpu.yaml")
    print("2. Fix GPU issues:")
    print("   - Check ROCm installation")
    print("   - Try different PyTorch version")
    print("   - Restart Python session")
    print("3. Use SLURM with proper GPU allocation")

if __name__ == "__main__":
    main()
