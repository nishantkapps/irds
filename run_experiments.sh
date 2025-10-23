#!/bin/bash

echo "=== CLIP Gesture Model Experiments ==="
echo "Start time: $(date)"

# Set ROCm environment variables
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PLATFORM=amd

# Install PyYAML if not already installed
pip install PyYAML

echo "=== Available Configurations ==="
echo "1. training_config.yaml - Default configuration"
echo "2. config_high_accuracy.yaml - High accuracy (90%+ target)"
echo "3. config_fast_training.yaml - Fast training (quick experiments)"
echo "4. config_experimental.yaml - Experimental (advanced techniques)"

echo ""
echo "=== Running Experiments ==="

# Experiment 1: High Accuracy (Target: 90%+)
echo "=== Experiment 1: High Accuracy ==="
echo "Start time: $(date)"
python train_with_config.py --config config_high_accuracy.yaml
echo "High accuracy experiment completed at: $(date)"

# Experiment 2: Fast Training (Quick test)
echo "=== Experiment 2: Fast Training ==="
echo "Start time: $(date)"
python train_with_config.py --config config_fast_training.yaml
echo "Fast training experiment completed at: $(date)"

# Experiment 3: Experimental (Advanced techniques)
echo "=== Experiment 3: Experimental ==="
echo "Start time: $(date)"
python train_with_config.py --config config_experimental.yaml
echo "Experimental experiment completed at: $(date)"

echo ""
echo "=== All Experiments Completed ==="
echo "End time: $(date)"

# Show results
echo "=== Generated Models ==="
ls -la clip_gesture_model_*.pth
ls -la clip_gesture_scaler_*.pkl
ls -la clip_*.png

echo "=== Model Sizes ==="
du -h clip_gesture_model_*.pth

echo "=== Experiment Summary ==="
echo "1. High Accuracy: clip_gesture_model_high_acc.pth"
echo "2. Fast Training: clip_gesture_model_fast.pth"
echo "3. Experimental: clip_gesture_model_experimental.pth"
