#!/bin/bash

# Script to run multiple gesture recognition experiments with ALL GPUs

echo "==================================="
echo "Running Gesture Recognition Experiments"
echo "Using ALL GPUs (Multi-GPU Mode)"
echo "==================================="

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Array of experiment configs
configs=("experiment_tiny" "experiment_small" "experiment_medium" "experiment_large")

# Run each experiment using all GPUs
for config in "${configs[@]}"; do
    echo ""
    echo "==================================="
    echo "Starting experiment: $config"
    echo "Using all 4 GPUs"
    echo "==================================="
    
    # Run training with all GPUs
    python train_pytorch_multi_gpu.py --config "../../config/${config}.yaml" --all-gpus
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ $config completed successfully"
    else
        echo "✗ $config failed"
    fi
    
    echo ""
done

echo "==================================="
echo "All multi-GPU experiments completed!"
echo "==================================="
echo "Results saved in outputs/ directory"

