#!/bin/bash

# Script to run multiple gesture recognition experiments with different model sizes

echo "==================================="
echo "Running Gesture Recognition Experiments"
echo "==================================="

# Set GPU device (change as needed)
export HIP_VISIBLE_DEVICES=1

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Array of experiment configs
configs=("experiment_tiny" "experiment_small" "experiment_medium" "experiment_large")

# Run each experiment
for config in "${configs[@]}"; do
    echo ""
    echo "==================================="
    echo "Starting experiment: $config"
    echo "==================================="
    
    # Run training
    python train_pytorch_only.py --config "../../config/${config}.yaml"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ $config completed successfully"
    else
        echo "✗ $config failed"
    fi
    
    echo ""
done

echo "==================================="
echo "All experiments completed!"
echo "==================================="
echo "Results saved in outputs/ directory"

