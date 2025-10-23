#!/bin/bash

echo "Submitting gesture recognition training jobs..."

# Submit standard gesture model training
echo "Submitting standard gesture model training..."
sbatch train_gesture_model.slurm

# Submit CLIP gesture model training
echo "Submitting CLIP gesture model training..."
sbatch train_clip_model.slurm

echo "Jobs submitted! Check status with: squeue -u $USER"
echo "Monitor output files: gesture_training_*.out and clip_training_*.out"
