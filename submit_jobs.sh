#!/bin/bash

echo "Submitting gesture recognition training jobs..."

# Submit standard gesture model training
echo "Submitting standard gesture model training..."
JOB1=$(sbatch train_gesture_model.slurm | awk '{print $4}')
echo "Standard model job ID: $JOB1"

# Submit CLIP gesture model training
echo "Submitting CLIP gesture model training..."
JOB2=$(sbatch train_clip_model.slurm | awk '{print $4}')
echo "CLIP model job ID: $JOB2"

echo ""
echo "Jobs submitted successfully!"
echo "Standard model job ID: $JOB1"
echo "CLIP model job ID: $JOB2"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $JOB1,$JOB2"
echo ""
echo "Check output with:"
echo "  tail -f gesture_training_*.out"
echo "  tail -f clip_training_*.out"
echo ""
echo "Cancel jobs if needed:"
echo "  scancel $JOB1"
echo "  scancel $JOB2"
