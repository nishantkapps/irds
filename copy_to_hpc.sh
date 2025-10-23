#!/bin/bash

# Script to copy necessary files to HPC machine for gesture model training
# Usage: ./copy_to_hpc.sh <hpc_username>@<hpc_hostname>:/path/to/destination/

if [ $# -eq 0 ]; then
    echo "Usage: $0 <hpc_username>@<hpc_hostname>:/path/to/destination/"
    echo "Example: $0 user@hpc.university.edu:/home/user/irds/"
    exit 1
fi

DESTINATION=$1
echo "Copying files to: $DESTINATION"

# Create destination directory
ssh ${DESTINATION%:*} "mkdir -p ${DESTINATION#*:}"

echo "=== Copying Core Data Files ==="
# Copy the data directory (this is the largest)
echo "Copying data directory (this may take a while)..."
rsync -avz --progress data/ ${DESTINATION}data/

echo "=== Copying Model Files ==="
# Copy Python model files
scp gesture_model.py ${DESTINATION}
scp clip_gesture_model.py ${DESTINATION}

echo "=== Copying Configuration Files ==="
# Copy configuration files
scp labels.csv ${DESTINATION}
scp joints.txt ${DESTINATION}
scp connections.txt ${DESTINATION}

echo "=== Copying SLURM Job Scripts ==="
# Copy SLURM scripts
scp train_gesture_model.slurm ${DESTINATION}
scp train_clip_model.slurm ${DESTINATION}
scp submit_training.sh ${DESTINATION}

echo "=== Copying Visualization Files (Optional) ==="
# Copy visualization files (optional, for reference)
scp repetition_visualizer.py ${DESTINATION}
scp irds-eda.py ${DESTINATION}

echo "=== Copying Configuration Files ==="
# Copy config files
scp config*.yaml ${DESTINATION}

echo "=== Copying Notebook (Optional) ==="
# Copy notebook for reference
scp irds-eda.ipynb ${DESTINATION}

echo "=== Setting up HPC Environment ==="
# Create a setup script on the HPC machine
cat << 'EOF' > setup_hpc.sh
#!/bin/bash
# Setup script for HPC environment

echo "Setting up HPC environment for gesture model training..."

# Make scripts executable
chmod +x submit_training.sh
chmod +x train_gesture_model.slurm
chmod +x train_clip_model.slurm

# Create output directory
mkdir -p outputs
mkdir -p models

echo "Setup complete!"
echo "To submit jobs, run: ./submit_training.sh"
echo "To check job status, run: squeue -u \$USER"
echo "To monitor output, run: tail -f gesture_training_*.out"
EOF

scp setup_hpc.sh ${DESTINATION}
ssh ${DESTINATION%:*} "chmod +x ${DESTINATION#*:}setup_hpc.sh"

echo "=== Copy Complete ==="
echo "Files copied to: $DESTINATION"
echo ""
echo "Next steps on HPC machine:"
echo "1. SSH to the HPC machine"
echo "2. Navigate to the project directory"
echo "3. Run: ./setup_hpc.sh"
echo "4. Run: ./submit_training.sh"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check output with: tail -f *.out"
