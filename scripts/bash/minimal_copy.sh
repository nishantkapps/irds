#!/bin/bash

# Minimal file copy for HPC training
# Only copies the absolutely essential files

if [ $# -eq 0 ]; then
    echo "Usage: $0 <hpc_username>@<hpc_hostname>:/path/to/destination/"
    echo "Example: $0 user@hpc.university.edu:/home/user/irds/"
    exit 1
fi

DESTINATION=$1
echo "Copying minimal essential files to: $DESTINATION"

# Create destination directory
ssh ${DESTINATION%:*} "mkdir -p ${DESTINATION#*:}"

echo "=== Copying Essential Files ==="

# 1. Data directory (MOST IMPORTANT)
echo "1. Copying data directory..."
rsync -avz --progress data/ ${DESTINATION}data/

# 2. Model training scripts
echo "2. Copying model scripts..."
scp gesture_model.py ${DESTINATION}
scp clip_gesture_model.py ${DESTINATION}

# 3. SLURM job scripts
echo "3. Copying SLURM scripts..."
scp train_gesture_model.slurm ${DESTINATION}
scp train_clip_model.slurm ${DESTINATION}
scp submit_training.sh ${DESTINATION}

# 4. Configuration files
echo "4. Copying configuration files..."
scp labels.csv ${DESTINATION}
scp joints.txt ${DESTINATION}
scp connections.txt ${DESTINATION}

echo "=== Minimal Copy Complete ==="
echo "Essential files copied to: $DESTINATION"
echo ""
echo "Next steps on HPC:"
echo "1. SSH to HPC machine"
echo "2. cd to project directory" 
echo "3. chmod +x *.slurm submit_training.sh"
echo "4. ./submit_training.sh"
