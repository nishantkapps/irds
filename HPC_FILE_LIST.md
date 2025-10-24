# Files to Copy to HPC Machine for Gesture Model Training

## Essential Files (Required)

### 1. Data Files
- **`data/`** - The entire data directory containing all .txt files
  - Size: ~2.5GB (2589 files)
  - **CRITICAL**: This is the largest and most important directory

### 2. Model Training Scripts
- **`gesture_model.py`** - Standard gesture recognition model
- **`clip_gesture_model.py`** - CLIP-style gesture recognition model

### 3. SLURM Job Scripts
- **`train_gesture_model.slurm`** - SLURM script for standard model
- **`train_clip_model.slurm`** - SLURM script for CLIP model  
- **`submit_training.sh`** - Script to submit both jobs

### 4. Configuration Files
- **`labels.csv`** - Gesture label mappings
- **`joints.txt`** - Joint definitions
- **`connections.txt`** - Skeleton connections

## Optional Files (For Reference)

### 5. Visualization Scripts
- **`repetition_visualizer.py`** - For visualizing repetitions
- **`irds-eda.py`** - Data loading utilities

### 6. Configuration Files
- **`config*.yaml`** - Various configuration files
- **`irds-eda.ipynb`** - Jupyter notebook for reference

## Quick Copy Commands

### Option 1: Use the provided script
```bash
chmod +x copy_to_hpc.sh
./copy_to_hpc.sh user@hpc.university.edu:/home/user/irds/
```

### Option 2: Manual copy commands
```bash
# Copy data directory (largest)
rsync -avz --progress data/ user@hpc.university.edu:/home/user/irds/data/

# Copy model files
scp gesture_model.py clip_gesture_model.py user@hpc.university.edu:/home/user/irds/

# Copy SLURM scripts
scp *.slurm submit_training.sh user@hpc.university.edu:/home/user/irds/

# Copy configuration files
scp labels.csv joints.txt connections.txt user@hpc.university.edu:/home/user/irds/
```

## HPC Setup Commands

Once files are copied, run these on the HPC machine:

```bash
# Make scripts executable
chmod +x *.slurm submit_training.sh

# Submit training jobs
./submit_training.sh

# Monitor jobs
squeue -u $USER

# Check output
tail -f gesture_training_*.out
tail -f clip_training_*.out
```

## Expected Output Files

After training completes, you'll have:

### Standard Model Outputs
- `gesture_model.pth` - Trained model weights
- `gesture_scaler.pkl` - Data scaler
- `confusion_matrix.png` - Confusion matrix plot
- `training_curves.png` - Training curves plot

### CLIP Model Outputs  
- `clip_gesture_model.pth` - Trained CLIP model weights
- `clip_gesture_scaler.pkl` - Data scaler
- `clip_gesture_info.json` - Gesture information
- `clip_confusion_matrix.png` - Confusion matrix plot
- `clip_training_curves.png` - Training curves plot

## Resource Requirements

### Standard Model
- **Time**: ~24 hours
- **Memory**: 32GB
- **GPU**: 1 GPU
- **CPU**: 8 cores

### CLIP Model  
- **Time**: ~36 hours
- **Memory**: 64GB
- **GPU**: 1 GPU
- **CPU**: 16 cores

## File Sizes (Approximate)
- `data/`: ~2.5GB
- Python scripts: ~50MB total
- SLURM scripts: ~5KB total
- Configuration files: ~10KB total

