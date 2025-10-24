#!/bin/bash
#SBATCH --job-name=clip_gesture_training
#SBATCH --output=clip_training_%j.out
#SBATCH --error=clip_training_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules for ROCm
module load python/3.9
module load rocm/5.4.0
module load gcc/9.3.0

# Set environment variables for ROCm
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=16

# Navigate to project directory (update this path for your HPC machine)
# cd /path/to/your/project/directory

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Print system information
echo "=== System Information ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Print GPU information
echo "=== GPU Information ==="
rocm-smi

# Print Python and PyTorch versions
echo "=== Software Versions ==="
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "ROCm available:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        print('ROCm test: SUCCESS - GPU tensor creation works')
    except Exception as e:
        print(f'ROCm test: FAILED - {e}')
else:
    print('ROCm test: FAILED - torch.cuda.is_available() returned False')
"

# Train CLIP gesture model
echo "=== Starting CLIP Gesture Model Training ==="
echo "Start time: $(date)"
python model/clip_gesture_model.py
echo "End time: $(date)"

# Print final results
echo "=== Training Results ==="
echo "Generated files:"
ls -la outputs/clip_*.pth outputs/clip_*.pkl outputs/clip_*.png outputs/clip_*.json 2>/dev/null || echo "No CLIP output files found"

# Print file sizes
echo "Model file sizes:"
du -h outputs/clip_gesture_model.pth 2>/dev/null || echo "Model file not found"
du -h outputs/clip_gesture_scaler.pkl 2>/dev/null || echo "Scaler file not found"

# Deactivate virtual environment
deactivate

echo "=== Job Completed ==="
echo "End time: $(date)"
