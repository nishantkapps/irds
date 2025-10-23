#!/bin/bash
#SBATCH --job-name=gesture_training
#SBATCH --output=gesture_training_%j.out
#SBATCH --error=gesture_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/nishant/project/irds"

# Navigate to project directory
cd /home/nishant/project/irds

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Print Python and PyTorch versions
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

# Train standard gesture model
echo "Starting standard gesture model training..."
python gesture_model.py

# Train CLIP gesture model
echo "Starting CLIP gesture model training..."
python clip_gesture_model.py

# Print final results
echo "Training completed!"
echo "Generated files:"
ls -la *.pth *.pkl *.png *.json 2>/dev/null || echo "No output files found"

# Deactivate virtual environment
deactivate
