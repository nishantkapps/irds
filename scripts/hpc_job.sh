#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 110G
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH --job-name="Runame"
#SBATCH -o ./slurm/out%j.txt
#SBATCH -e ./slurm/err%j.txt
#SBATCH --gres=gpu:1
nvidia-smi

conda env list
source activate irdsenv

cd ../model/train
python train_pytorch_only.py --config ../../config/experiment_tiny.yaml
