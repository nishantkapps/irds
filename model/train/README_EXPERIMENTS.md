# Multi-Model Gesture Recognition Experiments

This directory contains scripts for running and comparing multiple gesture recognition experiments with different model architectures.

## Available Models

The system supports 6 different model architectures optimized for different use cases:

| Model | Parameters | Description | Use Case |
|-------|------------|-------------|----------|
| `tiny` | ~500K | Fast training | Quick experiments, prototyping |
| `small` | ~1.5M | Simple patterns | Small datasets, baseline |
| `medium` | ~4M | Balanced | **Recommended for most cases** |
| `large` | ~11M | High capacity | Complex gesture patterns |
| `xlarge` | ~20M | Very high capacity | Large datasets, may overfit |
| `xxlarge` | ~35M | Extreme capacity | Very large datasets only |

## Quick Start

### 1. Run a Single Experiment

```bash
cd /home/nishant/project/irds/model/train

# Run with a specific model size
python train_pytorch_only.py --config ../../config/experiment_tiny.yaml
python train_pytorch_only.py --config ../../config/experiment_small.yaml
python train_pytorch_only.py --config ../../config/experiment_medium.yaml
python train_pytorch_only.py --config ../../config/experiment_large.yaml
```

### 2. Run All Experiments

Run multiple experiments sequentially:

```bash
cd /home/nishant/project/irds/model/train
./run_experiments.sh
```

This will:
- Run experiments for tiny, small, medium, and large models
- Save results in the `outputs/` directory
- Log progress for each experiment

### 3. Compare Results

After running experiments, compare the results:

```bash
python compare_experiments.py
```

This will display:
- Comparison table of all models
- Training vs test accuracy
- Overfitting analysis
- Best performing model

## Experiment Configuration

Each experiment has its own config file in `/home/nishant/project/irds/config/`:

- `experiment_tiny.yaml` - Tiny model configuration
- `experiment_small.yaml` - Small model configuration
- `experiment_medium.yaml` - Medium model configuration  
- `experiment_large.yaml` - Large model configuration

### Config Structure

```yaml
model:
  architecture: "medium"  # Select model: tiny, small, medium, large, xlarge, xxlarge

training:
  num_epochs: 150
  batch_size: 16
  learning_rate: 0.0003

data:
  max_files: null  # Use all files
  sequence_length: 20
  
output:
  model_save_path: "outputs/model_medium.pth"
  # ... other output paths
  
experiment:
  name: "gesture_medium"
  tags: ["medium", "4M", "recommended"]
  notes: "Medium model - balanced capacity and speed"
```

## Running on HPC

### Setup on HPC

```bash
# Copy files to HPC
scp -r /home/nishant/project/irds user@hpc:/path/to/destination/

# On HPC, set GPU device
export HIP_VISIBLE_DEVICES=1  # or your preferred GPU
```

### Run Experiments on HPC

```bash
cd /path/to/irds/model/train

# Run single experiment
python train_pytorch_only.py --config ../../config/experiment_medium.yaml

# Or run all experiments
./run_experiments.sh
```

### SLURM Batch Job (Optional)

Create a SLURM script for batch execution:

```bash
#!/bin/bash
#SBATCH --job-name=gesture_experiments
#SBATCH --output=logs/experiments_%j.out
#SBATCH --error=logs/experiments_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load rocm
module load python

cd /path/to/irds/model/train
./run_experiments.sh
```

## Output Files

Each experiment generates the following files in `outputs/`:

- `model_<size>.pth` - Trained model weights
- `scaler_<size>.json` - Feature scaler parameters
- `info_<size>.json` - Training metrics and metadata
- `curves_<size>.png` - Training curves plot
- `confusion_<size>.png` - Confusion matrix

## Tracking Experiments

### Experiment Metadata

Each model saves metadata including:
- Model architecture and parameter count
- Training/test accuracy and loss
- Hyperparameters used
- Gesture class names
- Training time

### Analyzing Overfitting

The comparison script identifies overfitting by checking the gap between training and test accuracy:

- **< 5% gap**: Well-balanced model ✓
- **5-10% gap**: Slight overfitting ⚠
- **10-20% gap**: Moderate overfitting ⚠⚠
- **> 20% gap**: Severe overfitting ✗✗

If you see severe overfitting:
1. Try a smaller model
2. Increase dropout
3. Add more training data
4. Reduce training epochs

## Model Architecture Details

All models use the same basic structure but with varying capacity:

1. **Input Projection**: Linear + LayerNorm
2. **LSTM Encoder**: Multi-layer LSTM (no dropout in LSTM for ROCm compatibility)
3. **Manual Dropout**: Applied after LSTM
4. **Classification Head**: Multiple FC layers with BatchNorm and Dropout
5. **Output**: Gesture class logits

Models differ in:
- Hidden dimensions
- Number of LSTM layers
- Number of classification layers
- Dropout rates

## Troubleshooting

### Import Errors

If you get module import errors:
```bash
# Ensure __init__.py exists
ls /home/nishant/project/irds/__init__.py

# Check PYTHONPATH
echo $PYTHONPATH
```

### GPU Errors

If you get segmentation faults on ROCm:
- The script includes GPU warm-up code at the start to prevent segfaults
- Set `HIP_VISIBLE_DEVICES` to your target GPU

### Out of Memory

If you run out of GPU memory:
1. Reduce batch size in config
2. Use a smaller model
3. Reduce sequence length

## Next Steps

1. **Run baseline experiments**: Start with all 4 standard sizes
2. **Analyze results**: Use `compare_experiments.py` to find best model
3. **Fine-tune**: Adjust hyperparameters for best performing size
4. **Deploy**: Use the best model for inference

