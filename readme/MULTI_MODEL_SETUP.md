# Multi-Model Experiment Setup - Summary

## What Was Created

A flexible multi-model experiment framework for gesture recognition with 6 different model architectures.

## Files Created/Modified

### New Files

1. **`model/model_architectures.py`**
   - Contains 6 model architectures (tiny to xxlarge)
   - Parameter counts: 500K to 35M parameters
   - Utility functions: `get_model()`, `count_parameters()`, `MODEL_INFO`

2. **Configuration Files**
   - `config/experiment_tiny.yaml` - ~500K params
   - `config/experiment_small.yaml` - ~1.5M params
   - `config/experiment_medium.yaml` - ~4M params
   - `config/experiment_large.yaml` - ~11M params

3. **Experiment Scripts**
   - `model/train/run_experiments.sh` - Run all experiments sequentially
   - `model/train/compare_experiments.py` - Compare and analyze results
   - `model/train/README_EXPERIMENTS.md` - Detailed documentation

### Modified Files

1. **`model/clip_gesture_model_pytorch.py`**
   - Added import: `from model.model_architectures import get_model, count_parameters, MODEL_INFO`
   - Modified `train_clip_gesture_model_pytorch()` to accept `model_architecture` parameter
   - Replaced inline `SimpleCLIPModel` with dynamic model selection using `get_model()`
   - Added logging for model info and parameter counts

2. **`model/train/train_pytorch_only.py`**
   - Added model architecture config reading: `model_architecture = model_config.get('architecture', 'medium')`
   - Pass architecture to training function: `model_architecture=model_architecture`
   - Added logging for selected architecture

3. **`config/config_high_accuracy.yaml`**
   - Replaced old model config with simple architecture selector
   - Added model size guide comments

## Model Architectures

| Model | Params | Input | LSTM | Hidden | FC Layers | Description |
|-------|--------|-------|------|--------|-----------|-------------|
| tiny | ~500K | 75→128 | 256×1 | 128 | 2 | Fast experiments |
| small | ~1.5M | 75→256 | 384×2 | 192 | 2 | Simple datasets |
| medium | ~4M | 75→384 | 512×2 | 256 | 3 | **Recommended** |
| large | ~11M | 75→512 | 768×3 | 512 | 3 | Complex patterns |
| xlarge | ~20M | 75→768 | 1024×3 | 768 | 4 | May overfit |
| xxlarge | ~35M | 75→1024 | 1536×4 | 1024 | 4 | Very large datasets |

## Usage Examples

### Single Experiment

```bash
cd /home/nishant/project/irds/model/train
python train_pytorch_only.py --config ../../config/experiment_medium.yaml --skip_gpu_test
```

### All Experiments

```bash
cd /home/nishant/project/irds/model/train
./run_experiments.sh
```

### Compare Results

```bash
cd /home/nishant/project/irds/model/train
python compare_experiments.py
```

## Config Structure

Each experiment config specifies the model architecture:

```yaml
model:
  architecture: "medium"  # Options: tiny, small, medium, large, xlarge, xxlarge
```

The system automatically:
1. Loads the appropriate model from `model_architectures.py`
2. Logs model details and parameter count
3. Trains with specified hyperparameters
4. Saves results with unique filenames

## HPC Deployment

To copy to HPC:

```bash
# Copy the project
scp -r /home/nishant/project/irds user@hpc:/path/to/destination/

# On HPC
export HIP_VISIBLE_DEVICES=1
cd /path/to/irds/model/train
./run_experiments.sh
```

## Key Features

1. **Flexible Model Selection**: Choose model size via config file
2. **Consistent Interface**: All models use same training pipeline
3. **ROCm Compatible**: No dropout in LSTM layers to avoid MIOpen errors
4. **Experiment Tracking**: Unique output files for each model size
5. **Easy Comparison**: Built-in comparison script for analyzing results
6. **Overfitting Detection**: Automatic analysis of train/test gap

## Output Files

For each model size (e.g., "medium"):
- `outputs/model_medium.pth` - Model weights
- `outputs/scaler_medium.json` - Feature scaler
- `outputs/info_medium.json` - Metrics and metadata
- `outputs/curves_medium.png` - Training curves
- `outputs/confusion_medium.png` - Confusion matrix

## Next Steps

1. Run baseline experiments with all model sizes
2. Use `compare_experiments.py` to identify best architecture
3. Fine-tune hyperparameters for best performing size
4. Track and log results for paper/documentation

