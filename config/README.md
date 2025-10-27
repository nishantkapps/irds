# Configuration Files

All configuration files for the IRDS Gesture Recognition experiments.

## Available Configurations

### Experiment Configs (Recommended)

**experiment_tiny.yaml** - Tiny Model (~150K parameters)
- Fast training for quick iteration
- Batch size: 32
- Learning rate: 0.001
- Epochs: 100
- Best for: Rapid prototyping and testing

**experiment_small.yaml** - Small Model (~1.1M parameters)
- Good balance for simple datasets
- Batch size: 16
- Learning rate: 0.0005
- Epochs: 120
- Best for: Simple gesture patterns

**experiment_medium.yaml** - Medium Model (~2.5M parameters) ⭐ **Recommended**
- Balanced capacity and speed
- Batch size: 16
- Learning rate: 0.0003
- Epochs: 150
- Best for: Most use cases

**experiment_large.yaml** - Large Model (~14M parameters)
- High capacity for complex patterns
- Batch size: 8
- Learning rate: 0.0001
- Epochs: 150
- Best for: Complex gesture recognition

### General Config

**config_high_accuracy.yaml** - High Accuracy Training
- Uses medium architecture
- Optimized hyperparameters for accuracy
- Can be customized as needed

## Configuration Structure

All configs follow this structure:

```yaml
# Debug settings
debug:
  level: "Info"
  show_memory_usage: true
  show_timing: true

# Data settings
data:
  folder_path: null          # null = auto-detect
  max_files: null            # null = use all files
  sequence_length: 20
  test_size: 0.2
  random_state: 42

# Model selection (NEW!)
model:
  architecture: "medium"     # Options: tiny, small, medium, large, xlarge, xxlarge

# Training hyperparameters
training:
  num_epochs: 150
  batch_size: 16
  learning_rate: 0.0003
  optimizer: "Adam"
  weight_decay: 0.0001

# Output paths
output:
  model_save_path: "outputs/model_medium.pth"
  scaler_save_path: "outputs/scaler_medium.json"
  info_save_path: "outputs/info_medium.json"
  plots_save_path: "outputs/curves_medium.png"
  confusion_matrix_path: "outputs/confusion_medium.png"

# Experiment metadata (optional)
experiment:
  name: "gesture_medium"
  tags: ["medium", "2.5M", "recommended"]
  notes: "Medium model - balanced capacity and speed"
```

## Model Architectures

| Architecture | Parameters | Input→LSTM→Hidden | Use Case |
|-------------|-----------|-------------------|----------|
| tiny | 150K | 128→256×1→128 | Quick experiments |
| small | 1.1M | 256→384×2→192 | Simple patterns |
| **medium** | **2.5M** | **384→512×2→256** | **Recommended** |
| large | 14M | 512→768×3→512 | Complex patterns |
| xlarge | 24M | 768→1024×3→768 | Very large datasets |
| xxlarge | 51M | 1024→1536×4→1024 | Extreme capacity |

## Usage

### Run with Specific Config

```bash
cd model/train
python train_pytorch_only.py --config ../../config/experiment_medium.yaml
```

### Run All Experiments

```bash
cd model/train
./run_experiments.sh
```

This automatically runs: tiny, small, medium, and large experiments.

### Create Custom Config

Copy an existing experiment config and modify:

```bash
cp experiment_medium.yaml experiment_custom.yaml
# Edit experiment_custom.yaml
python train_pytorch_only.py --config ../../config/experiment_custom.yaml
```

## Key Parameters to Tune

### For Faster Training
- Reduce `num_epochs`
- Increase `batch_size` (if GPU memory allows)
- Use smaller `architecture` (tiny or small)
- Reduce `sequence_length`

### For Better Accuracy
- Increase `num_epochs`
- Use larger `architecture` (large or xlarge)
- Reduce `learning_rate`
- Increase `sequence_length`
- Set `max_files: null` to use all data

### For Avoiding Overfitting
- Use smaller `architecture`
- Reduce `num_epochs`
- Increase `weight_decay`
- Add more training data

## Outputs

All trained models save to `model/train/outputs/`:
- `model_<name>.pth` - Model weights
- `scaler_<name>.json` - Feature scaler
- `info_<name>.json` - Training metrics
- `curves_<name>.png` - Training plots
- `confusion_<name>.png` - Confusion matrix

## Tips

1. **Start with medium** - Good balance for most cases
2. **Try tiny first** - Quick sanity check before long training
3. **Compare multiple sizes** - Use `run_experiments.sh` to compare
4. **Check overfitting** - Use `compare_experiments.py` to analyze
5. **Tune on best model** - Once you find best size, fine-tune its hyperparameters

---

For more details, see: `readme/README_EXPERIMENTS.md`

