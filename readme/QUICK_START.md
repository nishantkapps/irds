# Quick Start - Multi-Model Experiments

## 🎯 Goal
Run multiple gesture recognition experiments with different model sizes to find the best performing architecture and avoid overfitting.

## 📋 What's Available

6 model architectures ranging from 150K to 51M parameters:

```
tiny (150K)    → Quick experiments
small (1.1M)   → Simple datasets  
medium (2.5M)  → Balanced (recommended)
large (14M)    → High capacity
xlarge (24M)   → Very high capacity
xxlarge (51M)  → Extreme capacity
```

## 🚀 Quick Commands

### Run Single Experiment

```bash
cd /home/nishant/project/irds/model/train

# Medium model (recommended)
python train_pytorch_only.py --config ../../config/experiment_medium.yaml

# Or try different sizes
python train_pytorch_only.py --config ../../config/experiment_tiny.yaml
python train_pytorch_only.py --config ../../config/experiment_small.yaml
python train_pytorch_only.py --config ../../config/experiment_large.yaml
```

### Run All Experiments

```bash
cd /home/nishant/project/irds/model/train
./run_experiments.sh
```

### Compare Results

```bash
cd /home/nishant/project/irds/model/train
python compare_experiments.py
```

Output will show:
- Accuracy comparison table
- Train vs test performance
- Overfitting analysis
- Best model recommendation

## 🖥️ HPC Commands

```bash
# On HPC, set GPU first
export HIP_VISIBLE_DEVICES=1

# Then run experiments
cd /path/to/irds/model/train
./run_experiments.sh

# Compare results
python compare_experiments.py
```

## 📁 Output Files

Results saved in `model/train/outputs/`:

```
outputs/
├── model_tiny.pth          # Model weights
├── scaler_tiny.json        # Feature scaler
├── info_tiny.json          # Training metrics
├── curves_tiny.png         # Training plots
├── confusion_tiny.png      # Confusion matrix
├── model_small.pth
├── ...
```

## ⚙️ Customizing Experiments

Edit config files in `/home/nishant/project/irds/config/`:

```yaml
# config/experiment_medium.yaml
model:
  architecture: "medium"    # Change to: tiny, small, large, etc.

training:
  num_epochs: 150          # Adjust as needed
  batch_size: 16
  learning_rate: 0.0003

data:
  max_files: null          # null = use all files
  sequence_length: 20
```

## 🔍 Interpreting Results

After running `compare_experiments.py`, look for:

1. **Best Test Accuracy**: Which model performs best on unseen data
2. **Overfitting**: Gap between train and test accuracy
   - < 5% gap: ✓ Good
   - 5-10%: ⚠ Slight overfit
   - 10-20%: ⚠⚠ Moderate overfit  
   - > 20%: ✗✗ Severe overfit

3. **If Overfitting**:
   - Use a smaller model
   - Increase dropout in config
   - Get more training data

## 📊 Example Workflow

```bash
# 1. Run all experiments
cd /home/nishant/project/irds/model/train
./run_experiments.sh

# 2. Compare results
python compare_experiments.py

# 3. Based on results, fine-tune best model by editing its config
# Edit config/experiment_<best_model>.yaml

# 4. Re-run best model
python train_pytorch_only.py --config ../../config/experiment_<best_model>.yaml

# 5. Check results
python compare_experiments.py
```

## ✅ Testing

Verify everything works:

```bash
cd /home/nishant/project/irds/model/train

# Quick test (should complete in a few seconds)
python -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent.parent))
from model.model_architectures import get_model, MODEL_INFO
print('Available models:', list(MODEL_INFO.keys()))
import torch
model = get_model('tiny', 75, 10, 'cpu')
print('✓ Test passed!')
"
```

## 📖 More Information

- **Detailed docs**: `model/train/README_EXPERIMENTS.md`
- **Setup summary**: `MULTI_MODEL_SETUP.md`
- **Model code**: `model/model_architectures.py`

## 🆘 Troubleshooting

**Import errors?**
```bash
# Check __init__.py exists
ls /home/nishant/project/irds/__init__.py
```

**GPU errors on HPC?**
- Set `HIP_VISIBLE_DEVICES` before running
- GPU warm-up code is already included in the script to prevent segfaults

**Out of memory?**
- Reduce `batch_size` in config
- Use a smaller model architecture
- Reduce `sequence_length` in config

## 🎓 Tips

1. **Start with medium model** - Good balance of capacity and speed
2. **Run all sizes once** - Understand the capacity/performance tradeoff
3. **Watch for overfitting** - Large models may memorize training data
4. **Compare systematically** - Use `compare_experiments.py` after each run
5. **Track your experiments** - Output files named by model size for easy comparison

