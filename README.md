# IRDS Gesture Recognition Project

PyTorch-based gesture recognition system with multiple model architectures for skeleton data analysis.

## 🚀 Quick Start

```bash
cd model/train
export HIP_VISIBLE_DEVICES=1
python train_pytorch_only.py --config ../../config/experiment_medium.yaml
```

## 📚 Documentation

All documentation is located in the [`readme/`](readme/) folder:

### Getting Started
- **[QUICK_START.md](readme/QUICK_START.md)** - Fast reference guide for running experiments

### Model & Training
- **[README_EXPERIMENTS.md](readme/README_EXPERIMENTS.md)** - Comprehensive guide to multi-model experiments
- **[MULTI_MODEL_SETUP.md](readme/MULTI_MODEL_SETUP.md)** - Technical details of model architecture system

### Cleanup & Maintenance
- **[CLEANUP_SUMMARY.md](readme/CLEANUP_SUMMARY.md)** - Training directory cleanup summary
- **[MODEL_CLEANUP_SUMMARY.md](readme/MODEL_CLEANUP_SUMMARY.md)** - Model directory cleanup summary

### Inference & Testing
- **[VIDEO_PIPELINE_GUIDE.md](readme/VIDEO_PIPELINE_GUIDE.md)** - Video-to-gesture inference guide

### Benchmarking
- **[BENCHMARKING_GUIDE.md](readme/BENCHMARKING_GUIDE.md)** - Complete GPU benchmarking guide (AMD vs NVIDIA)
- **[BENCHMARK_SUMMARY.md](readme/BENCHMARK_SUMMARY.md)** - Quick reference for benchmarking

### HPC Deployment
- **[HPC_FILE_LIST.md](readme/HPC_FILE_LIST.md)** - Files needed for HPC deployment

## 📁 Project Structure

```
irds/
├── model/
│   ├── clip_gesture_model_pytorch.py    # Main model implementation
│   ├── model_architectures.py           # 6 model size variants (150K-51M params)
│   ├── train/                           # Training scripts
│   │   ├── train_pytorch_only.py        # Single GPU training
│   │   ├── train_pytorch_multi_gpu.py   # Multi-GPU training
│   │   ├── run_experiments.sh           # Run all experiments
│   │   └── compare_experiments.py       # Compare results
│   └── test/                            # Testing utilities
│       ├── video_to_gesture.py          # Video inference
│       └── test_pytorch_clean.py        # Environment tests
├── config/                              # Experiment configurations
│   ├── experiment_tiny.yaml             # ~150K params
│   ├── experiment_small.yaml            # ~1.1M params
│   ├── experiment_medium.yaml           # ~2.5M params (recommended)
│   └── experiment_large.yaml            # ~14M params
├── data/                                # Dataset directory
├── readme/                              # All documentation
└── utils.py, tensor_utils.py            # Utility functions
```

## 🎯 Available Models

| Model | Parameters | Best For |
|-------|-----------|----------|
| tiny | 150K | Quick experiments |
| small | 1.1M | Simple patterns |
| medium | 2.5M | **Recommended** |
| large | 14M | Complex patterns |
| xlarge | 24M | Very large datasets |
| xxlarge | 51M | Extreme capacity |

## ⚡ Common Commands

### Run Single Experiment
```bash
cd model/train
python train_pytorch_only.py --config ../../config/experiment_medium.yaml
```

### Run All Experiments
```bash
cd model/train
./run_experiments.sh
```

### Compare Results
```bash
cd model/train
python compare_experiments.py
```

### Multi-GPU Training
```bash
cd model/train
python train_pytorch_multi_gpu.py --config ../../config/experiment_medium.yaml --gpu 1
```

### Benchmark Performance
```bash
# Run complete benchmark (training + inference)
bash scripts/run_full_benchmark.sh

# Compare AMD vs NVIDIA
python scripts/compare_gpu_benchmarks.py outputs/benchmark_amd_*.json outputs/benchmark_nvidia_*.json
```

## 🖥️ HPC Deployment

```bash
# Copy to HPC
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    /home/nishant/project/irds/ user@hpc:/path/to/irds/

# On HPC
export HIP_VISIBLE_DEVICES=1
cd /path/to/irds/model/train
./run_experiments.sh
```

## 📊 Output Files

Trained models are saved in `model/train/outputs/`:
- `model_<size>.pth` - Model weights
- `scaler_<size>.json` - Feature scaler
- `info_<size>.json` - Training metrics
- `curves_<size>.png` - Training plots
- `confusion_<size>.png` - Confusion matrix

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+ (with ROCm support for AMD GPUs)
- pandas, scikit-learn, pyyaml
- For video inference: opencv-python, mediapipe

## 📖 More Information

See the [`readme/`](readme/) folder for detailed documentation on all aspects of the project.

---

**Last Updated**: October 27, 2025

