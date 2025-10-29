# Benchmarking System Summary

## Quick Reference

### Run Complete Benchmark
```bash
bash scripts/run_full_benchmark.sh
```

### Training Only
```bash
python model/train/train_pytorch_only.py --config config/config_high_accuracy.yaml
```

### Inference Only
```bash
python model/test/benchmark_inference.py \
    --model outputs/clip_gesture_model_pytorch.pth \
    --device cuda:0
```

### Compare Results
```bash
python scripts/compare_gpu_benchmarks.py <amd_file.json> <nvidia_file.json>
```

## Key Metrics

### Training Performance
- **Total runtime** (minutes)
- **Samples/second** (throughput)
- **Epoch time** (seconds)
- **GPU memory usage** (GB)

### Inference Performance  
- **Predictions/second** (throughput)
- **Time per prediction** (milliseconds)
- **Time to first prediction** (TTFP in ms)
- **Latency percentiles** (P50, P95, P99 in ms)

### Comparison Metrics
- **Speedup ratio** (AMD vs NVIDIA)
- **Throughput ratio**
- **Memory efficiency**
- **Accuracy comparison**

## Files Created

```
/home/nishant/project/irds/
├── utils/
│   ├── __init__.py                    # Utils module initialization
│   └── benchmark.py                   # Core benchmark utilities
├── model/
│   └── test/
│       └── benchmark_inference.py     # Inference benchmarking script
├── scripts/
│   ├── compare_gpu_benchmarks.py      # Comparison utility
│   └── run_full_benchmark.sh          # Automated benchmark runner
├── outputs/
│   └── benchmark_*.json               # Benchmark results (auto-generated)
└── readme/
    ├── BENCHMARKING_GUIDE.md          # Comprehensive guide
    └── BENCHMARK_SUMMARY.md           # This file
```

## Example Workflow

### On AMD System (HPC)
```bash
cd /home/rocm_2/irds
bash scripts/run_full_benchmark.sh
# Creates: outputs/benchmark_amd_TIMESTAMP.json
```

### On NVIDIA System
```bash
cd /path/to/irds
bash scripts/run_full_benchmark.sh
# Creates: outputs/benchmark_nvidia_TIMESTAMP.json
```

### Compare
```bash
python scripts/compare_gpu_benchmarks.py \
    outputs/benchmark_amd_*.json \
    outputs/benchmark_nvidia_*.json
# Creates: outputs/benchmark_comparison_TIMESTAMP.json
```

## Integration Points

The benchmark system is integrated into:

1. **Training** (`model/clip_gesture_model_pytorch.py`):
   - Tracks epoch timing
   - Records memory usage
   - Calculates throughput
   - Measures data loading time

2. **Training Runner** (`model/train/train_pytorch_only.py`):
   - Initializes benchmark
   - Saves results automatically
   - Prints summary report

3. **Inference** (`model/test/benchmark_inference.py`):
   - Measures latency distributions
   - Tests multiple batch sizes
   - Calculates predictions/second
   - Records time-to-first-prediction

## Customization

### Change Batch Sizes
Edit `scripts/run_full_benchmark.sh`:
```bash
--batch-sizes 1,2,4,8,16,32,64,128
```

### Change Number of Iterations
```bash
python model/test/benchmark_inference.py --iterations 500
```

### Add Custom Metrics
Edit `utils/benchmark.py` and add new recording methods:
```python
def record_custom_metric(self, metric_name: str, value: float):
    self.metrics['custom'][metric_name] = value
```

## See Also

- [BENCHMARKING_GUIDE.md](./BENCHMARKING_GUIDE.md) - Full documentation
- [README_EXPERIMENTS.md](../model/train/README_EXPERIMENTS.md) - Training experiments
- [config/README.md](../config/README.md) - Configuration guide

