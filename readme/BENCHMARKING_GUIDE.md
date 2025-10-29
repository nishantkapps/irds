# GPU Benchmarking Guide

This guide explains how to benchmark the gesture recognition model on AMD (ROCm) and NVIDIA (CUDA) GPUs and compare their performance.

## Overview

The benchmarking system tracks:
- **Training metrics**: Total time, epoch time, throughput (samples/sec), memory usage
- **Inference metrics**: Latency (P50/P95/P99), throughput (predictions/sec), time-to-first-prediction
- **System info**: GPU vendor, model, memory, PyTorch version, platform

## Quick Start

### Option 1: Automated Full Benchmark

Run the complete benchmark suite:

```bash
cd /path/to/irds
bash scripts/run_full_benchmark.sh
```

This will:
1. Detect your GPU type (AMD/NVIDIA)
2. Run training benchmark
3. Run inference benchmark
4. Save results to `outputs/benchmark_<gpu_type>_<timestamp>.json`

### Option 2: Manual Benchmarking

#### Step 1: Training Benchmark

```bash
python model/train/train_pytorch_only.py --config config/config_high_accuracy.yaml
```

This automatically:
- Tracks training time per epoch
- Measures data loading time
- Records GPU memory usage
- Calculates samples/second throughput
- Saves benchmark to `outputs/benchmark_<gpu_type>_<timestamp>.json`

#### Step 2: Inference Benchmark

```bash
python model/test/benchmark_inference.py \
    --model outputs/clip_gesture_model_pytorch.pth \
    --device cuda:0 \
    --warmup 20 \
    --iterations 200 \
    --batch-sizes 1,2,4,8,16,32,64
```

Options:
- `--model`: Path to trained model checkpoint
- `--device`: GPU device (cuda:0, cuda:1, etc.)
- `--warmup`: Number of warmup iterations (default: 10)
- `--iterations`: Number of benchmark iterations (default: 100)
- `--batch-sizes`: Comma-separated batch sizes to test

## Comparing AMD vs NVIDIA

### Step 1: Run on AMD System

```bash
# On AMD/ROCm system
cd /path/to/irds
bash scripts/run_full_benchmark.sh

# Copy the benchmark file
cp outputs/benchmark_amd_*.json ~/amd_benchmark.json
```

### Step 2: Run on NVIDIA System

```bash
# On NVIDIA/CUDA system
cd /path/to/irds
bash scripts/run_full_benchmark.sh

# Copy the benchmark file
cp outputs/benchmark_nvidia_*.json ~/nvidia_benchmark.json
```

### Step 3: Compare Results

```bash
python scripts/compare_gpu_benchmarks.py \
    ~/amd_benchmark.json \
    ~/nvidia_benchmark.json \
    --output outputs/amd_vs_nvidia_comparison.json
```

This generates a detailed comparison report showing:
- Side-by-side GPU specifications
- Training time comparison and speedup ratio
- Throughput comparison (samples/sec)
- Inference latency comparison
- Which GPU is faster for training and inference

## Benchmark Metrics Explained

### Training Metrics

| Metric | Description |
|--------|-------------|
| `total_runtime_minutes` | Total training time |
| `avg_epoch_time` | Average time per epoch (seconds) |
| `throughput` | Training samples processed per second |
| `final_train_accuracy` | Final training accuracy (%) |
| `test_accuracy` | Test set accuracy (%) |

### Inference Metrics

| Metric | Description |
|--------|-------------|
| `predictions_per_second` | Inference throughput |
| `avg_time_per_prediction` | Average latency per prediction (ms) |
| `time_to_first_prediction` | Time to first prediction (ms) |
| `p50_latency` | Median latency (ms) |
| `p95_latency` | 95th percentile latency (ms) |
| `p99_latency` | 99th percentile latency (ms) |

### Memory Metrics

| Metric | Description |
|--------|-------------|
| `allocated_gb` | GPU memory actively used |
| `reserved_gb` | GPU memory reserved by PyTorch |
| `max_allocated_gb` | Peak GPU memory usage |

## Example Output

### Training Benchmark Summary

```
================================================================================
BENCHMARK REPORT: clip_gesture_high_accuracy
================================================================================

[SYSTEM]
  Platform: Linux-6.8.0-85-generic-x86_64-with-glibc2.39
  PyTorch: 2.1.0+rocm5.7

[GPU]
  0: Radeon RX 7900 XTX
     Vendor: AMD (ROCm)
     Memory: 24.0 GB
     Compute: 11.0

[MODEL]
  Architecture: medium
  Parameters: 4,123,456

[DATA]
  Training samples: 19,587
  Test samples: 4,893
  Classes: 9
  Sequence length: 20

[TRAINING]
  Total epochs: 150
  Avg epoch time: 12.34s
  Final accuracy: 98.76%
  Test accuracy: 87.23%

[THROUGHPUT]
  training: 1587.3 samples/sec

[TOTAL RUNTIME]
  1851.23s (30.85m)
```

### Inference Benchmark Summary

```
[INFERENCE PERFORMANCE]
  Total predictions: 6,400
  Total batches: 500
  Avg predictions/sec: 2341.5
  Avg time per prediction: 0.43ms
  P50 latency: 12.34ms
  P95 latency: 15.67ms
  P99 latency: 18.92ms
  Min latency: 10.12ms
  Max latency: 21.45ms
  Avg time to first prediction: 12.56ms
```

### Comparison Report

```
================================================================================
BENCHMARK COMPARISON
================================================================================

[BENCHMARK 1]
  Experiment: clip_gesture_high_accuracy
  GPU: AMD - Radeon RX 7900 XTX
  Model: medium (4,123,456 params)
  Runtime: 30.85 minutes
  Train accuracy: 98.76%
  Test accuracy: 87.23%
  Throughput: 1587.3 samples/sec

[BENCHMARK 2]
  Experiment: clip_gesture_high_accuracy
  GPU: NVIDIA - RTX 4090
  Model: medium (4,123,456 params)
  Runtime: 28.12 minutes
  Train accuracy: 98.81%
  Test accuracy: 87.45%
  Throughput: 1742.8 samples/sec

[SPEEDUP ANALYSIS]
  AMD_vs_NVIDIA:
    Time ratio: 1.10x
    Faster GPU: NVIDIA

[THROUGHPUT ANALYSIS]
  AMD_vs_NVIDIA:
    Throughput ratio: 0.91x
    Faster GPU: NVIDIA
```

## Tips for Accurate Benchmarking

1. **Warmup**: Always include warmup iterations (10-20) to allow GPU context initialization
2. **Consistency**: Use the same model architecture and config across GPUs
3. **Multiple runs**: Run benchmarks 3-5 times and average results
4. **GPU exclusive**: Close other GPU applications during benchmarking
5. **Power/Thermal**: Ensure GPUs are properly cooled and at consistent temperatures
6. **Driver versions**: Document PyTorch, ROCm, and CUDA versions

## Benchmark Files

All benchmark reports are saved as JSON in `outputs/`:

```
outputs/
├── benchmark_amd_20251029_143022.json      # AMD training benchmark
├── benchmark_nvidia_20251029_150134.json   # NVIDIA training benchmark
└── benchmark_comparison_20251029_153045.json  # Comparison report
```

## Troubleshooting

### "No module named 'utils.benchmark'"

```bash
# Ensure you're running from project root
cd /path/to/irds
python -c "import sys; sys.path.insert(0, '.'); from utils.benchmark import GPUBenchmark"
```

### GPU out of memory during benchmark

Reduce batch size:
```bash
python model/test/benchmark_inference.py --batch-sizes 1,2,4,8
```

### Inconsistent timing results

- Increase `--iterations` (e.g., 500 or 1000)
- Ensure no other processes are using the GPU
- Check GPU temperature and throttling

## Advanced Usage

### Custom Batch Sizes

```bash
python model/test/benchmark_inference.py --batch-sizes 1,3,5,7,9,11,13,15
```

### Benchmark on Specific GPU

```bash
# Use GPU 1 instead of GPU 0
python model/test/benchmark_inference.py --device cuda:1
```

### CPU Benchmark

```bash
python model/test/benchmark_inference.py --device cpu
```

## Integration with CI/CD

You can integrate benchmarking into your CI/CD pipeline:

```yaml
# .github/workflows/benchmark.yml
name: GPU Benchmark
on: [push]
jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmark
        run: bash scripts/run_full_benchmark.sh
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: outputs/benchmark_*.json
```

## Related Documentation

- [Training Guide](./README_EXPERIMENTS.md)
- [Model Architectures](../model/model_architectures.py)
- [Configuration Guide](../config/README.md)

