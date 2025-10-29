# Benchmarking System - Setup Complete

## What Was Added

A comprehensive GPU benchmarking system has been integrated into your project to compare AMD (ROCm) vs NVIDIA (CUDA) performance.

### New Files Created

1. **`utils/benchmark.py`** - Core benchmarking utilities
   - `GPUBenchmark` class for tracking metrics
   - `compare_benchmarks()` function for comparing results
   - Tracks training, inference, memory, and throughput metrics

2. **`model/test/benchmark_inference.py`** - Inference benchmarking script
   - Measures latency (P50, P95, P99)
   - Calculates throughput (predictions/sec)
   - Records time-to-first-prediction (TTFP)
   - Tests multiple batch sizes

3. **`scripts/compare_gpu_benchmarks.py`** - Comparison utility
   - Compares benchmark JSON files
   - Generates speedup ratios
   - Creates comprehensive comparison reports

4. **`scripts/run_full_benchmark.sh`** - Automated benchmark runner
   - Detects GPU type (AMD/NVIDIA)
   - Runs training benchmark
   - Runs inference benchmark
   - Saves results automatically

5. **Documentation**
   - `readme/BENCHMARKING_GUIDE.md` - Complete guide
   - `readme/BENCHMARK_SUMMARY.md` - Quick reference

### Modified Files

1. **`model/clip_gesture_model_pytorch.py`**
   - Added `benchmark` parameter to `train_clip_gesture_model_pytorch()`
   - Integrated timing and memory tracking
   - Records epoch metrics, throughput, and GPU memory

2. **`model/train/train_pytorch_only.py`**
   - Initializes `GPUBenchmark` object
   - Passes benchmark to training function
   - Saves and prints benchmark reports

3. **`utils/__init__.py`** - Created proper module initialization

4. **`README.md`** - Updated with benchmarking section

## How to Use on Different Devices

### On AMD System (HPC with ROCm)

```bash
# 1. Copy all files to HPC
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    /home/nishant/project/irds/ user@hpc:/home/rocm_2/irds/

# 2. On HPC, run benchmark
cd /home/rocm_2/irds
bash scripts/run_full_benchmark.sh

# 3. Copy results back
# Output: outputs/benchmark_amd_TIMESTAMP.json
```

### On NVIDIA System

```bash
# 1. Navigate to project
cd /path/to/irds

# 2. Run benchmark
bash scripts/run_full_benchmark.sh

# 3. Results saved to:
# Output: outputs/benchmark_nvidia_TIMESTAMP.json
```

### Compare Results

```bash
# After running on both AMD and NVIDIA:
python scripts/compare_gpu_benchmarks.py \
    outputs/benchmark_amd_20251029_143022.json \
    outputs/benchmark_nvidia_20251029_150134.json

# Output: outputs/benchmark_comparison_TIMESTAMP.json
```

## Key Metrics Tracked

### Training Performance
- ✅ Total runtime (seconds/minutes)
- ✅ Average epoch time (seconds)
- ✅ Throughput (samples/second)
- ✅ GPU memory usage (allocated/reserved/peak)
- ✅ Training accuracy over time
- ✅ Test accuracy

### Inference Performance (LLM-style metrics)
- ✅ **Predictions per second** (throughput)
- ✅ **Time per prediction** (latency in milliseconds)
- ✅ **Time to first prediction** (TTFP in milliseconds)
- ✅ **P50, P95, P99 latencies** (milliseconds)
- ✅ Min/max latency
- ✅ Performance across multiple batch sizes

### System Information
- ✅ GPU vendor (AMD/NVIDIA)
- ✅ GPU model name
- ✅ Total GPU memory
- ✅ PyTorch version
- ✅ CUDA/ROCm version
- ✅ Platform info

## Quick Commands

```bash
# Training benchmark only
python model/train/train_pytorch_only.py --config config/config_high_accuracy.yaml

# Inference benchmark only
python model/test/benchmark_inference.py \
    --model outputs/clip_gesture_model_pytorch.pth \
    --device cuda:0 \
    --iterations 200 \
    --batch-sizes 1,4,8,16,32

# Full automated benchmark
bash scripts/run_full_benchmark.sh

# Compare two benchmarks
python scripts/compare_gpu_benchmarks.py file1.json file2.json
```

## Example Output

```
================================================================================
BENCHMARK REPORT: clip_gesture_high_accuracy
================================================================================

[GPU]
  0: Radeon RX 7900 XTX
     Vendor: AMD (ROCm)
     Memory: 24.0 GB

[MODEL]
  Architecture: medium
  Parameters: 4,123,456

[TRAINING]
  Total epochs: 150
  Avg epoch time: 12.34s
  Test accuracy: 87.23%

[INFERENCE PERFORMANCE]
  Avg predictions/sec: 2341.5
  Avg time per prediction: 0.43ms
  P50 latency: 12.34ms
  P95 latency: 15.67ms
  Time to first prediction: 12.56ms

[TOTAL RUNTIME]
  1851.23s (30.85m)
```

## Files to Copy to HPC

All the new/modified files that need to be on both systems:

```
irds/
├── utils/
│   ├── __init__.py              ← MODIFIED
│   └── benchmark.py             ← NEW
├── model/
│   ├── clip_gesture_model_pytorch.py    ← MODIFIED
│   ├── train/
│   │   └── train_pytorch_only.py        ← MODIFIED
│   └── test/
│       └── benchmark_inference.py       ← NEW
├── scripts/
│   ├── compare_gpu_benchmarks.py        ← NEW
│   └── run_full_benchmark.sh            ← NEW
└── readme/
    ├── BENCHMARKING_GUIDE.md            ← NEW
    └── BENCHMARK_SUMMARY.md             ← NEW
```

## Next Steps

1. **Copy to HPC**: Copy all modified files to your HPC system
   ```bash
   rsync -avz /home/nishant/project/irds/ user@hpc:/home/rocm_2/irds/
   ```

2. **Run on AMD**: On HPC, run the benchmark
   ```bash
   cd /home/rocm_2/irds
   bash scripts/run_full_benchmark.sh
   ```

3. **Run on NVIDIA**: On your NVIDIA system, run the same
   ```bash
   bash scripts/run_full_benchmark.sh
   ```

4. **Compare**: Generate comparison report
   ```bash
   python scripts/compare_gpu_benchmarks.py <amd_file> <nvidia_file>
   ```

## Documentation

See `readme/BENCHMARKING_GUIDE.md` for:
- Detailed usage instructions
- Metric explanations
- Troubleshooting tips
- Advanced configuration
- CI/CD integration examples

## Support

If you encounter issues:
1. Check that all files are copied correctly
2. Verify PyTorch and GPU drivers are working
3. Review `readme/BENCHMARKING_GUIDE.md` troubleshooting section
4. Check that `utils/` is a proper Python module with `__init__.py`

---

**Ready to benchmark!** Run `bash scripts/run_full_benchmark.sh` on each system.

