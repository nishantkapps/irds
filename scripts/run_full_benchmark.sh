#!/bin/bash
# Complete benchmark workflow for AMD vs NVIDIA comparison
# Run this script on both AMD and NVIDIA systems

set -e

echo "========================================="
echo "GPU Benchmark Suite for Gesture Recognition"
echo "========================================="
echo ""

# Detect GPU type
if command -v rocm-smi &> /dev/null; then
    GPU_TYPE="amd"
    echo "Detected AMD GPU (ROCm)"
    rocm-smi --showproductname
elif command -v nvidia-smi &> /dev/null; then
    GPU_TYPE="nvidia"
    echo "Detected NVIDIA GPU (CUDA)"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    GPU_TYPE="unknown"
    echo "Warning: No GPU detected, using CPU"
fi

echo ""
echo "========================================="
echo "Step 1: Training Benchmark"
echo "========================================="

# Run training with benchmark
python model/train/train_pytorch_only.py --config config/config_high_accuracy.yaml

echo ""
echo "========================================="
echo "Step 2: Inference Benchmark"
echo "========================================="

# Run inference benchmark for multiple batch sizes
python model/test/benchmark_inference.py \
    --model outputs/clip_gesture_model_pytorch.pth \
    --device cuda:0 \
    --warmup 20 \
    --iterations 200 \
    --batch-sizes 1,2,4,8,16,32,64

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="

# Find latest benchmark files
TRAINING_BENCHMARK=$(ls -t outputs/benchmark_${GPU_TYPE}_*.json 2>/dev/null | head -1)
TRAINING_TXT=$(ls -t outputs/benchmark_report_${GPU_TYPE}_*.txt 2>/dev/null | head -1)
INFERENCE_TXT=$(ls -t outputs/benchmark_report_${GPU_TYPE}_*.txt 2>/dev/null | tail -1)

echo ""
echo "========================================="
echo "Step 3: Creating Combined Report"
echo "========================================="

# Create combined report
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMBINED_REPORT="outputs/benchmark_combined_${GPU_TYPE}_${TIMESTAMP}.txt"

if [ -f "$TRAINING_TXT" ]; then
    # Copy training report
    cat "$TRAINING_TXT" > "$COMBINED_REPORT"
    
    # If inference benchmark was run separately, append it
    if [ -f "$INFERENCE_TXT" ] && [ "$TRAINING_TXT" != "$INFERENCE_TXT" ]; then
        echo "" >> "$COMBINED_REPORT"
        echo "=========================================" >> "$COMBINED_REPORT"
        echo "INFERENCE BENCHMARK (SEPARATE RUN)" >> "$COMBINED_REPORT"
        echo "=========================================" >> "$COMBINED_REPORT"
        grep -A 100 "INFERENCE PERFORMANCE" "$INFERENCE_TXT" >> "$COMBINED_REPORT" 2>/dev/null || true
    fi
    
    echo "Combined report created: $COMBINED_REPORT"
fi

echo ""
echo "Benchmark files created:"
echo "  Training JSON: $TRAINING_BENCHMARK"
echo "  Training TXT: $TRAINING_TXT"
echo "  Combined Report: $COMBINED_REPORT"
echo ""
echo "To compare with another GPU:"
echo "  1. Copy benchmark files from this system"
echo "  2. Run this script on the other GPU system"
echo "  3. Run: python scripts/compare_gpu_benchmarks.py <file1.json> <file2.json>"
echo ""

