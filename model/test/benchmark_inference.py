#!/usr/bin/env python3
"""
Benchmark inference performance for gesture recognition model
Measures throughput, latency, time-to-first-prediction
"""

import torch
import time
import argparse
import sys
from pathlib import Path

# Add project paths
path_to_model = Path(__file__).parent.parent
sys.path.insert(0, str(path_to_model))
path_to_irds = Path(__file__).parent.parent.parent
sys.path.insert(0, str(path_to_irds))

from utils.benchmark import GPUBenchmark
from utils import get_logger
from model.model_architectures import get_model


def benchmark_model_inference(model_path: str, device: str = 'cuda:0', 
                              num_warmup: int = 10, num_iterations: int = 100,
                              batch_sizes: list = [1, 4, 8, 16, 32]):
    """Benchmark model inference performance"""
    
    logger = get_logger()
    benchmark = GPUBenchmark("inference_benchmark")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    architecture = checkpoint.get('architecture', 'medium')
    num_classes = len(checkpoint['gesture_names'])
    
    logger.info(f"Model architecture: {architecture}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    benchmark.start_timer('model_loading')
    model = get_model(architecture, 75, num_classes, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    benchmark.stop_timer('model_loading')
    
    from model.model_architectures import count_parameters
    param_count = count_parameters(model)
    benchmark.record_model_metrics(architecture, param_count)
    
    logger.info(f"Model loaded with {param_count:,} parameters")
    logger.info(f"Device: {device}")
    
    # Benchmark for different batch sizes
    logger.info(f"\n{'='*80}")
    logger.info("INFERENCE BENCHMARK")
    logger.info(f"{'='*80}")
    
    for batch_size in batch_sizes:
        logger.info(f"\nBatch size: {batch_size}")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 20, 75).to(device)
        
        # Warmup
        logger.info(f"  Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Synchronize GPU
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        
        # Benchmark
        logger.info(f"  Benchmarking ({num_iterations} iterations)...")
        latencies = []
        
        for i in range(num_iterations):
            # Measure time to first prediction
            iter_start = time.time()
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Synchronize to get accurate timing
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            
            iter_end = time.time()
            latency = iter_end - iter_start
            latencies.append(latency)
            
            # Record batch metrics
            if i == 0:
                # First iteration - record time to first prediction
                benchmark.record_inference_metrics(
                    batch_size=batch_size,
                    total_time=latency,
                    time_to_first_prediction=latency
                )
            else:
                benchmark.record_inference_metrics(
                    batch_size=batch_size,
                    total_time=latency
                )
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
        
        throughput = batch_size / avg_latency
        
        logger.info(f"  Results:")
        logger.info(f"    Throughput: {throughput:.1f} predictions/sec")
        logger.info(f"    Avg latency: {avg_latency*1000:.2f} ms")
        logger.info(f"    Time per prediction: {avg_latency/batch_size*1000:.2f} ms")
        logger.info(f"    P50: {p50*1000:.2f} ms")
        logger.info(f"    P95: {p95*1000:.2f} ms")
        logger.info(f"    P99: {p99*1000:.2f} ms")
        logger.info(f"    Min: {min_latency*1000:.2f} ms")
        logger.info(f"    Max: {max_latency*1000:.2f} ms")
    
    # Record memory usage
    if device.startswith('cuda'):
        benchmark.record_memory('after_inference', device)
    
    # Finalize and save
    benchmark.finalize_inference_metrics()
    benchmark.finalize()
    
    report_path = benchmark.save_report()
    logger.info(f"\nBenchmark report saved to: {report_path}")
    
    # Print summary
    benchmark.print_summary()
    
    return benchmark


def main():
    parser = argparse.ArgumentParser(description='Benchmark gesture recognition inference')
    parser.add_argument('--model', type=str, default='outputs/clip_gesture_model_pytorch.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on (cuda:0, cuda:1, cpu)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--batch-sizes', type=str, default='1,4,8,16,32',
                       help='Comma-separated list of batch sizes to test')
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    
    # Run benchmark
    benchmark_model_inference(
        model_path=args.model,
        device=args.device,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        batch_sizes=batch_sizes
    )


if __name__ == "__main__":
    main()

