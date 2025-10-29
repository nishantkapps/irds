#!/usr/bin/env python3
"""
GPU Benchmark Utilities
Tracks performance metrics for AMD (ROCm) vs NVIDIA (CUDA) GPUs
"""

import time
import torch
import json
import platform
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class GPUBenchmark:
    """Track and report GPU performance metrics"""
    
    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        self.metrics = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'gpu_info': self._get_gpu_info(),
            'timings': {},
            'memory': {},
            'training': {},
            'throughput': {}
        }
        self.timers = {}
        self.start_time = time.time()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Collect system information"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Collect GPU information"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        gpu_count = torch.cuda.device_count()
        gpus = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': props.total_memory / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count,
            }
            
            # Detect GPU vendor
            gpu_name = gpu_info['name'].lower()
            if 'radeon' in gpu_name or 'amd' in gpu_name:
                gpu_info['vendor'] = 'AMD'
                gpu_info['platform'] = 'ROCm'
            elif 'nvidia' in gpu_name or 'geforce' in gpu_name or 'tesla' in gpu_name or 'quadro' in gpu_name:
                gpu_info['vendor'] = 'NVIDIA'
                gpu_info['platform'] = 'CUDA'
            else:
                gpu_info['vendor'] = 'Unknown'
                gpu_info['platform'] = 'Unknown'
            
            gpus.append(gpu_info)
        
        return {
            'available': True,
            'count': gpu_count,
            'devices': gpus
        }
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = {'start': time.time()}
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time"""
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.timers[name]['start']
        self.timers[name]['elapsed'] = elapsed
        self.metrics['timings'][name] = elapsed
        return elapsed
    
    def record_memory(self, stage: str, device: str = 'cuda:0'):
        """Record GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        try:
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            
            self.metrics['memory'][stage] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            }
        except Exception as e:
            self.metrics['memory'][stage] = {'error': str(e)}
    
    def record_training_metrics(self, epoch: int, loss: float, accuracy: float, 
                                learning_rate: float = None):
        """Record training metrics for an epoch"""
        if 'epochs' not in self.metrics['training']:
            self.metrics['training']['epochs'] = []
        
        epoch_data = {
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': time.time() - self.start_time
        }
        
        if learning_rate is not None:
            epoch_data['learning_rate'] = learning_rate
        
        self.metrics['training']['epochs'].append(epoch_data)
    
    def record_test_metrics(self, test_accuracy: float, test_loss: float = None):
        """Record final test metrics"""
        self.metrics['training']['final_test_accuracy'] = test_accuracy
        if test_loss is not None:
            self.metrics['training']['final_test_loss'] = test_loss
    
    def record_throughput(self, samples_per_second: float, stage: str = 'training'):
        """Record throughput metrics"""
        self.metrics['throughput'][stage] = samples_per_second
    
    def record_inference_metrics(self, batch_size: int, total_time: float, 
                                time_to_first_prediction: float = None,
                                predictions_per_second: float = None):
        """Record inference performance metrics (similar to LLM metrics)
        
        Args:
            batch_size: Number of samples in batch
            total_time: Total inference time in seconds
            time_to_first_prediction: Time until first prediction available (TTFP)
            predictions_per_second: Throughput metric
        """
        if 'inference' not in self.metrics:
            self.metrics['inference'] = {
                'batches': [],
                'summary': {}
            }
        
        batch_metrics = {
            'batch_size': batch_size,
            'total_time': total_time,
            'time_per_prediction': total_time / batch_size if batch_size > 0 else 0,
            'predictions_per_second': batch_size / total_time if total_time > 0 else 0
        }
        
        if time_to_first_prediction is not None:
            batch_metrics['time_to_first_prediction'] = time_to_first_prediction
        
        if predictions_per_second is not None:
            batch_metrics['predictions_per_second'] = predictions_per_second
        
        self.metrics['inference']['batches'].append(batch_metrics)
    
    def finalize_inference_metrics(self):
        """Calculate summary statistics for inference"""
        if 'inference' not in self.metrics or not self.metrics['inference']['batches']:
            return
        
        batches = self.metrics['inference']['batches']
        
        # Calculate averages
        avg_total_time = sum(b['total_time'] for b in batches) / len(batches)
        avg_time_per_pred = sum(b['time_per_prediction'] for b in batches) / len(batches)
        avg_preds_per_sec = sum(b['predictions_per_second'] for b in batches) / len(batches)
        
        # Calculate p50, p95, p99 latencies
        times = sorted([b['time_per_prediction'] for b in batches])
        p50_idx = len(times) // 2
        p95_idx = int(len(times) * 0.95)
        p99_idx = int(len(times) * 0.99)
        
        self.metrics['inference']['summary'] = {
            'total_batches': len(batches),
            'total_predictions': sum(b['batch_size'] for b in batches),
            'avg_batch_time': avg_total_time,
            'avg_time_per_prediction': avg_time_per_pred,
            'avg_predictions_per_second': avg_preds_per_sec,
            'p50_latency': times[p50_idx] if times else 0,
            'p95_latency': times[p95_idx] if p95_idx < len(times) else times[-1] if times else 0,
            'p99_latency': times[p99_idx] if p99_idx < len(times) else times[-1] if times else 0,
            'min_latency': min(times) if times else 0,
            'max_latency': max(times) if times else 0
        }
        
        # Time to first prediction
        ttfp_times = [b['time_to_first_prediction'] for b in batches if 'time_to_first_prediction' in b]
        if ttfp_times:
            self.metrics['inference']['summary']['avg_time_to_first_prediction'] = sum(ttfp_times) / len(ttfp_times)
    
    def record_data_metrics(self, train_samples: int, test_samples: int, 
                          num_classes: int, sequence_length: int):
        """Record dataset metrics"""
        self.metrics['data'] = {
            'train_samples': train_samples,
            'test_samples': test_samples,
            'total_samples': train_samples + test_samples,
            'num_classes': num_classes,
            'sequence_length': sequence_length
        }
    
    def record_model_metrics(self, architecture: str, num_parameters: int):
        """Record model architecture metrics"""
        self.metrics['model'] = {
            'architecture': architecture,
            'num_parameters': num_parameters
        }
    
    def finalize(self):
        """Finalize benchmark - calculate summary statistics"""
        total_time = time.time() - self.start_time
        self.metrics['total_runtime_seconds'] = total_time
        self.metrics['total_runtime_minutes'] = total_time / 60
        
        # Calculate training statistics
        if 'epochs' in self.metrics['training']:
            epochs = self.metrics['training']['epochs']
            if epochs:
                final_epoch = epochs[-1]
                initial_epoch = epochs[0]
                
                self.metrics['training']['summary'] = {
                    'initial_loss': initial_epoch['loss'],
                    'final_loss': final_epoch['loss'],
                    'loss_improvement': initial_epoch['loss'] - final_epoch['loss'],
                    'initial_accuracy': initial_epoch['accuracy'],
                    'final_accuracy': final_epoch['accuracy'],
                    'accuracy_improvement': final_epoch['accuracy'] - initial_epoch['accuracy'],
                    'total_epochs': len(epochs),
                    'avg_epoch_time': total_time / len(epochs) if len(epochs) > 0 else 0
                }
        
        # Finalize inference metrics if available
        self.finalize_inference_metrics()
    
    def save_report(self, output_path: str = None):
        """Save benchmark report to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            gpu_type = 'unknown'
            if self.metrics['gpu_info']['available']:
                gpu_type = self.metrics['gpu_info']['devices'][0]['vendor'].lower()
            output_path = f"outputs/benchmark_{gpu_type}_{timestamp}.json"
        
        # Ensure outputs directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return output_path
    
    def print_summary(self, save_to_file: bool = True):
        """Print a summary of benchmark results"""
        output_lines = []
        
        def print_and_save(text):
            """Helper to both print and save to list"""
            print(text)
            output_lines.append(text)
        
        print_and_save("\n" + "="*80)
        print_and_save(f"BENCHMARK REPORT: {self.experiment_name}")
        print_and_save("="*80)
        
        # System Info
        print_and_save("\n[SYSTEM]")
        print_and_save(f"  Platform: {self.metrics['system_info']['platform']}")
        print_and_save(f"  PyTorch: {self.metrics['system_info']['pytorch_version']}")
        
        # GPU Info
        print_and_save("\n[GPU]")
        if self.metrics['gpu_info']['available']:
            for gpu in self.metrics['gpu_info']['devices']:
                print_and_save(f"  {gpu['index']}: {gpu['name']} | {gpu['vendor']} ({gpu['platform']}) | {gpu['total_memory_gb']:.1f} GB | Compute {gpu['compute_capability']}")
        else:
            print_and_save("  No GPU available")
        
        # Model Info
        if 'model' in self.metrics:
            print_and_save("\n[MODEL]")
            print_and_save(f"  Architecture: {self.metrics['model']['architecture']}")
            print_and_save(f"  Parameters: {self.metrics['model']['num_parameters']:,}")
        
        # Data Info
        if 'data' in self.metrics:
            print_and_save("\n[DATA]")
            print_and_save(f"  Training samples: {self.metrics['data']['train_samples']:,}")
            print_and_save(f"  Test samples: {self.metrics['data']['test_samples']:,}")
            print_and_save(f"  Classes: {self.metrics['data']['num_classes']}")
            print_and_save(f"  Sequence length: {self.metrics['data']['sequence_length']}")
        
        # Training Summary
        if 'summary' in self.metrics['training']:
            summary = self.metrics['training']['summary']
            print_and_save("\n[TRAINING]")
            print_and_save(f"  Total epochs: {summary['total_epochs']}")
            print_and_save(f"  Avg epoch time: {summary['avg_epoch_time']:.2f}s")
            print_and_save(f"  Initial loss: {summary['initial_loss']:.4f}")
            print_and_save(f"  Final loss: {summary['final_loss']:.4f}")
            print_and_save(f"  Loss improvement: {summary['loss_improvement']:.4f}")
            print_and_save(f"  Initial accuracy: {summary['initial_accuracy']:.2f}%")
            print_and_save(f"  Final accuracy: {summary['final_accuracy']:.2f}%")
            print_and_save(f"  Accuracy improvement: {summary['accuracy_improvement']:.2f}%")
        
        # Test Results
        if 'final_test_accuracy' in self.metrics['training']:
            print_and_save(f"\n[TEST RESULTS]")
            print_and_save(f"  Test accuracy: {self.metrics['training']['final_test_accuracy']:.2f}%")
        
        # Throughput
        if self.metrics['throughput']:
            print_and_save("\n[THROUGHPUT]")
            for stage, sps in self.metrics['throughput'].items():
                print_and_save(f"  {stage}: {sps:.1f} samples/sec")
        
        # Inference Metrics
        if 'inference' in self.metrics and 'summary' in self.metrics['inference']:
            inf = self.metrics['inference']['summary']
            print_and_save("\n[INFERENCE PERFORMANCE]")
            print_and_save(f"  Total predictions: {inf['total_predictions']:,}")
            print_and_save(f"  Total batches: {inf['total_batches']}")
            print_and_save(f"  Avg predictions/sec: {inf['avg_predictions_per_second']:.1f}")
            print_and_save(f"  Avg time per prediction: {inf['avg_time_per_prediction']*1000:.2f}ms")
            print_and_save(f"  P50 latency: {inf['p50_latency']*1000:.2f}ms")
            print_and_save(f"  P95 latency: {inf['p95_latency']*1000:.2f}ms")
            print_and_save(f"  P99 latency: {inf['p99_latency']*1000:.2f}ms")
            print_and_save(f"  Min latency: {inf['min_latency']*1000:.2f}ms")
            print_and_save(f"  Max latency: {inf['max_latency']*1000:.2f}ms")
            if 'avg_time_to_first_prediction' in inf:
                print_and_save(f"  Avg time to first prediction: {inf['avg_time_to_first_prediction']*1000:.2f}ms")
        
        # Memory Usage
        if self.metrics['memory']:
            print_and_save("\n[GPU MEMORY]")
            memory_order = ['initial', 'after_data_transfer', 'after_model_creation', 'after_training']
            
            # Print in order
            printed = set()
            for stage in memory_order:
                if stage in self.metrics['memory']:
                    mem = self.metrics['memory'][stage]
                    if 'error' not in mem:
                        print_and_save(f"  {stage.replace('_', ' ').title()}: Allocated {mem['allocated_gb']:.2f} GB | Reserved {mem['reserved_gb']:.2f} GB | Peak {mem['max_allocated_gb']:.2f} GB")
                    printed.add(stage)
            
            # Print any remaining memory stages
            for stage, mem in self.metrics['memory'].items():
                if stage not in printed and 'error' not in mem:
                    print_and_save(f"  {stage.replace('_', ' ').title()}: Allocated {mem['allocated_gb']:.2f} GB | Reserved {mem['reserved_gb']:.2f} GB | Peak {mem['max_allocated_gb']:.2f} GB")
        
        # Detailed Timings
        if self.metrics['timings']:
            print_and_save("\n[DETAILED TIMINGS]")
            timing_order = ['data_loading', 'data_preparation', 'gpu_transfer', 'model_creation', 'total_training']
            
            # Print in order, then any remaining
            printed = set()
            for name in timing_order:
                if name in self.metrics['timings']:
                    duration = self.metrics['timings'][name]
                    print_and_save(f"  {name.replace('_', ' ').title()}: {duration:.2f}s ({duration/60:.2f}m)")
                    printed.add(name)
            
            # Print any remaining timings not in the order list
            for name, duration in self.metrics['timings'].items():
                if name not in printed:
                    print_and_save(f"  {name.replace('_', ' ').title()}: {duration:.2f}s ({duration/60:.2f}m)")
        
        # Total Runtime
        print_and_save(f"\n[TOTAL RUNTIME]")
        print_and_save(f"  {self.metrics['total_runtime_seconds']:.2f}s ({self.metrics['total_runtime_minutes']:.2f}m)")
        
        print_and_save("\n" + "="*80 + "\n")
        
        # Save to text file with timestamp
        if save_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            gpu_type = 'unknown'
            if self.metrics['gpu_info']['available']:
                gpu_type = self.metrics['gpu_info']['devices'][0]['vendor'].lower()
            
            text_output_path = f"outputs/benchmark_report_{gpu_type}_{timestamp}.txt"
            Path(text_output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(text_output_path, 'w') as f:
                f.write('\n'.join(output_lines))
            
            print(f"Text report saved to: {text_output_path}")
            return text_output_path


def compare_benchmarks(benchmark_files: list, output_path: str = None):
    """Compare multiple benchmark files and generate comparison report"""
    
    benchmarks = []
    for file_path in benchmark_files:
        with open(file_path, 'r') as f:
            benchmarks.append(json.load(f))
    
    # Create comparison
    comparison = {
        'comparison_date': datetime.now().isoformat(),
        'num_benchmarks': len(benchmarks),
        'benchmarks': []
    }
    
    for bm in benchmarks:
        gpu_name = 'CPU'
        gpu_vendor = 'N/A'
        if bm['gpu_info']['available']:
            gpu_name = bm['gpu_info']['devices'][0]['name']
            gpu_vendor = bm['gpu_info']['devices'][0]['vendor']
        
        summary = {
            'experiment': bm['experiment_name'],
            'timestamp': bm['timestamp'],
            'gpu_vendor': gpu_vendor,
            'gpu_name': gpu_name,
            'total_runtime_minutes': bm.get('total_runtime_minutes', 0),
            'model_parameters': bm.get('model', {}).get('num_parameters', 0),
            'architecture': bm.get('model', {}).get('architecture', 'unknown'),
        }
        
        if 'summary' in bm.get('training', {}):
            summary.update({
                'total_epochs': bm['training']['summary']['total_epochs'],
                'avg_epoch_time': bm['training']['summary']['avg_epoch_time'],
                'final_train_accuracy': bm['training']['summary']['final_accuracy'],
                'test_accuracy': bm['training'].get('final_test_accuracy', 0),
            })
        
        if bm.get('throughput'):
            summary['throughput_samples_per_sec'] = bm['throughput'].get('training', 0)
        
        comparison['benchmarks'].append(summary)
    
    # Calculate speedup ratios
    if len(comparison['benchmarks']) == 2:
        bm1, bm2 = comparison['benchmarks']
        
        if bm1['total_runtime_minutes'] > 0 and bm2['total_runtime_minutes'] > 0:
            comparison['speedup'] = {
                f"{bm1['gpu_vendor']}_vs_{bm2['gpu_vendor']}": {
                    'time_ratio': bm1['total_runtime_minutes'] / bm2['total_runtime_minutes'],
                    'faster_gpu': bm1['gpu_vendor'] if bm1['total_runtime_minutes'] < bm2['total_runtime_minutes'] else bm2['gpu_vendor']
                }
            }
        
        if bm1.get('throughput_samples_per_sec', 0) > 0 and bm2.get('throughput_samples_per_sec', 0) > 0:
            comparison['throughput_comparison'] = {
                f"{bm1['gpu_vendor']}_vs_{bm2['gpu_vendor']}": {
                    'throughput_ratio': bm1['throughput_samples_per_sec'] / bm2['throughput_samples_per_sec'],
                    'faster_gpu': bm1['gpu_vendor'] if bm1['throughput_samples_per_sec'] > bm2['throughput_samples_per_sec'] else bm2['gpu_vendor']
                }
            }
    
    # Save comparison report
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"outputs/benchmark_comparison_{timestamp}.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    for i, bm in enumerate(comparison['benchmarks'], 1):
        print(f"\n[BENCHMARK {i}]")
        print(f"  Experiment: {bm['experiment']}")
        print(f"  GPU: {bm['gpu_vendor']} - {bm['gpu_name']}")
        print(f"  Model: {bm['architecture']} ({bm['model_parameters']:,} params)")
        print(f"  Runtime: {bm['total_runtime_minutes']:.2f} minutes")
        if 'avg_epoch_time' in bm:
            print(f"  Avg epoch time: {bm['avg_epoch_time']:.2f}s")
            print(f"  Train accuracy: {bm['final_train_accuracy']:.2f}%")
            print(f"  Test accuracy: {bm['test_accuracy']:.2f}%")
        if 'throughput_samples_per_sec' in bm:
            print(f"  Throughput: {bm['throughput_samples_per_sec']:.1f} samples/sec")
    
    if 'speedup' in comparison:
        print(f"\n[SPEEDUP ANALYSIS]")
        for key, data in comparison['speedup'].items():
            print(f"  {key}:")
            print(f"    Time ratio: {data['time_ratio']:.2f}x")
            print(f"    Faster GPU: {data['faster_gpu']}")
    
    if 'throughput_comparison' in comparison:
        print(f"\n[THROUGHPUT ANALYSIS]")
        for key, data in comparison['throughput_comparison'].items():
            print(f"  {key}:")
            print(f"    Throughput ratio: {data['throughput_ratio']:.2f}x")
            print(f"    Faster GPU: {data['faster_gpu']}")
    
    print("\n" + "="*80 + "\n")
    print(f"Comparison report saved to: {output_path}\n")
    
    return comparison, output_path


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Compare mode
        benchmark_files = sys.argv[1:]
        compare_benchmarks(benchmark_files)
    else:
        print("Usage:")
        print("  Compare benchmarks: python benchmark.py <file1.json> <file2.json> ...")

