#!/usr/bin/env python3
"""
Generate a visual benchmark report image for presentations
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL/Pillow not available. Install with: pip install pillow")


def create_benchmark_image(benchmark_json_path: str, output_path: str = None):
    """Create a visual benchmark report image"""
    
    if not PIL_AVAILABLE:
        print("Error: Pillow library required. Install with: pip install pillow")
        return None
    
    # Load benchmark data
    with open(benchmark_json_path, 'r') as f:
        data = json.load(f)
    
    # Image settings
    width = 1400
    line_height = 35
    margin = 40
    section_spacing = 20
    
    # Prepare content lines
    lines = []
    
    # Header
    lines.append(("header", "=" * 80))
    lines.append(("header", f"BENCHMARK REPORT: {data['experiment_name']}"))
    lines.append(("header", "=" * 80))
    lines.append(("blank", ""))
    
    # System Info
    lines.append(("section", "[SYSTEM]"))
    lines.append(("content", f"  Platform: {data['system_info']['platform']}"))
    lines.append(("content", f"  PyTorch: {data['system_info']['pytorch_version']}"))
    lines.append(("blank", ""))
    
    # GPU Info
    lines.append(("section", "[GPU]"))
    if data['gpu_info']['available']:
        for gpu in data['gpu_info']['devices']:
            lines.append(("content", f"  {gpu['index']}: {gpu['name']} | {gpu['vendor']} ({gpu['platform']}) | {gpu['total_memory_gb']:.1f} GB | Compute {gpu['compute_capability']}"))
    else:
        lines.append(("content", "  No GPU available"))
    lines.append(("blank", ""))
    
    # Model Info
    if 'model' in data:
        lines.append(("section", "[MODEL]"))
        lines.append(("content", f"  Architecture: {data['model']['architecture']}"))
        lines.append(("content", f"  Parameters: {data['model']['num_parameters']:,}"))
        lines.append(("blank", ""))
    
    # Data Info
    if 'data' in data:
        lines.append(("section", "[DATA]"))
        lines.append(("content", f"  Training samples: {data['data']['train_samples']:,}"))
        lines.append(("content", f"  Test samples: {data['data']['test_samples']:,}"))
        lines.append(("content", f"  Classes: {data['data']['num_classes']}"))
        lines.append(("content", f"  Sequence length: {data['data']['sequence_length']}"))
        lines.append(("blank", ""))
    
    # Training Summary
    if 'summary' in data['training']:
        summary = data['training']['summary']
        lines.append(("section", "[TRAINING]"))
        lines.append(("content", f"  Total epochs: {summary['total_epochs']}"))
        lines.append(("content", f"  Avg epoch time: {summary['avg_epoch_time']:.2f}s"))
        lines.append(("content", f"  Initial loss: {summary['initial_loss']:.4f}"))
        lines.append(("content", f"  Final loss: {summary['final_loss']:.4f}"))
        lines.append(("content", f"  Loss improvement: {summary['loss_improvement']:.4f}"))
        lines.append(("content", f"  Initial accuracy: {summary['initial_accuracy']:.2f}%"))
        lines.append(("content", f"  Final accuracy: {summary['final_accuracy']:.2f}%"))
        lines.append(("content", f"  Accuracy improvement: {summary['accuracy_improvement']:.2f}%"))
        lines.append(("blank", ""))
    
    # Test Results
    if 'final_test_accuracy' in data['training']:
        lines.append(("section", "[TEST RESULTS]"))
        lines.append(("content", f"  Test accuracy: {data['training']['final_test_accuracy']:.2f}%"))
        lines.append(("blank", ""))
    
    # Throughput
    if data.get('throughput'):
        lines.append(("section", "[THROUGHPUT]"))
        for stage, sps in data['throughput'].items():
            lines.append(("content", f"  {stage}: {sps:.1f} samples/sec"))
        lines.append(("blank", ""))
    
    # Inference Performance
    if 'inference' in data and 'summary' in data['inference']:
        inf = data['inference']['summary']
        lines.append(("section", "[INFERENCE PERFORMANCE]"))
        lines.append(("content", f"  Total predictions: {inf['total_predictions']:,}"))
        lines.append(("content", f"  Total batches: {inf['total_batches']}"))
        lines.append(("content", f"  Avg predictions/sec: {inf['avg_predictions_per_second']:.1f}"))
        lines.append(("content", f"  Avg time per prediction: {inf['avg_time_per_prediction']*1000:.2f}ms"))
        lines.append(("content", f"  P50 latency: {inf['p50_latency']*1000:.2f}ms"))
        lines.append(("content", f"  P95 latency: {inf['p95_latency']*1000:.2f}ms"))
        lines.append(("content", f"  P99 latency: {inf['p99_latency']*1000:.2f}ms"))
        lines.append(("content", f"  Min latency: {inf['min_latency']*1000:.2f}ms"))
        lines.append(("content", f"  Max latency: {inf['max_latency']*1000:.2f}ms"))
        if 'avg_time_to_first_prediction' in inf:
            lines.append(("content", f"  Avg time to first prediction: {inf['avg_time_to_first_prediction']*1000:.2f}ms"))
        lines.append(("blank", ""))
    
    # GPU Memory
    if data.get('memory'):
        lines.append(("section", "[GPU MEMORY]"))
        memory_order = ['initial', 'after_data_transfer', 'after_model_creation', 'after_training']
        for stage in memory_order:
            if stage in data['memory']:
                mem = data['memory'][stage]
                if 'error' not in mem:
                    stage_name = stage.replace('_', ' ').title()
                    lines.append(("content", f"  {stage_name}: Allocated {mem['allocated_gb']:.2f} GB | Reserved {mem['reserved_gb']:.2f} GB | Peak {mem['max_allocated_gb']:.2f} GB"))
        lines.append(("blank", ""))
    
    # Detailed Timings
    if data.get('timings'):
        lines.append(("section", "[DETAILED TIMINGS]"))
        timing_order = ['data_loading', 'data_preparation', 'gpu_transfer', 'model_creation', 'total_training']
        for name in timing_order:
            if name in data['timings']:
                duration = data['timings'][name]
                lines.append(("content", f"  {name.replace('_', ' ').title()}: {duration:.2f}s ({duration/60:.2f}m)"))
        lines.append(("blank", ""))
    
    # Total Runtime
    lines.append(("section", "[TOTAL RUNTIME]"))
    lines.append(("content", f"  {data['total_runtime_seconds']:.2f}s ({data['total_runtime_minutes']:.2f}m)"))
    lines.append(("blank", ""))
    lines.append(("header", "=" * 80))
    
    # Calculate image height
    height = margin * 2 + len(lines) * line_height
    
    # Create image
    img = Image.new('RGB', (width, height), color='#1e1e1e')  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Try to use a monospace font, fallback to default
    try:
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
        font_section = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
        font_content = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except:
        try:
            font_header = ImageFont.truetype("courier", 16)
            font_section = ImageFont.truetype("courier", 14)
            font_content = ImageFont.truetype("courier", 13)
        except:
            font_header = ImageFont.load_default()
            font_section = ImageFont.load_default()
            font_content = ImageFont.load_default()
    
    # Color scheme
    colors = {
        'header': '#00d4ff',      # Cyan
        'section': '#00ff9f',     # Green
        'content': '#e0e0e0',     # Light gray
        'blank': '#1e1e1e'        # Background
    }
    
    # Draw lines
    y = margin
    for line_type, text in lines:
        if line_type == 'blank':
            y += line_height // 2
            continue
        
        if line_type == 'header':
            draw.text((margin, y), text, fill=colors['header'], font=font_header)
        elif line_type == 'section':
            draw.text((margin, y), text, fill=colors['section'], font=font_section)
        elif line_type == 'content':
            draw.text((margin, y), text, fill=colors['content'], font=font_content)
        
        y += line_height
    
    # Save image
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        gpu_type = 'unknown'
        if data['gpu_info']['available']:
            gpu_type = data['gpu_info']['devices'][0]['vendor'].lower()
        output_path = f"outputs/benchmark_report_{gpu_type}_{timestamp}.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, 'PNG')
    
    print(f"Benchmark image saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark report image')
    parser.add_argument('benchmark_json', help='Path to benchmark JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (default: auto-generate)')
    
    args = parser.parse_args()
    
    create_benchmark_image(args.benchmark_json, args.output)


if __name__ == "__main__":
    main()

