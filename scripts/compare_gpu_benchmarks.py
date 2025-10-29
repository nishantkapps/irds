#!/usr/bin/env python3
"""
Compare benchmark results between AMD (ROCm) and NVIDIA (CUDA) GPUs
Generate comprehensive comparison report
"""

import argparse
import sys
from pathlib import Path

# Add project paths
path_to_irds = Path(__file__).parent.parent
sys.path.insert(0, str(path_to_irds))

from utils.benchmark import compare_benchmarks


def main():
    parser = argparse.ArgumentParser(description='Compare GPU benchmarks')
    parser.add_argument('benchmarks', nargs='+', 
                       help='Paths to benchmark JSON files to compare')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for comparison report')
    
    args = parser.parse_args()
    
    print(f"\nComparing {len(args.benchmarks)} benchmark(s):")
    for i, bm in enumerate(args.benchmarks, 1):
        print(f"  {i}. {bm}")
    
    # Run comparison
    comparison, output_path = compare_benchmarks(args.benchmarks, args.output)
    
    print(f"\nComparison complete!")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()

