#!/usr/bin/env python3
"""
Compare results from multiple gesture recognition experiments
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_logger

def load_experiment_results(outputs_dir: str = "outputs") -> List[Dict]:
    """Load all experiment info files from outputs directory"""
    logger = get_logger()
    results = []
    
    if not os.path.exists(outputs_dir):
        logger.warning(f"Outputs directory not found: {outputs_dir}")
        return results
    
    # Find all info JSON files
    info_files = [f for f in os.listdir(outputs_dir) if f.startswith("info_") and f.endswith(".json")]
    
    for info_file in info_files:
        file_path = os.path.join(outputs_dir, info_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract model name from filename
                model_name = info_file.replace("info_", "").replace(".json", "")
                data['model_name'] = model_name
                results.append(data)
                logger.info(f"Loaded results for: {model_name}")
        except Exception as e:
            logger.error(f"Error loading {info_file}: {e}")
    
    return results

def print_comparison_table(results: List[Dict]):
    """Print formatted comparison table"""
    logger = get_logger()
    
    if not results:
        logger.warning("No experiment results found")
        return
    
    # Sort by model name
    results = sorted(results, key=lambda x: x.get('model_name', ''))
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON")
    print("="*100)
    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Train Loss':<12} {'Test Loss':<12} {'Params':<12}")
    print("-"*100)
    
    for result in results:
        model_name = result.get('model_name', 'unknown')
        train_acc = result.get('train_accuracy', 0) * 100
        test_acc = result.get('test_accuracy', 0) * 100
        train_loss = result.get('train_loss', 0)
        test_loss = result.get('test_loss', 0)
        
        # Try to get parameter count from model info
        params = "N/A"
        
        print(f"{model_name:<15} {train_acc:>10.2f}% {test_acc:>10.2f}% {train_loss:>11.4f} {test_loss:>11.4f} {params:<12}")
    
    print("="*100)
    
    # Find best model
    best_test_acc = max(results, key=lambda x: x.get('test_accuracy', 0))
    print(f"\n✓ Best Test Accuracy: {best_test_acc['model_name']} "
          f"({best_test_acc.get('test_accuracy', 0)*100:.2f}%)")
    
    # Check for overfitting
    print("\n" + "="*100)
    print("OVERFITTING ANALYSIS")
    print("="*100)
    print(f"{'Model':<15} {'Gap (Train-Test)':<20} {'Status':<20}")
    print("-"*100)
    
    for result in results:
        model_name = result.get('model_name', 'unknown')
        train_acc = result.get('train_accuracy', 0) * 100
        test_acc = result.get('test_accuracy', 0) * 100
        gap = train_acc - test_acc
        
        if gap < 5:
            status = "✓ Well-balanced"
        elif gap < 10:
            status = "⚠ Slight overfit"
        elif gap < 20:
            status = "⚠⚠ Moderate overfit"
        else:
            status = "✗✗ Severe overfit"
        
        print(f"{model_name:<15} {gap:>9.2f}% {status:<20}")
    
    print("="*100)

def main():
    """Main function"""
    logger = get_logger()
    logger.info("Loading experiment results...")
    
    # Load results
    results = load_experiment_results()
    
    if not results:
        logger.error("No experiment results found in outputs/ directory")
        logger.info("Run experiments first using: ./run_experiments.sh")
        return
    
    # Print comparison
    print_comparison_table(results)
    
    logger.info(f"\nTotal experiments: {len(results)}")

if __name__ == "__main__":
    main()

