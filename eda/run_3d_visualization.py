#!/usr/bin/env python3
"""
CLI wrapper for IRDS 3D visualization using configuration file.
Usage: ./run_3d_visualization [config_file]
"""
import os
import sys
import yaml
import argparse

# Add current directory to path to import irds-eda
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add project root to path for utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_path, get_config_path

try:
    from irds_eda import run_3d_visualization
except ImportError:
    # Fallback: try importing from irds-eda.py directly
    import importlib.util
    irds_eda_path = os.path.join(os.path.dirname(__file__), "irds-eda.py")
    spec = importlib.util.spec_from_file_location("irds_eda", irds_eda_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load irds-eda.py from {irds_eda_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_3d_visualization = module.run_3d_visualization


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Create a config.yaml file or specify a different config file.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)


def find_config_file():
    """Find the config file in common locations"""
    # Use utility to get config path
    config_path = get_config_path() / 'config.yaml'
    if config_path.exists():
        return str(config_path)
    
    # Fallback to common locations
    possible_paths = [
        "../config/config.yaml",  # From eda/ directory
        "config/config.yaml",     # From project root
        "./config.yaml",         # Current directory
        "config.yaml"            # Current directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Run 3D visualization for IRDS dataset using configuration file")
    parser.add_argument("config", nargs="?", default=None, help="Path to configuration file (auto-detected if not specified)")
    args = parser.parse_args()
    
    # Determine config file path
    if args.config:
        config_path = args.config
    else:
        config_path = find_config_file()
        if config_path is None:
            print("Error: No configuration file found.")
            print("Tried the following locations:")
            for path in ["../config/config.yaml", "config/config.yaml", "./config.yaml", "config.yaml"]:
                print(f"  - {path}")
            print("\nPlease specify a config file: python run_3d_visualization <path_to_config>")
            sys.exit(1)
        print(f"Using config file: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration sections with proper null handling
    data_config = config.get('data', {}) or {}
    viz_config = config.get('visualization', {}) or {}
    skeleton_config = config.get('skeleton', {}) or {}
    filter_config = config.get('filters', {}) or {}
    
    # Prepare function arguments
    kwargs = {
        # Data loading
        'folder_path': data_config.get('folder_path', '../data'),
        'file_pattern': data_config.get('file_pattern', '*.txt'),
        'has_header': data_config.get('has_header', False),
        'add_metadata': data_config.get('add_metadata', True),
        'columns': data_config.get('columns') if data_config.get('columns') is not None else None,
        'labels_path': data_config.get('labels_path', '../data/labels.csv'),
        'max_files': data_config.get('max_files', 10),
        
        # Visualization
        'x_cols': tuple(viz_config.get('x_cols')) if viz_config.get('x_cols') is not None else None,
        'frame_col': viz_config.get('frame_col') if viz_config.get('frame_col') is not None else None,
        'max_rows': viz_config.get('max_rows', 500),
        'interval_ms': viz_config.get('interval_ms', 100),
        'point_size': viz_config.get('point_size', 20),
        'elev': viz_config.get('elev', 20),
        'azim': viz_config.get('azim', -60),
        'save_path': viz_config.get('save_path') if viz_config.get('save_path') is not None else None,
        'dpi': viz_config.get('dpi', 100),
        'show': viz_config.get('show', True),
        
        # Skeleton
        'skeleton': skeleton_config.get('enabled', True),
        'num_joints': skeleton_config.get('num_joints', 25),
        'start_col': skeleton_config.get('start_col', 0),
        'order': skeleton_config.get('order', 'xyz'),
        'connect': skeleton_config.get('connect', False),
        
        # Filters
        'source_file': filter_config.get('source_file') if filter_config.get('source_file') is not None else None,
        'subject_id': filter_config.get('subject_id') if filter_config.get('subject_id') is not None else None,
        'date_id': filter_config.get('date_id') if filter_config.get('date_id') is not None else None,
        'gesture_label': filter_config.get('gesture_label') if filter_config.get('gesture_label') is not None else None,
        'rep_number': filter_config.get('rep_number') if filter_config.get('rep_number') is not None else None,
        'correct_label': filter_config.get('correct_label') if filter_config.get('correct_label') is not None else None,
        'position': filter_config.get('position') if filter_config.get('position') is not None else None,
    }
    
    # Run visualization
    print(f"Loading configuration from: {args.config}")
    run_3d_visualization(**kwargs)


if __name__ == "__main__":
    main()
