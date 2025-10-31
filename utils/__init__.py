"""
Utility modules for IRDS project
"""

from pathlib import Path
import os
import logging
import yaml


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    This function finds the project root by looking for the 'data' directory
    that contains the IRDS dataset files.
    
    Returns:
        Path: The project root directory
    """
    # Start from current file location
    current_dir = Path(__file__).parent
    
    # Look for the 'data' directory going up the directory tree
    for parent in [current_dir] + list(current_dir.parents):
        data_dir = parent / 'data'
        if data_dir.exists():
            # Check if it has .txt files (skeleton data) or labels.csv
            txt_files = list(data_dir.glob('*.txt'))
            if txt_files or (data_dir / 'labels.csv').exists():
                return parent
    
    # Fallback: assume we're in the project root
    return current_dir


def get_data_path() -> Path:
    """
    Get the path to the data directory.
    
    Returns:
        Path: The data directory path
    """
    return get_project_root() / 'data'


def get_config_path() -> Path:
    """
    Get the path to the config directory.
    
    Returns:
        Path: The config directory path
    """
    return get_project_root() / 'config'


def setup_logger(config: dict):
    """Setup logger based on config"""
    log_level_str = config.get('debug', {}).get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
    return logging.getLogger(__name__)


def get_logger():
    """Get logger"""
    return logging.getLogger(__name__)


__all__ = ['get_project_root', 'get_data_path', 'get_config_path', 'setup_logger', 'get_logger']

