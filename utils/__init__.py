"""
Utility modules for IRDS project
"""

from pathlib import Path
import os
import logging
import yaml


def get_data_path():
    """Get data path"""
    return Path(os.getcwd()) / 'data'


def get_config_path():
    """Get config path"""
    return Path(os.getcwd()) / 'config'


def setup_logger(config: dict):
    """Setup logger based on config"""
    log_level_str = config.get('debug', {}).get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
    return logging.getLogger(__name__)


def get_logger():
    """Get logger"""
    return logging.getLogger(__name__)


__all__ = ['get_data_path', 'get_config_path', 'setup_logger', 'get_logger']

