"""
Utility functions for the IRDS project.
"""
import os
import time
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any


class LogLevel(Enum):
    """Log levels in order of priority"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    OFF = 4


class Logger:
    """Simple logger for the IRDS project"""
    
    def __init__(self, level: str = "Info", show_tensor_info: bool = True, 
                 show_memory_usage: bool = True, show_timing: bool = True):
        self.level = LogLevel[level.upper()]
        self.show_tensor_info = show_tensor_info
        self.show_memory_usage = show_memory_usage
        self.show_timing = show_timing
        self.start_times = {}
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on current level"""
        return level.value >= self.level.value
    
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        if self._should_log(LogLevel.DEBUG):
            print(f"[DEBUG] {message}")
    
    def info(self, message: str, **kwargs):
        """Info level logging"""
        if self._should_log(LogLevel.INFO):
            print(f"[INFO] {message}")
    
    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        if self._should_log(LogLevel.WARNING):
            print(f"[WARNING] {message}")
    
    def error(self, message: str, **kwargs):
        """Error level logging"""
        if self._should_log(LogLevel.ERROR):
            print(f"[ERROR] {message}")
    
    def tensor_info(self, tensor, name: str = "tensor"):
        """Log tensor information if enabled"""
        if self.show_tensor_info and self._should_log(LogLevel.INFO):
            print(f"[TENSOR] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    def memory_usage(self, tensor, name: str = "tensor"):
        """Log memory usage if enabled"""
        if self.show_memory_usage and self._should_log(LogLevel.INFO):
            memory_gb = tensor.element_size() * tensor.nelement() / 1e9
            print(f"[MEMORY] {name}: {memory_gb:.2f} GB")
    
    def start_timer(self, name: str):
        """Start a timer"""
        if self.show_timing:
            self.start_times[name] = time.time()
    
    def end_timer(self, name: str):
        """End a timer and log duration"""
        if self.show_timing and name in self.start_times:
            duration = time.time() - self.start_times[name]
            if self._should_log(LogLevel.INFO):
                print(f"[TIMING] {name}: {duration:.2f}s")
            del self.start_times[name]


# Global logger instance
_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """Get the global logger instance"""
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger


def setup_logger(config: Dict[str, Any]) -> Logger:
    """Setup logger from config"""
    global _logger
    debug_config = config.get('debug', {})
    _logger = Logger(
        level=debug_config.get('level', 'Info'),
        show_tensor_info=debug_config.get('show_tensor_info', True),
        show_memory_usage=debug_config.get('show_memory_usage', True),
        show_timing=debug_config.get('show_timing', True)
    )
    return _logger


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
