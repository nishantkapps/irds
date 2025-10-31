#!/usr/bin/env python3
"""Test script to debug path resolution"""

from pathlib import Path
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_project_root, get_data_path

print("=== Path Resolution Test ===")
print(f"utils.py location: {Path(__file__).parent / 'utils.py'}")
print(f"Project root: {get_project_root()}")
print(f"Data path: {get_data_path()}")
print(f"Data path exists: {get_data_path().exists()}")

data_path = get_data_path()
if data_path.exists():
    import glob
    txt_files = glob.glob(str(data_path / "*.txt"))
    print(f"Number of .txt files found: {len(txt_files)}")
    if txt_files:
        print(f"First file: {txt_files[0]}")
else:
    print("ERROR: Data path does not exist!")

