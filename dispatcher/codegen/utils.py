#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Utility Functions for GEMM Codegen

Common helper functions used across the codegen system.
"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache
import json

log = logging.getLogger(__name__)


# ============================================================================
# Path Utilities
# ============================================================================

@lru_cache(None)
def get_project_root() -> Path:
    """Get composable_kernel project root directory"""
    # Start from this file and go up until we find CMakeLists.txt
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "CMakeLists.txt").exists():
            return current
        current = current.parent
    
    # Fallback: assume we're in dispatcher/codegen
    return Path(__file__).parent.parent.parent


@lru_cache(None)
def get_library_path() -> Optional[Path]:
    """Get CK library path"""
    root = get_project_root()
    
    # Try common locations
    candidates = [
        root / "library",
        root / "build" / "library",
        Path(os.environ.get("CK_LIBRARY_PATH", "")),
        Path("/opt/rocm/composable_kernel/library"),
    ]
    
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    
    return None


@lru_cache(None)
def get_tile_engine_path() -> Optional[Path]:
    """Get tile_engine path"""
    root = get_project_root()
    tile_engine = root / "tile_engine"
    
    if tile_engine.exists():
        return tile_engine
    
    return None


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if needed"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# String Utilities
# ============================================================================

def sanitize_identifier(name: str) -> str:
    """Sanitize string to be valid C++ identifier"""
    # Replace invalid characters with underscore
    sanitized = ""
    for char in name:
        if char.isalnum() or char == '_':
            sanitized += char
        else:
            sanitized += '_'
    
    # Ensure doesn't start with digit
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    
    return sanitized


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case"""
    import re
    # Insert underscore before uppercase letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before uppercase letters preceded by lowercase
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase"""
    components = name.split('_')
    return ''.join(x.title() for x in components)


def generate_hash(content: str, length: int = 8) -> str:
    """Generate short hash of content"""
    return hashlib.sha256(content.encode()).hexdigest()[:length]


# ============================================================================
# File Utilities
# ============================================================================

def read_json(path: Path) -> Dict:
    """Read JSON file with error handling"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"File not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in {path}: {e}")
        return {}
    except Exception as e:
        log.error(f"Failed to read {path}: {e}")
        return {}


def write_json(data: Dict, path: Path, indent: int = 2):
    """Write JSON file with error handling"""
    try:
        ensure_dir(path.parent)
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        log.debug(f"Wrote JSON to {path}")
    except Exception as e:
        log.error(f"Failed to write {path}: {e}")


def atomic_write(content: str, path: Path):
    """
    Atomically write file (write to temp, then rename)
    
    Prevents partial writes if process is interrupted.
    """
    import tempfile
    
    ensure_dir(path.parent)
    
    # Write to temporary file
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )
    
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        
        # Atomic rename
        os.replace(temp_path, path)
        log.debug(f"Atomically wrote {path}")
        
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_datatype(dtype: str) -> bool:
    """Validate datatype string"""
    valid_types = ['fp16', 'bf16', 'fp32', 'fp8', 'bf8', 'int8']
    return dtype.lower() in valid_types


def validate_layout(layout: str) -> bool:
    """Validate layout string"""
    if len(layout) != 3:
        return False
    return all(c in 'rc' for c in layout.lower())


def validate_gpu_arch(arch: str) -> bool:
    """Validate GPU architecture string"""
    # Common AMD GPU architectures
    valid_archs = [
        'gfx900', 'gfx906', 'gfx908', 'gfx90a',
        'gfx940', 'gfx941', 'gfx942',
        'gfx1030', 'gfx1100', 'gfx1101',
    ]
    return arch.lower() in valid_archs


# ============================================================================
# Logging Utilities
# ============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        ensure_dir(log_file.parent)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class ProgressLogger:
    """Simple progress logger"""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.last_percent = -1
    
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        percent = int(100 * self.current / self.total)
        
        # Only log every 10%
        if percent >= self.last_percent + 10:
            log.info(f"{self.desc}: {percent}% ({self.current}/{self.total})")
            self.last_percent = percent
    
    def finish(self):
        """Mark as complete"""
        log.info(f"{self.desc}: 100% ({self.total}/{self.total}) - Complete!")


# ============================================================================
# Performance Utilities
# ============================================================================

class Timer:
    """Simple timer for performance measurement"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        log.info(f"{self.name} took {elapsed:.2f} seconds")
    
    def elapsed(self) -> float:
        """Get elapsed time"""
        import time
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0


def memoize_to_file(cache_file: Path):
    """
    Decorator to cache function results to file
    
    Usage:
        @memoize_to_file(Path("cache.json"))
        def expensive_function(arg):
            # ... expensive computation ...
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            import pickle
            key = generate_hash(pickle.dumps((args, kwargs)))
            
            # Try to load from cache
            if cache_file.exists():
                cache = read_json(cache_file)
                if key in cache:
                    log.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
            else:
                cache = {}
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Save to cache
            cache[key] = result
            write_json(cache, cache_file)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# System Utilities
# ============================================================================

def get_cpu_count() -> int:
    """Get number of CPU cores"""
    try:
        return os.cpu_count() or 1
    except:
        return 1


def get_available_memory() -> int:
    """Get available system memory in bytes"""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        # Fallback: assume 8GB
        return 8 * 1024 * 1024 * 1024


def check_command_available(command: str) -> bool:
    """Check if command is available in PATH"""
    import shutil
    return shutil.which(command) is not None


# ============================================================================
# Data Structure Utilities
# ============================================================================

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
    """Unflatten dictionary"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ============================================================================
# Version Utilities
# ============================================================================

def get_git_hash(length: int = 8) -> str:
    """Get current git commit hash"""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:length]
    except:
        pass
    return "unknown"


def get_git_branch() -> str:
    """Get current git branch"""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "unknown"


# ============================================================================
# Testing Utilities
# ============================================================================

def create_test_config(output_path: Path) -> Path:
    """Create minimal test configuration"""
    config = {
        "tile_config": {
            "tile_m": [128],
            "tile_n": [128],
            "tile_k": [32],
            "warp_m": [2],
            "warp_n": [2],
            "warp_k": [1],
            "warp_tile_m": [32],
            "warp_tile_n": [32],
            "warp_tile_k": [16],
        },
        "trait_config": {
            "pipeline": ["compv4"],
            "epilogue": ["cshuffle"],
            "scheduler": ["intrawave"],
            "pad_m": [False],
            "pad_n": [False],
            "pad_k": [False],
            "persistent": [False],
        }
    }
    
    write_json(config, output_path)
    return output_path


# ============================================================================
# CLI Utilities
# ============================================================================

def confirm_action(prompt: str, default: bool = False) -> bool:
    """Ask user for confirmation"""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes']


def print_table(headers: List[str], rows: List[List[Any]]):
    """Print formatted table"""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


# ============================================================================
# Module Info
# ============================================================================

def get_module_info() -> Dict[str, str]:
    """Get module information"""
    return {
        'project': 'composable_kernel',
        'module': 'dispatcher.codegen',
        'version': '2.0.0',
        'git_hash': get_git_hash(),
        'git_branch': get_git_branch(),
    }


if __name__ == '__main__':
    # Test utilities
    print("CK Tile GEMM Codegen Utilities")
    print("=" * 50)
    
    info = get_module_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\nProject root:", get_project_root())
    print("Library path:", get_library_path())
    print("Tile engine path:", get_tile_engine_path())
    print("CPU count:", get_cpu_count())
    print("Available memory:", f"{get_available_memory() / (1024**3):.1f} GB")
    print("grep available:", check_command_available('grep'))
    print("git available:", check_command_available('git'))

