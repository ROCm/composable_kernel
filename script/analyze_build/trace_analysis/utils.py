# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Utility functions for trace analysis.

Helper functions for file discovery, path handling, and other common operations.
"""

import subprocess
from pathlib import Path
from typing import List


def find_trace_files(trace_dir: Path) -> List[Path]:
    """
    Find all JSON trace files in a directory.

    Uses Unix 'find' command when available (2-5x faster than Python),
    with automatic fallback to Python's rglob for cross-platform compatibility.

    Args:
        trace_dir: Directory to search for trace files

    Returns:
        List of Path objects pointing to .json files

    Example:
        >>> from pathlib import Path
        >>> from trace_analysis import find_trace_files
        >>> trace_files = find_trace_files(Path("build/CMakeFiles"))
        >>> print(f"Found {len(trace_files)} trace files")
    """
    try:
        # Try Unix find (2-5x faster than Python)
        result = subprocess.run(
            ["find", str(trace_dir), "-name", "*.json", "-type", "f"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        sample_files = [Path(p) for p in result.stdout.strip().split("\n") if p]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # Fallback to Python (cross-platform)
        print("Using Python to find trace files (this may be slower)...")
        sample_files = list(trace_dir.rglob("*.json"))

    return sample_files
