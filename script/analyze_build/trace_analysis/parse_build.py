# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Utility functions for trace analysis.

Helper functions for file discovery, path handling, and other common operations.
"""

import subprocess
import pandas as pd
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
            ["find", str(trace_dir), "-name", "*.cpp.json", "-type", "f"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        json_files = [Path(p) for p in result.stdout.strip().split("\n") if p]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # Fallback to Python (cross-platform)
        print("Using Python to find trace files (this may be slower)...")
        json_files = list(trace_dir.rglob("*.cpp.json"))

    return json_files


def read_trace_files(json_files: List[Path], workers: int = -1) -> List["pd.DataFrame"]:
    """
    Parse trace files in parallel and return list of DataFrames.

    This is a convenience function that uses the Pipeline API to parse
    multiple trace files in parallel with progress tracking.

    Args:
        json_files: List of paths to trace JSON files
        workers: Number of parallel workers to use:
            - -1: Use all available CPUs (default)
            - None: Sequential processing (single-threaded)
            - N > 0: Use N worker processes

    Returns:
        List of parsed DataFrames, one per file

    Example:
        >>> from pathlib import Path
        >>> from trace_analysis import find_trace_files, read_trace_files
        >>>
        >>> # Find and parse all trace files
        >>> trace_files = find_trace_files(Path("build/CMakeFiles"))
        >>> dataframes = read_trace_files(trace_files, workers=8)
        >>> print(f"Parsed {len(dataframes)} files")
        >>>
        >>> # Use Pipeline directly for more control
        >>> from trace_analysis import Pipeline
        >>> from trace_analysis.parse_file import parse_file
        >>>
        >>> pipeline = Pipeline(trace_files).map(parse_file, workers=8)
        >>> all_events, metadata = pipeline.tee(
        ...     lambda dfs: pd.concat(dfs, ignore_index=True),
        ...     lambda dfs: [get_metadata(df) for df in dfs]
        ... )
    """
    from trace_analysis.pipeline import Pipeline
    from trace_analysis.parse_file import parse_file

    return (
        Pipeline(json_files)
        .map(parse_file, workers=workers, desc="Parsing trace files")
        .collect()
    )
