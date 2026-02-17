# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Parse a single Clang -ftime-trace JSON file into a Pandas DataFrame.

This module provides fast parsing of Clang's -ftime-trace output using orjson
for performance. The JSON file is typically a single-line array of trace events.
"""

import orjson
import pandas as pd
from pathlib import Path
from typing import Union, Optional
from datetime import datetime
from dataclasses import dataclass


# Expected schema for trace event DataFrames with optimized dtypes
# This enforces strict column validation and memory-efficient types
# The memory usage is dominated by arg detail, but we optimize each series.
TRACE_EVENT_DTYPES = {
    "pid": "int32",  # Process ID (max observed: ~2.3M, fits in int32)
    "tid": "int32",  # Thread ID (max observed: ~2.3M, fits in int32)
    "ts": "int64",  # Timestamp in microseconds (requires int64 for epoch times)
    "cat": "category",  # Category (low cardinality, use categorical)
    "ph": "category",  # Phase type (very low cardinality: X, B, E, i, etc.)
    "id": "int64",  # Event ID
    "name": "category",  # Event name (medium cardinality, use categorical)
    "dur": "int64",  # Duration in microseconds (max 10 days = 864B μs, requires int64)
    "arg_detail": "string",  # Detail string (high cardinality, keep as string)
    "arg_count": "int64",  # Argument count
    "arg_avg ms": "int64",  # Average milliseconds
    "arg_name": "category",  # Argument name (medium cardinality, use categorical)
}


@dataclass
class FileMetadata:
    """
    Processed metadata with computed fields for compilation analysis.

    This extends the raw metadata with derived values like formatted timestamps
    and converted time units for convenience.

    Attributes:
        source_file: Main .cpp/.c file being compiled
        time_granularity: Time unit used in trace (always "microseconds" for Clang)
        beginning_of_time: Epoch timestamp in microseconds from JSON root
        execute_compiler_ts: Timestamp of ExecuteCompiler event (microseconds)
        execute_compiler_dur: Duration of ExecuteCompiler event (microseconds)
        total_wall_time_us: Total compilation time in microseconds (same as execute_compiler_dur)
        total_wall_time_s: Total compilation time in seconds (computed from microseconds)
        wall_start_time: Wall clock start time in microseconds since epoch (computed)
        wall_end_time: Wall clock end time in microseconds since epoch (computed)
        wall_start_datetime: Human-readable start time string (formatted)
        wall_end_datetime: Human-readable end time string (formatted)
    """

    source_file: Optional[str] = None
    time_granularity: str = "microseconds"
    beginning_of_time: Optional[int] = None
    execute_compiler_ts: Optional[int] = None
    execute_compiler_dur: Optional[int] = None
    total_wall_time_us: Optional[int] = None
    total_wall_time_s: Optional[float] = None
    wall_start_time: Optional[int] = None
    wall_end_time: Optional[int] = None
    wall_start_datetime: Optional[str] = None
    wall_end_datetime: Optional[str] = None

    def __repr__(self):
        # auto-generate pretty lines
        fields = "\n".join(
            f"  {name} = {value!r}" for name, value in self.__dict__.items()
        )
        return f"{self.__class__.__name__}(\n{fields}\n)"


def parse_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a Clang -ftime-trace JSON file into a Pandas DataFrame.

    The -ftime-trace format is a JSON array of trace events. Each event contains
    fields like name, phase (ph), timestamp (ts), duration (dur), process/thread IDs,
    and optional arguments (args).

    The beginningOfTime value from the JSON structure is automatically extracted
    and stored in df.attrs['beginningOfTime']. Use get_metadata(df) to get
    processed metadata with event-derived fields and computed values.

    Args:
        filepath: Path to the -ftime-trace JSON file

    Returns:
        DataFrame with columns for each event field. Nested 'args' are flattened
        with an 'arg_' prefix. The beginningOfTime value is stored in
        df.attrs['beginningOfTime'].

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid or empty

    Examples:
        >>> df = parse_file('build/trace.json')
        >>> df[['name', 'dur']].head()
        >>>
        >>> # Access processed metadata
        >>> metadata = get_metadata(df)
        >>> print(f"Compiled: {metadata.source_file}")
        >>> print(f"Duration: {metadata.total_wall_time_s:.2f}s")
        >>>
        >>> # Access beginningOfTime directly if needed
        >>> beginning = df.attrs.get('beginningOfTime')
        >>> print(f"Beginning of time: {beginning}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Trace file not found: {filepath}")

    # Read and parse JSON using orjson for speed
    with open(filepath, "rb") as f:
        data = orjson.loads(f.read())

    if not data:
        raise ValueError(f"Empty trace data in file: {filepath}")

    # Handle both formats: direct array or {"traceEvents": [...]}
    if isinstance(data, dict):
        if "traceEvents" in data:
            events = data["traceEvents"]
        else:
            raise ValueError(
                f"Expected 'traceEvents' key in JSON object, got keys: {list(data.keys())}"
            )
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError(f"Expected JSON array or object, got {type(data).__name__}")

    # Convert to DataFrame
    df = pd.DataFrame(events)

    if df.empty:
        raise ValueError(f"No trace events found in file: {filepath}")

    # Flatten 'args' column if it exists
    if "args" in df.columns:
        df = _flatten_args(df)

    # Validate schema: check for missing columns
    expected_columns = set(TRACE_EVENT_DTYPES.keys())
    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns
    if missing_columns:
        raise ValueError(
            f"Missing expected columns in trace data: {sorted(missing_columns)}"
        )

    # Validate schema: check for unexpected columns
    unexpected_columns = actual_columns - expected_columns
    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns found in trace data: {sorted(unexpected_columns)}"
        )

    # Apply optimized dtypes with strict type enforcement
    for col, dtype in TRACE_EVENT_DTYPES.items():
        if dtype in ("int64", "int32"):
            # Fill missing values with 0 for integer columns, then convert to specified int type
            df[col] = df[col].fillna(0).astype(dtype)
        elif dtype == "category":
            # Convert to categorical for memory efficiency with repeated values
            df[col] = df[col].astype("category")
        elif dtype == "string":
            # Convert to pandas string dtype for memory efficiency
            df[col] = df[col].astype("string")
        else:
            raise ValueError(f"Unsupported dtype '{dtype}' for column '{col}'")

    # Extract and store beginningOfTime in DataFrame attributes
    df.attrs["beginningOfTime"] = (
        data.get("beginningOfTime") if isinstance(data, dict) else None
    )

    # Store the source file path derived from the trace filename
    # The trace filename format is: <source_file>.json
    # Remove the .json extension to get the source file path
    source_file_path = filepath.stem  # Gets filename without .json extension
    full_path = filepath.parent / source_file_path
    df.attrs["sourceFile"] = _remove_cmake_artifacts(str(full_path))
    df.attrs["traceFilePath"] = str(filepath)

    return df


def _flatten_args(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the 'args' column into separate columns with 'arg_' prefix.

    The 'args' field in trace events contains additional metadata as a dictionary.
    This function extracts those key-value pairs into separate columns.

    Args:
        df: DataFrame with an 'args' column containing dictionaries

    Returns:
        DataFrame with flattened args columns and original 'args' column removed
    """
    args_list = df["args"].tolist()
    args_data = [arg if isinstance(arg, dict) else {} for arg in args_list]

    if args_data:
        args_df = pd.DataFrame(args_data)
        # Prefix all args columns with 'arg_'
        args_df.columns = [f"arg_{col}" for col in args_df.columns]

        # Drop original args column and concatenate flattened args
        df = df.drop(columns=["args"])
        df = pd.concat([df, args_df], axis=1)

    return df


def _remove_cmake_artifacts(file_path: str) -> str:
    """
    Remove CMake build artifacts from a file path.

    CMake creates build directories with the pattern:
    <build-dir>/<source-path>/CMakeFiles/<target>.dir/<source-file>

    This function removes the CMakeFiles and .dir segments to reconstruct
    the logical source file path while preserving the build directory prefix.

    Args:
        file_path: Path potentially containing CMake artifacts

    Returns:
        Path with CMakeFiles and .dir segments removed

    Examples:
        >>> _remove_cmake_artifacts('build/library/src/foo/CMakeFiles/target.dir/bar.cpp')
        'build/library/src/foo/bar.cpp'
        >>> _remove_cmake_artifacts('library/src/foo/bar.cpp')
        'library/src/foo/bar.cpp'
    """
    path = Path(file_path)
    parts = path.parts

    # Filter out CMakeFiles and any parts ending with .dir
    filtered_parts = [
        part for part in parts if part != "CMakeFiles" and not part.endswith(".dir")
    ]

    # Reconstruct the path
    if filtered_parts:
        return str(Path(*filtered_parts))
    return file_path


def _normalize_source_path(file_path: str) -> str:
    """
    Normalize a source file path to be relative to composable_kernel if present.

    If 'composable_kernel' appears in the path, returns the path starting from
    'composable_kernel/'. Otherwise, returns the original path unchanged.

    Args:
        file_path: Full filesystem path to a source file

    Returns:
        Normalized path starting from composable_kernel, or original path if
        composable_kernel is not found

    Examples:
        >>> _normalize_source_path('/home/user/composable_kernel/include/ck/tensor.hpp')
        'composable_kernel/include/ck/tensor.hpp'
        >>> _normalize_source_path('/usr/include/vector')
        '/usr/include/vector'
    """
    path = Path(file_path)
    parts = path.parts

    # Find the last occurrence of 'composable_kernel' in the path
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "composable_kernel":
            # Return path from composable_kernel onwards
            return str(Path(*parts[i:]))

    # If composable_kernel not found, return original path
    return file_path


def get_metadata(df: pd.DataFrame) -> FileMetadata:
    """
    Extract and process compilation metadata from a DataFrame.

    This function processes events from the DataFrame to extract compilation
    information, then computes derived fields like formatted timestamps and
    converted time units.

    Args:
        df: DataFrame returned by parse_file() with beginningOfTime in its .attrs

    Returns:
        FileMetadata instance with both raw and computed fields:
        - source_file: Main .cpp/.c file being compiled (from events)
        - time_granularity: Time unit used in trace ("microseconds")
        - beginning_of_time: Epoch timestamp in microseconds from JSON root
        - execute_compiler_ts: Timestamp of ExecuteCompiler event (from events)
        - execute_compiler_dur: Duration of ExecuteCompiler event (from events)
        - total_wall_time_us: Total compilation time in microseconds
        - total_wall_time_s: Total compilation time in seconds (computed)
        - wall_start_time: Wall clock start time (computed)
        - wall_end_time: Wall clock end time (computed)
        - wall_start_datetime: Human-readable start time (formatted)
        - wall_end_datetime: Human-readable end time (formatted)

    Examples:
        >>> df = parse_file('trace.json')
        >>> metadata = get_metadata(df)
        >>> print(f"Compiled: {metadata.source_file}")
        >>> print(f"Duration: {metadata.total_wall_time_s:.2f}s")
        >>> print(f"Started: {metadata.wall_start_datetime}")
    """
    # Extract beginningOfTime and source_file from DataFrame attributes
    beginning_of_time = None
    source_file = None
    if hasattr(df, "attrs"):
        beginning_of_time = df.attrs.get("beginningOfTime")
        source_file = df.attrs.get("source_file")

    # Initialize metadata with values from DataFrame attributes
    metadata = FileMetadata(
        beginning_of_time=beginning_of_time, source_file=source_file
    )

    # Process events to extract ExecuteCompiler timing information
    if "name" in df.columns:
        execute_compiler = df[df["name"] == "ExecuteCompiler"]
        if not execute_compiler.empty:
            # Get the first ExecuteCompiler event
            event = execute_compiler.iloc[0]
            if "ts" in event:
                metadata.execute_compiler_ts = event["ts"]
            if "dur" in event:
                metadata.execute_compiler_dur = event["dur"]

    # Fallback: Try to find source file from ParseDeclarationOrFunctionDefinition events
    # This is only used if source_file wasn't already set from the filename
    if (
        metadata.source_file is None
        and "name" in df.columns
        and "arg_detail" in df.columns
    ):
        # Look for ParseDeclarationOrFunctionDefinition events with .cpp or .c files
        source_extensions = (".cpp", ".cc", ".cxx", ".c")
        parse_events = df[df["name"] == "ParseDeclarationOrFunctionDefinition"]

        for _, event in parse_events.iterrows():
            detail = event.get("arg_detail", "")
            if detail:
                # Extract file path (may include line:column info)
                file_path = str(detail).split(":")[0]

                # Check if it's a source file (not a header)
                if any(file_path.endswith(ext) for ext in source_extensions):
                    metadata.source_file = _normalize_source_path(file_path)
                    break

    # Compute derived fields
    if metadata.execute_compiler_dur is not None:
        metadata.total_wall_time_us = metadata.execute_compiler_dur
        metadata.total_wall_time_s = metadata.execute_compiler_dur / 1_000_000.0

    # Calculate wall clock times if we have the necessary data
    if (
        metadata.beginning_of_time is not None
        and metadata.execute_compiler_ts is not None
        and metadata.execute_compiler_dur is not None
    ):
        metadata.wall_start_time = (
            metadata.beginning_of_time + metadata.execute_compiler_ts
        )
        metadata.wall_end_time = (
            metadata.wall_start_time + metadata.execute_compiler_dur
        )

        # Convert to human-readable datetime strings
        try:
            start_dt = datetime.fromtimestamp(metadata.wall_start_time / 1_000_000.0)
            end_dt = datetime.fromtimestamp(metadata.wall_end_time / 1_000_000.0)
            metadata.wall_start_datetime = start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[
                :-3
            ]
            metadata.wall_end_datetime = end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except (OSError, ValueError):
            # Handle invalid timestamps gracefully
            pass

    return metadata
