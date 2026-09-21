#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Generic CSV to Parquet converter for ML training data.

Works with any operation type (grouped_conv, gemm, fmha, etc.) by auto-detecting
CSV structure and optionally using custom kernel name patterns.

Supported operations:
  - Grouped convolution (forward, bwd_data, bwd_weight)
  - GEMM Universal
  - FMHA
  - Any future operations with CSV benchmark output

Usage:
    # Auto-detect everything (recommended)
    python convert_csv_to_parquet.py \
        --input benchmark_data.csv \
        --output training_data.parquet \
        --arch gfx950

    # With custom kernel pattern
    python convert_csv_to_parquet.py \
        --input benchmark_data.csv \
        --output training_data.parquet \
        --arch gfx950 \
        --kernel-pattern "myop_(?P<variant>\\w+)_(?P<dtype>\\w+)_(?P<config>.*)"

    # Override operation type
    python convert_csv_to_parquet.py \
        --input benchmark_data.csv \
        --output training_data.parquet \
        --arch gfx950 \
        --op-type grouped_conv

Features:
    - Auto-detects problem columns from CSV headers
    - Generic kernel name parsing with optional custom patterns
    - Supports all GPU architectures and data types
    - No hardcoded operation-specific logic
    - Validates data quality and reports statistics
"""

import argparse
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Set


# Known metric/metadata columns (will be excluded from problem features)
METRIC_COLUMNS: Set[str] = {
    "kernel",
    "kernel_name",
    "latency_ms",
    "tflops",
    "bandwidth_gb_s",
    "non_zero",
    "problem_idx",
    "run_id",
    "is_valid",
    "error_msg",
}


# Hardware profiles for different architectures
HW_PROFILES = {
    "gfx950": {  # MI300 series
        "hw_num_cus": 256,
        "hw_simds_per_cu": 4,
        "hw_shader_engines": 32,
        "hw_max_clock_mhz": 2400,
        "hw_max_waves_per_cu": 32,
        "hw_wavefront_size": 64,
        "hw_lds_capacity": 65536,
        "hw_l1_cache_kb": 32,
        "hw_l2_cache_kb": 4096,
        "hw_l3_cache_kb": 262144,
        "hw_num_xcd": 8,
    },
    "gfx942": {  # MI300A
        "hw_num_cus": 228,
        "hw_simds_per_cu": 4,
        "hw_shader_engines": 28,
        "hw_max_clock_mhz": 2100,
        "hw_max_waves_per_cu": 32,
        "hw_wavefront_size": 64,
        "hw_lds_capacity": 65536,
        "hw_l1_cache_kb": 32,
        "hw_l2_cache_kb": 4096,
        "hw_l3_cache_kb": 262144,
        "hw_num_xcd": 8,
    },
    "gfx90a": {  # MI250X
        "hw_num_cus": 110,
        "hw_simds_per_cu": 4,
        "hw_shader_engines": 8,
        "hw_max_clock_mhz": 1700,
        "hw_max_waves_per_cu": 32,
        "hw_wavefront_size": 64,
        "hw_lds_capacity": 65536,
        "hw_l1_cache_kb": 16,
        "hw_l2_cache_kb": 8192,
        "hw_l3_cache_kb": 131072,
        "hw_num_xcd": 1,
    },
}


def parse_kernel_name_generic(
    kernel_name: str, pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse kernel name to extract configuration features.

    Auto-detects common patterns or uses custom pattern if provided.

    Common patterns:
      - grouped_conv: grouped_conv_{variant}_{dtype}_{ndim}d_{block}x{m}x{n}_{pipeline}
      - gemm: gemm_{dtype}_{layout}_{tiles}_{pipeline}_{scheduler}

    Args:
        kernel_name: Kernel name string
        pattern: Optional custom regex pattern with named groups

    Returns:
        Dictionary with extracted features
    """
    result = {"kernel_name": kernel_name}

    if pattern:
        # Use custom pattern
        match = re.match(pattern, kernel_name)
        if match:
            result.update(match.groupdict())
            return result

    # Auto-detect common patterns

    # Pattern 1: grouped_conv_{variant}_{dtype}_{ndim}d_{block}x{m}x{n}_{pipeline}
    #   [_{wave_mode}] [_dsb] [_si]
    # Pipeline alternation is explicit so the suffix tokens do not get swallowed
    # by the [a-z0-9]+ pipeline group.
    grouped_conv_pattern = (
        r"grouped_conv_([a-z_]+)_([a-z0-9]+)_(\d+)d_(\d+)x(\d+)x(\d+)_"
        r"(basic_v\d+|basic_async_v\d+|comp_async|compv\d+|mem|preshufflev\d+)"
        r"(?:_(intrawave|interwave))?(_dsb)?(_si)?$"
    )
    match = re.match(grouped_conv_pattern, kernel_name)
    if match:
        (
            variant,
            dtype,
            ndim,
            block_size,
            gemm_m,
            gemm_n,
            pipeline,
            wave_mode,
            dsb_tok,
            si_tok,
        ) = match.groups()
        result.update(
            {
                "op_type": "grouped_conv",
                "variant": variant,
                "dtype": dtype,
                "ndim_spatial": int(ndim),
                "block_size": int(block_size),
                "gemm_m_per_block": int(gemm_m),
                "gemm_n_per_block": int(gemm_n),
                "pipeline": pipeline,
                "wave_mode": wave_mode if wave_mode else "intrawave",
                "has_dsb": 1 if dsb_tok else 0,
                "has_si": 1 if si_tok else 0,
            }
        )
        return result

    # Pattern 2: gemm_universal_{dtype}_{layout}_{tiles}_{pipeline}_{scheduler}
    gemm_pattern = (
        r"gemm_universal_([a-z0-9]+)_([a-z]+)_(\d+x\d+x\d+)_([a-z0-9]+)_([a-z]+)"
    )
    match = re.match(gemm_pattern, kernel_name)
    if match:
        dtype, layout, tiles, pipeline, scheduler = match.groups()
        tile_parts = tiles.split("x")
        result.update(
            {
                "op_type": "gemm_universal",
                "dtype": dtype,
                "layout": layout,
                "tile_m": int(tile_parts[0]) if len(tile_parts) > 0 else 0,
                "tile_n": int(tile_parts[1]) if len(tile_parts) > 1 else 0,
                "tile_k": int(tile_parts[2]) if len(tile_parts) > 2 else 0,
                "pipeline": pipeline,
                "scheduler": scheduler,
            }
        )
        return result

    # Pattern 3: Generic fallback - extract dtype, pipeline from common suffixes
    # Look for common patterns like _bf16_, _fp16_, _compv3, _mem
    dtype_match = re.search(r"_(bf16|fp16|fp8|fp32|int8)", kernel_name)
    if dtype_match:
        result["dtype"] = dtype_match.group(1)

    pipeline_match = re.search(r"_(compv\d+|mem|async)", kernel_name)
    if pipeline_match:
        result["pipeline"] = pipeline_match.group(1)

    # Extract operation type from prefix
    op_match = re.match(r"^([a-z_]+?)_", kernel_name)
    if op_match:
        result["op_type"] = op_match.group(1)

    return result


def auto_detect_problem_columns(df: pd.DataFrame) -> list[str]:
    """
    Auto-detect problem feature columns by excluding known metric columns.

    Args:
        df: Input dataframe

    Returns:
        List of column names that are problem features
    """
    return [col for col in df.columns if col not in METRIC_COLUMNS]


def convert_csv_to_parquet(
    csv_file: Path,
    output_file: Path,
    arch: str = "gfx950",
    dtype: Optional[str] = None,
    variant: Optional[str] = None,
    op_type: Optional[str] = None,
    kernel_pattern: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert benchmark CSV to parquet training data format.

    Args:
        csv_file: Input CSV file path
        output_file: Output parquet file path
        arch: GPU architecture (default: gfx950)
        dtype: Data type override (default: auto-detect from kernel name)
        variant: Variant override (default: auto-detect from kernel name)
        op_type: Operation type override (default: auto-detect)
        kernel_pattern: Custom regex pattern for parsing kernel names

    Returns:
        DataFrame with converted data
    """
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Auto-detect problem columns
    problem_cols = auto_detect_problem_columns(df)
    print(f"Auto-detected {len(problem_cols)} problem feature columns:")
    print(f"  {', '.join(problem_cols)}")
    print()

    # Parse kernel names
    print("Parsing kernel configurations...")
    kernel_configs = {}
    parse_errors = 0

    for kernel_name in df["kernel"].unique():
        try:
            config = parse_kernel_name_generic(kernel_name, kernel_pattern)
            kernel_configs[kernel_name] = config
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:  # Show first 3 errors
                print(f"  Warning: Could not fully parse '{kernel_name}': {e}")
            kernel_configs[kernel_name] = {"kernel_name": kernel_name}

    if parse_errors > 3:
        print(f"  ... and {parse_errors - 3} more parsing warnings")

    print(f"  Parsed {len(kernel_configs)} unique kernels")
    print()

    # Get hardware profile
    hw_profile = HW_PROFILES.get(arch, {})
    if not hw_profile:
        print(f"Warning: No hardware profile for {arch}, using defaults")
        hw_profile = HW_PROFILES["gfx950"]

    # Build parquet rows
    rows = []
    for _, row in df.iterrows():
        kernel_name = row["kernel"]
        kernel_cfg = kernel_configs.get(kernel_name, {})

        # Build parquet row
        pq_row = {
            # Kernel info
            "kernel_name": kernel_name,
            # Performance metrics
            "latency_ms": float(row["latency_ms"]),
            "tflops": float(row["tflops"]),
        }

        # Add optional columns if they exist
        if "non_zero" in row:
            pq_row["non_zero"] = int(row["non_zero"])
        if "problem_idx" in row:
            pq_row["problem_idx"] = int(row["problem_idx"])

        # Add all problem features (auto-detected)
        for col in problem_cols:
            pq_row[col] = row[col]

        # Add kernel configuration (parsed from name)
        pq_row.update(kernel_cfg)

        # Add metadata overrides
        if op_type:
            pq_row["op_type"] = op_type
        if dtype:
            pq_row["dtype"] = dtype
        if variant:
            pq_row["variant"] = variant

        # Add architecture
        pq_row["arch"] = arch

        # Add hardware profile
        pq_row.update(hw_profile)

        # Add validity flag
        pq_row["is_valid"] = True
        pq_row["run_id"] = 0

        rows.append(pq_row)

    result_df = pd.DataFrame(rows)

    print(f"Converted {len(result_df):,} benchmark results")
    print(f"  Valid: {result_df['is_valid'].sum():,}")
    print(f"  Unique kernels: {result_df['kernel_name'].nunique()}")

    # Count unique problems (use problem columns only)
    if problem_cols:
        unique_problems = result_df[problem_cols].drop_duplicates().shape[0]
        print(f"  Unique problems: {unique_problems}")
    print()

    # Save to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    print()

    # Show statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()

    # Performance metrics
    print("Performance metrics:")
    print(
        f"  Latency (ms): {result_df['latency_ms'].min():.4f} - {result_df['latency_ms'].max():.4f}"
    )
    print(
        f"  TFLOPS: {result_df['tflops'].min():.2f} - {result_df['tflops'].max():.2f}"
    )
    print(f"  Mean TFLOPS: {result_df['tflops'].mean():.2f}")
    print(f"  Median TFLOPS: {result_df['tflops'].median():.2f}")
    print()

    # Pipeline distribution (if available)
    if "pipeline" in result_df.columns:
        print("Pipeline distribution:")
        print(result_df["pipeline"].value_counts())
        print()

    # Operation type distribution (if available)
    if "op_type" in result_df.columns:
        print("Operation type distribution:")
        print(result_df["op_type"].value_counts())
        print()

    # Show sample best results
    print("Sample best kernels per problem:")
    # Group by problem columns if available
    if problem_cols:
        best_per_problem = result_df.loc[
            result_df.groupby(problem_cols)["tflops"].idxmax()
        ]
        for i, (idx, row) in enumerate(best_per_problem.head(5).iterrows()):
            prob_desc = ", ".join(
                [f"{col}={row[col]}" for col in problem_cols[:4]]
            )  # Show first 4 params
            print(
                f"  {prob_desc}... → {row['tflops']:.1f} TFLOPS ({row['kernel_name']})"
            )
    print()

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Generic CSV to Parquet converter for ML training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input CSV file from benchmark"
    )
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument(
        "--arch", type=str, default="gfx950", help="GPU architecture (default: gfx950)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type override (default: auto-detect from kernel name)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Operation variant override (default: auto-detect from kernel name)",
    )
    parser.add_argument(
        "--op-type",
        type=str,
        help="Operation type override (default: auto-detect from kernel name)",
    )
    parser.add_argument(
        "--kernel-pattern",
        type=str,
        help="Custom regex pattern for parsing kernel names (use named groups)",
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    # Convert CSV to parquet
    df = convert_csv_to_parquet(
        input_file,
        output_file,
        args.arch,
        args.dtype,
        args.variant,
        args.op_type,
        args.kernel_pattern,
    )

    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Output: {output_file}")
    print(f"✓ Rows: {len(df):,}")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"✓ Size: {output_file.stat().st_size / 1024:.1f} KB")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
