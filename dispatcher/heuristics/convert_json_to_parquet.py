#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Convert benchmark JSON results to parquet format for training.

Usage:
    python convert_json_to_parquet.py \
        --input benchmark_results_fp16_rcr.json \
        --output fp16_training_data.parquet

Features:
    - Converts JSON benchmark results to flat row format
    - Automatically fixes pad flags for _mem kernels
    - Captures both successes and failures
    - Compatible with existing training data format
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def convert_json_to_parquet(json_file: Path, output_file: Path, arch: str = "gfx950"):
    """Convert benchmark JSON to parquet training data format."""

    print(f"Loading {json_file}...")
    with open(json_file) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    dtype = metadata.get("dtype", "fp16")
    layout = metadata.get("layout", "rcr")

    print(f"  Data type: {dtype}")
    print(f"  Layout: {layout}")
    print(f"  Kernels: {metadata.get('num_kernels', 0)}")
    print(f"  Problem sizes: {metadata.get('num_problems', 0)}")
    print()

    rows = []
    for kernel_result in data["results"]:
        kernel_config = kernel_result["kernel_config"]

        for benchmark in kernel_result["benchmarks"]:
            # Common fields for both valid and invalid runs
            row = {
                "op_type": "gemm_universal",
                "dtype": dtype,
                "layout": layout,
                "arch": arch,
                "kernel_name": kernel_config["name"],
                "m": benchmark["m"],
                "n": benchmark["n"],
                "k": benchmark["k"],
                "split_k": 1,
                "is_valid": benchmark["is_valid"],
                "run_id": 0,
                "pipeline": kernel_config["pipeline"],
                "epilogue": kernel_config["epilogue"],
                "scheduler": kernel_config["scheduler"],
                "pad_m": kernel_config["pad_m"],
                "pad_n": kernel_config["pad_n"],
                "pad_k": kernel_config["pad_k"],
                "persistent": kernel_config["persistent"],
                "tile_m": kernel_config["tile_m"],
                "tile_n": kernel_config["tile_n"],
                "tile_k": kernel_config["tile_k"],
                "warp_m": kernel_config["warp_m"],
                "warp_n": kernel_config["warp_n"],
                "warp_k": kernel_config["warp_k"],
                "warp_tile_m": kernel_config["warp_tile_m"],
                "warp_tile_n": kernel_config["warp_tile_n"],
                "warp_tile_k": kernel_config["warp_tile_k"],
            }

            if benchmark["is_valid"]:
                # Valid run - include performance metrics
                row["measured_tflops"] = benchmark["tflops"]
                row["latency_ms"] = benchmark["avg_time_ms"]
                # Calculate bandwidth if needed
                m, n, k = benchmark["m"], benchmark["n"], benchmark["k"]
                bytes_transferred = (m * k + k * n + m * n) * 2  # FP16 = 2 bytes
                if benchmark["avg_time_ms"] > 0:
                    row["bandwidth_gb_s"] = (bytes_transferred / 1e9) / (
                        benchmark["avg_time_ms"] / 1000
                    )
                else:
                    row["bandwidth_gb_s"] = 0.0
            else:
                # Failed run - zero metrics
                row["measured_tflops"] = 0.0
                row["latency_ms"] = 0.0
                row["bandwidth_gb_s"] = 0.0

            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"Converted {len(df):,} benchmark results")
    print(f"  Valid: {df['is_valid'].sum():,}")
    print(f"  Failed: {(~df['is_valid']).sum():,}")
    print()

    # Fix pad flags for _mem kernels (critical for P1 features!)
    print("Fixing pad flags for _mem kernels...")
    mem_mask = df["pipeline"] == "mem"
    mem_count = mem_mask.sum()

    if mem_count > 0:
        df.loc[mem_mask, "pad_m"] = True
        df.loc[mem_mask, "pad_n"] = True
        df.loc[mem_mask, "pad_k"] = True
        print(f"  ✓ Fixed {mem_count:,} _mem kernel rows")
        print()

    # Save to parquet
    df.to_parquet(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    print()

    # Show statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()

    print("Dimension ranges:")
    print(f"  M: {df['m'].min():,} - {df['m'].max():,}")
    print(f"  N: {df['n'].min():,} - {df['n'].max():,}")
    print(f"  K: {df['k'].min():,} - {df['k'].max():,}")
    print()

    print("Pipeline distribution:")
    print(df["pipeline"].value_counts())
    print()

    print("Pad flag distribution:")
    pad_combos = df[["pad_m", "pad_n", "pad_k"]].value_counts()
    print(pad_combos)
    print()

    if (~df["is_valid"]).sum() > 0:
        print("Failure analysis:")
        failed = df[~df["is_valid"]]
        print(f"  Total failures: {len(failed):,}")

        # Group by pipeline
        print("\n  By pipeline:")
        for pipeline, count in failed["pipeline"].value_counts().items():
            print(f"    {pipeline}: {count:,}")

        # Show sample failures
        print("\n  Sample failures:")
        for _, row in failed.head(5).iterrows():
            print(
                f"    {row['kernel_name'][:60]:60s} M={row['m']:4d} N={row['n']:4d} K={row['k']:4d}"
            )

    return df


def merge_datasets(parquet_files: list[Path], output_file: Path):
    """Merge multiple parquet files into one."""

    print("=" * 80)
    print("MERGING DATASETS")
    print("=" * 80)
    print()

    dfs = []
    for pq_file in parquet_files:
        if pq_file.exists():
            df = pd.read_parquet(pq_file)
            print(f"  {pq_file.name}: {len(df):,} rows")
            dfs.append(df)
        else:
            print(f"  ✗ {pq_file} not found, skipping")

    if not dfs:
        print("No files to merge!")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(output_file, index=False)

    print()
    print(f"✓ Merged {len(combined):,} total rows to {output_file}")
    print()

    # Show dtype distribution
    print("Data type distribution:")
    print(combined["dtype"].value_counts())
    print()

    print("Layout distribution:")
    print(combined["layout"].value_counts())


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark JSON to parquet training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSON file from benchmark"
    )
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument("--arch", type=str, default="gfx950", help="GPU architecture")
    parser.add_argument(
        "--merge_with", type=str, nargs="*", help="Additional parquet files to merge"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    # Convert JSON to parquet
    df = convert_json_to_parquet(input_file, output_file, args.arch)

    # Merge if requested
    if args.merge_with:
        merge_files = [output_file] + [Path(f) for f in args.merge_with]
        merged_output = output_file.parent / f"{output_file.stem}_merged.parquet"
        merge_datasets(merge_files, merged_output)


if __name__ == "__main__":
    main()
