#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Data pipeline for CK Tile heuristics.

Parses benchmark logs and structured JSON into a canonical parquet dataset.
Supports:
  - Streaming log format (Shape N: headers + inline JSON) from ck_tile profiling runs
  - Structured JSON from generate_benchmark_data.py
  - Direct parquet passthrough
"""

import json
import re
import subprocess
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd


CANONICAL_COLUMNS = [
    "op_type",
    "dtype",
    "layout",
    "arch",
    "kernel_name",
    "m",
    "n",
    "k",
    "split_k",
    "measured_tflops",
    "latency_ms",
    "bandwidth_gb_s",
    "is_valid",
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "pipeline",
    "scheduler",
    "epilogue",
    "pad_m",
    "pad_n",
    "pad_k",
    "persistent",
    "run_id",
]


def parse_kernel_name(name: str) -> dict:
    """Extract kernel config fields from a gemm_universal kernel name.

    Name format:
      gemm_universal_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}
      _{padM}_{padN}_{padK}_{persistent}_{tileM}x{tileN}x{tileK}
      _{warpM}x{warpN}x{warpK}_{warpTileM}x{warpTileN}x{warpTileK}
    """
    result = {}
    try:
        prefix_match = re.match(
            r"gemm_universal_(\w+?)_((?:rcr|rrr|crr|ccr))_(.*)", name
        )
        if not prefix_match:
            return result
        result["dtype"] = prefix_match.group(1)
        result["layout"] = prefix_match.group(2)
        remainder = prefix_match.group(3)

        parts = remainder.split("_")
        if len(parts) < 10:
            return result

        result["pipeline"] = parts[0]
        result["epilogue"] = parts[1]
        result["scheduler"] = parts[2]
        result["pad_m"] = parts[3] == "True"
        result["pad_n"] = parts[4] == "True"
        result["pad_k"] = parts[5] == "True"
        result["persistent"] = parts[6] == "True"

        tile_dims = parts[7].split("x")
        warp_dims = parts[8].split("x")
        warp_tile_dims = parts[9].split("x")

        result["tile_m"] = int(tile_dims[0])
        result["tile_n"] = int(tile_dims[1])
        result["tile_k"] = int(tile_dims[2])
        result["warp_m"] = int(warp_dims[0])
        result["warp_n"] = int(warp_dims[1])
        result["warp_k"] = int(warp_dims[2])
        result["warp_tile_m"] = int(warp_tile_dims[0])
        result["warp_tile_n"] = int(warp_tile_dims[1])
        result["warp_tile_k"] = int(warp_tile_dims[2])
    except (IndexError, ValueError):
        pass
    return result


def _layout_from_problem(problem: dict) -> str:
    """Derive layout shorthand (rcr/rrr/etc.) from problem JSON fields."""
    la = problem.get("layout_a", "")
    lb = problem.get("layout_b", "")
    lc = problem.get("layout_c", "")

    def _tag(s):
        s = s.lower()
        if "row" in s:
            return "r"
        if "col" in s:
            return "c"
        return "?"

    return _tag(la) + _tag(lb) + _tag(lc)


def parse_streaming_log(
    path: str | Path,
    arch: str = "unknown",
    run_id: Optional[str] = None,
    op_type: str = "gemm_universal",
) -> pd.DataFrame:
    """Parse a CK Tile streaming benchmark log into a canonical DataFrame.

    The log alternates between shape headers and JSON result blocks:
        Shape N: M=16 N=1536 K=7168 dtype=fp8 layout=rcr
        {
          "name": "gemm_universal_...",
          "problem": { ... },
          "perf_result": { "latency(ms)": ..., "tflops(TFlops)": ..., "bandwidth(GB/s)": ... }
        }
    """
    path = Path(path)
    if run_id is None:
        run_id = hashlib.md5(path.name.encode()).hexdigest()[:12]

    shape_re = re.compile(
        r"Shape\s+\d+:\s+M=(\d+)\s+N=(\d+)\s+K=(\d+)\s+dtype=(\w+)\s+layout=(\w+)"
    )

    rows = []
    current_m, current_n, current_k = 0, 0, 0
    current_dtype, current_layout = "", ""
    json_buf = []
    brace_depth = 0

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()

            shape_match = shape_re.search(stripped)
            if shape_match:
                current_m = int(shape_match.group(1))
                current_n = int(shape_match.group(2))
                current_k = int(shape_match.group(3))
                current_dtype = shape_match.group(4)
                current_layout = shape_match.group(5)
                continue

            if brace_depth == 0 and stripped.startswith("{"):
                json_buf = [stripped]
                brace_depth = stripped.count("{") - stripped.count("}")
                if brace_depth == 0:
                    raw = "\n".join(json_buf)
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            elif brace_depth > 0:
                json_buf.append(stripped)
                brace_depth += stripped.count("{") - stripped.count("}")
                if brace_depth <= 0:
                    brace_depth = 0
                    raw = "\n".join(json_buf)
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            else:
                continue

            # If we get here, obj was successfully parsed
            kernel_name = obj.get("name", "")
            problem = obj.get("problem", {})
            perf = obj.get("perf_result", {})

            m = problem.get("m", current_m)
            n = problem.get("n", current_n)
            k = problem.get("k", current_k)
            split_k = problem.get("split_k", 1)
            dtype = problem.get("dtype_a", current_dtype)
            layout = (
                _layout_from_problem(problem)
                if problem.get("layout_a")
                else current_layout
            )

            tflops = perf.get("tflops(TFlops)", 0.0)
            latency = perf.get("latency(ms)", 0.0)
            bandwidth = perf.get("bandwidth(GB/s)", 0.0)

            kp = parse_kernel_name(kernel_name)

            row = {
                "op_type": op_type,
                "dtype": dtype,
                "layout": layout,
                "arch": arch,
                "kernel_name": kernel_name,
                "m": m,
                "n": n,
                "k": k,
                "split_k": split_k,
                "measured_tflops": tflops,
                "latency_ms": latency,
                "bandwidth_gb_s": bandwidth,
                "is_valid": tflops > 0 and latency > 0,
                "run_id": run_id,
            }
            row.update(kp)
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def get_hardware_profile() -> dict:
    """Capture GPU hardware profile from rocminfo."""
    profile = {}
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        )
        output = result.stdout

        gpu_section = False
        for line in output.split("\n"):
            line = line.strip()
            if "Device Type:" in line and "GPU" in line:
                gpu_section = True
                continue
            if gpu_section and "Device Type:" in line and "GPU" not in line:
                break
            if not gpu_section:
                continue

            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()

            mapping = {
                "Name": "gfx_name",
                "Marketing Name": "marketing_name",
                "Compute Unit": "num_cus",
                "SIMDs per CU": "simds_per_cu",
                "Shader Engines": "shader_engines",
                "Shader Arrs. per Eng.": "shader_arrays_per_engine",
                "Max Clock Freq. (MHz)": "max_clock_mhz",
                "Wavefront Size": "wavefront_size",
                "Max Waves Per CU": "max_waves_per_cu",
                "Chip ID": "chip_id",
            }

            if key in mapping:
                raw = val.split("(")[0].strip()
                try:
                    profile[mapping[key]] = int(raw)
                except ValueError:
                    profile[mapping[key]] = raw

        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("L1:") and "num_cus" in profile:
                raw = line.split(":")[1].strip().split("(")[0].strip()
                try:
                    profile["l1_cache_kb"] = int(raw)
                except ValueError:
                    pass
            elif line.startswith("L2:"):
                raw = line.split(":")[1].strip().split("(")[0].strip()
                try:
                    profile["l2_cache_kb"] = int(raw)
                except ValueError:
                    pass
            elif line.startswith("L3:"):
                raw = line.split(":")[1].strip().split("(")[0].strip()
                try:
                    profile["l3_cache_kb"] = int(raw)
                except ValueError:
                    pass

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return profile


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a canonical parquet dataset."""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path):
    """Save a DataFrame in canonical parquet format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def build_training_dataset(
    data_dir: str | Path,
    op_type: str = "gemm_universal",
    dtype: str = "fp8",
) -> pd.DataFrame:
    """Load and merge all parquet files matching the given op/dtype from a directory."""
    data_dir = Path(data_dir)
    frames = []
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "op_type" in df.columns:
            df = df[df["op_type"] == op_type]
        if "dtype" in df.columns:
            df = df[df["dtype"] == dtype]
        if len(df) > 0:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No parquet files with op_type={op_type}, dtype={dtype} in {data_dir}"
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Parse CK Tile benchmark data")
    parser.add_argument("input", help="Input file (log or parquet)")
    parser.add_argument("--output", "-o", required=True, help="Output parquet path")
    parser.add_argument("--arch", default="gfx950", help="GPU architecture")
    parser.add_argument("--op_type", default="gemm_universal", help="Operation type")
    parser.add_argument(
        "--capture_hw",
        action="store_true",
        help="Capture hardware profile from rocminfo",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    print(f"Parsing {input_path}...")
    t0 = time.time()

    if input_path.suffix == ".parquet":
        df = load_parquet(input_path)
    else:
        df = parse_streaming_log(input_path, arch=args.arch, op_type=args.op_type)

    elapsed = time.time() - t0
    print(f"Parsed {len(df)} rows in {elapsed:.1f}s")
    print(f"  Unique shapes: {df.groupby(['m', 'n', 'k']).ngroups}")
    print(f"  Unique kernels: {df['kernel_name'].nunique()}")
    print(f"  Valid rows: {df['is_valid'].sum()} / {len(df)}")

    if df["measured_tflops"].max() > 0:
        print(
            f"  TFLOPS range: {df['measured_tflops'].min():.2f} - {df['measured_tflops'].max():.2f}"
        )

    if args.capture_hw:
        hw = get_hardware_profile()
        print(f"  Hardware profile: {hw}")
        for k, v in hw.items():
            df[f"hw_{k}"] = v

    save_parquet(df, args.output)
    print(f"Saved to {args.output}")
