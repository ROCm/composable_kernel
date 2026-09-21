#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Full FMHA benchmark sweep, organized by variant and dtype.

Compiles all kernels per variant (shared build dir for caching),
benchmarks against all smoke shapes, then splits results into:

    <output_dir>/
      fwd/fp16/results.csv
      fwd/bf16/results.csv
      splitkv/fp16/results.csv
      ...
      bwd_dot_do_o/fp16/results.csv
      bwd_dq_dk_dv/fp16/results.csv
      bwd_convert_dq/fp16/results.csv

Usage:
    python run_full_sweep.py --workers 256
    python run_full_sweep.py --workers 256 --category full --output /tmp/fmha_sweep
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

VARIANTS = ["fwd", "splitkv", "pagedkv", "appendkv", "batch_prefill", "bwd"]

BWD_FAMILIES = ["bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"]


def run_variant(variant, category, workers, build_dir, raw_csv, shape_timeout=600):
    """Run fmha_full_benchmark.py for one variant, return path to raw CSV."""
    cmd = [
        sys.executable,
        str(_THIS_DIR / "fmha_full_benchmark.py"),
        "--category",
        category,
        "--variant",
        variant,
        "--workers",
        str(workers),
        "--build-dir",
        str(build_dir),
        "--csv",
        str(raw_csv),
        "--json",
        str(raw_csv.with_suffix(".json")),
        "--shape-timeout",
        str(shape_timeout),
    ]
    print(f"\n{'=' * 80}")
    print(f"  Variant: {variant}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 80}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def split_csv(raw_csv, output_dir):
    """Split a raw CSV into per-family per-dtype subdirectories."""
    if not raw_csv.exists():
        return {}

    counts = defaultdict(int)
    writers = {}
    files = {}

    with open(raw_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            family = row.get("family", "unknown")
            dtype = row.get("dtype", "unknown")
            key = (family, dtype)

            if key not in writers:
                d = output_dir / family / dtype
                d.mkdir(parents=True, exist_ok=True)
                fh = open(d / "results.csv", "w", newline="")
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                writers[key] = w
                files[key] = fh

            writers[key].writerow(row)
            counts[key] += 1

    for fh in files.values():
        fh.close()

    return counts


def main():
    p = argparse.ArgumentParser(
        description="Full FMHA Sweep (organized by variant/dtype)"
    )
    p.add_argument("--workers", type=int, default=256)
    p.add_argument("--category", default="smoke", choices=["smoke", "full", "nightly"])
    p.add_argument("--output", default="/tmp/fmha_sweep")
    p.add_argument("--build-dir", default="/tmp/fmha_sweep_build")
    p.add_argument(
        "--variants",
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
        help="Which variants to run",
    )
    p.add_argument(
        "--shape-timeout", type=int, default=600, help="Per-shape timeout in seconds"
    )
    args = p.parse_args()

    output_dir = Path(args.output)
    build_dir = Path(args.build_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    grand_total = defaultdict(int)

    for variant in args.variants:
        raw_csv = output_dir / f"_raw_{variant}.csv"
        rc = run_variant(
            variant, args.category, args.workers, build_dir, raw_csv, args.shape_timeout
        )
        if rc != 0:
            print(f"\n  WARNING: {variant} exited with code {rc}", flush=True)

        counts = split_csv(raw_csv, output_dir)
        for key, n in counts.items():
            grand_total[key] += n
            family, dtype = key
            print(f"    {family}/{dtype}: {n} measurements")

    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 80}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"  Output dir: {output_dir}")
    print()
    print(f"  {'Family':<25} {'Dtype':<10} {'Measurements':>12}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 12}")
    total = 0
    for (family, dtype), n in sorted(grand_total.items()):
        print(f"  {family:<25} {dtype:<10} {n:>12,}")
        total += n
    print(f"  {'-' * 25} {'-' * 10} {'-' * 12}")
    print(f"  {'TOTAL':<25} {'':<10} {total:>12,}")

    print("\n  Directory structure:")
    for d in sorted(output_dir.rglob("results.csv")):
        rel = d.relative_to(output_dir)
        print(f"    {rel}")


if __name__ == "__main__":
    main()
