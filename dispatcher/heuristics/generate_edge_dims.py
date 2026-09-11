#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Supplementary edge-case benchmark generator for N=1 and K=1 dimensions.

These shapes represent vector-matrix multiply (N=1), rank-1 updates (K=1),
and other degenerate GEMM cases that stress tile efficiency and padding logic.
"""

import json
import subprocess
import sys
from pathlib import Path


def generate_edge_shapes():
    """Generate shapes with N=1, K=1, and other single-dimension edge cases."""
    shapes = set()

    # --- N=1: vector-matrix multiply / single output column ---
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        for k in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192]:
            shapes.add((m, 1, k))

    # --- K=1: rank-1 update / outer product ---
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        for n in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192]:
            shapes.add((m, n, 1))

    # --- M=1, N=1: dot product ---
    for k in [1, 16, 64, 256, 1024, 4096, 8192]:
        shapes.add((1, 1, k))

    # --- M=1, K=1: scalar-vector ---
    for n in [1, 16, 64, 256, 1024, 4096, 8192]:
        shapes.add((1, n, 1))

    # --- N=1, K=1: scalar-vector ---
    for m in [1, 16, 64, 256, 1024, 4096, 8192]:
        shapes.add((m, 1, 1))

    # --- All ones: 1x1x1 ---
    shapes.add((1, 1, 1))

    # --- Small N (2-16) ---
    for m in [64, 256, 1024, 4096]:
        for n in [2, 3, 4, 7, 8, 15, 16]:
            for k in [64, 256, 1024, 4096]:
                shapes.add((m, n, k))

    # --- Small K (2-16) ---
    for m in [64, 256, 1024, 4096]:
        for n in [64, 256, 1024, 4096]:
            for k in [2, 3, 4, 7, 8, 15, 16]:
                shapes.add((m, n, k))

    return sorted(shapes)


def run_shapes(bin_dir, shapes, out_file, warmup=3, repeat=10):
    """Run all kernels against shapes, writing streaming log."""
    executables = sorted(Path(bin_dir).glob("benchmark_gemm_universal_fp8_rcr_*"))
    if not executables:
        print(f"ERROR: No executables found in {bin_dir}", file=sys.stderr)
        return 0

    total = 0
    for idx, (m, n, k) in enumerate(shapes):
        out_file.write("\n========================================\n")
        out_file.write(f"Shape {idx + 1}: M={m} N={n} K={k} dtype=fp8 layout=rcr\n")
        out_file.write("========================================\n")
        out_file.write(f"Found {len(executables)} kernels\n")
        out_file.flush()

        for exe in executables:
            try:
                result = subprocess.run(
                    [
                        str(exe),
                        f"-m={m}",
                        f"-n={n}",
                        f"-k={k}",
                        f"-warmup={warmup}",
                        f"-repeat={repeat}",
                        "-verify=0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                output = result.stdout
                json_start = output.find("{")
                json_end = output.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_block = output[json_start:json_end]
                    try:
                        json.loads(json_block)
                        out_file.write(json_block + "\n")
                        total += 1
                    except json.JSONDecodeError:
                        pass
            except (subprocess.TimeoutExpired, Exception):
                pass

        out_file.flush()
        print(
            f"  Shape {idx + 1}/{len(shapes)}: M={m} N={n} K={k}",
            file=sys.stderr,
            flush=True,
        )

    return total


if __name__ == "__main__":
    bin_dir = "/workspace/ck_tile/bin"
    out_dir = Path("data/edge_dims")
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes = generate_edge_shapes()
    print(f"Generated {len(shapes)} edge-case shapes", file=sys.stderr, flush=True)

    n1_count = sum(1 for m, n, k in shapes if n == 1)
    k1_count = sum(1 for m, n, k in shapes if k == 1)
    both1 = sum(1 for m, n, k in shapes if n == 1 and k == 1)
    small_n = sum(1 for m, n, k in shapes if 2 <= n <= 16)
    small_k = sum(1 for m, n, k in shapes if 2 <= k <= 16)
    print(
        f"  N=1: {n1_count}, K=1: {k1_count}, both=1: {both1}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"  Small N(2-16): {small_n}, Small K(2-16): {small_k}",
        file=sys.stderr,
        flush=True,
    )

    batch_size = 25
    total = 0
    batch_idx = 0
    for i in range(0, len(shapes), batch_size):
        batch = shapes[i : i + batch_size]
        batch_idx += 1
        out_path = out_dir / f"edge_dims_batch_{batch_idx:03d}.log"
        print(
            f"\nBatch {batch_idx}: shapes {i + 1}-{i + len(batch)} -> {out_path}",
            file=sys.stderr,
            flush=True,
        )

        with open(out_path, "w") as f:
            f.write(f"CK Tile Edge Dims Benchmark Batch {batch_idx}\n")
            f.write("GPU ID: 0\nImplementation: gemm_universal\n\n")
            count = run_shapes(bin_dir, batch, f, warmup=3, repeat=10)
            total += count

        print(f"  Batch {batch_idx} done: {count} results", file=sys.stderr, flush=True)

    print(
        f"\nTotal: {total} benchmarks across {len(shapes)} shapes",
        file=sys.stderr,
        flush=True,
    )
