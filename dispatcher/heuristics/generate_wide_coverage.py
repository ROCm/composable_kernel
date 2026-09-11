#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Wide-coverage benchmark data generator.

Generates benchmark results for hundreds of diverse GEMM shapes across all
corner cases: skinny M, tall N, deep K, M=1, prime dimensions, power-of-2,
LLM inference shapes, training shapes, and edge cases.

Runs all 4608 kernels in /workspace/ck_tile/bin/ against each shape and
writes streaming log output parseable by data_pipeline.py.

Usage:
    python3 generate_wide_coverage.py --bin_dir /workspace/ck_tile/bin \
        --out_dir data/ --batch_size 20 --warmup 3 --repeat 10
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def generate_shape_list():
    """Generate a comprehensive list of (M, N, K) shapes covering all corner cases.

    Categories:
      1. M=1 (single token inference) -- the hardest case
      2. Tiny M (2-16) -- small batch inference
      3. Small M (32-128) -- medium batch
      4. Medium M (256-2048) -- large batch / training
      5. Large M (4096-20480) -- very large batch
      6. Square shapes (powers of 2)
      7. Skinny M, tall N (M << N)
      8. Tall M, skinny N (M >> N)
      9. Deep K (K >> M, N) -- compute-bound
     10. Shallow K (K << M, N) -- memory-bound
     11. Prime dimensions -- worst-case for tiling
     12. LLM-specific shapes (DeepSeek, LLaMA, etc.)
     13. Non-power-of-2 common sizes
    """
    shapes = set()

    # --- 1. M=1 (single token) across various N, K ---
    for n in [512, 1024, 1536, 2048, 3072, 4096, 4608, 7168, 8192, 11008, 14336, 28672]:
        for k in [256, 512, 1024, 1536, 2048, 2304, 4096, 7168, 8192]:
            shapes.add((1, n, k))

    # --- 2. Tiny M (2-16) ---
    for m in [2, 4, 8, 16]:
        for n in [512, 1536, 4096, 7168]:
            for k in [256, 1024, 4096, 7168]:
                shapes.add((m, n, k))

    # --- 3. Small M (32-128) ---
    for m in [32, 48, 64, 96, 128]:
        for n in [512, 1536, 4096, 7168, 8192]:
            for k in [256, 512, 2048, 4096, 7168]:
                shapes.add((m, n, k))

    # --- 4. Medium M (256-2048) ---
    for m in [256, 384, 512, 768, 1024, 1536, 2048]:
        for n in [512, 1536, 4096, 7168]:
            for k in [256, 1024, 2048, 4096, 7168]:
                shapes.add((m, n, k))

    # --- 5. Large M (4096-20480) ---
    for m in [4096, 8192, 12288, 16384, 20480]:
        for n in [512, 1536, 4096, 7168]:
            for k in [256, 1024, 2048, 7168]:
                shapes.add((m, n, k))

    # --- 6. Square shapes (powers of 2) ---
    for p in range(5, 14):  # 32 to 8192
        d = 2**p
        shapes.add((d, d, d))

    # --- 7. Skinny M, tall N ---
    for m in [1, 4, 16, 64]:
        for n in [8192, 16384, 28672]:
            for k in [1024, 4096, 8192]:
                shapes.add((m, n, k))

    # --- 8. Tall M, skinny N ---
    for m in [4096, 8192, 16384]:
        for n in [32, 64, 128, 256]:
            for k in [1024, 4096]:
                shapes.add((m, n, k))

    # --- 9. Deep K (K >> M, N) ---
    for m in [16, 64, 256]:
        for n in [16, 64, 256]:
            for k in [4096, 8192, 16384, 32768]:
                shapes.add((m, n, k))

    # --- 10. Shallow K (K << M, N) ---
    for m in [1024, 4096, 8192]:
        for n in [1024, 4096, 8192]:
            for k in [16, 32, 64, 128]:
                shapes.add((m, n, k))

    # --- 11. Prime dimensions ---
    primes = [17, 31, 37, 127, 251, 509, 1021, 2039, 4093]
    for p in primes:
        shapes.add((p, p, p))
    for p in primes[:5]:
        shapes.add((p, 4096, 4096))
        shapes.add((4096, p, 4096))
        shapes.add((4096, 4096, p))

    # --- 12. LLM-specific shapes ---
    llm_shapes = [
        # DeepSeek MoE
        (1, 1536, 7168),
        (1, 4608, 7168),
        (1, 7168, 2048),
        (1, 7168, 2304),
        (1, 7168, 256),
        (1, 576, 7168),
        (1, 512, 7168),
        (1, 3072, 1536),
        # LLaMA-7B
        (1, 4096, 4096),
        (32, 4096, 4096),
        (128, 4096, 4096),
        (1, 4096, 11008),
        (32, 4096, 11008),
        (1, 11008, 4096),
        (32, 11008, 4096),
        # LLaMA-70B
        (1, 8192, 8192),
        (32, 8192, 8192),
        (128, 8192, 8192),
        (1, 8192, 28672),
        (32, 8192, 28672),
        (1, 28672, 8192),
        # GPT-style attention
        (128, 128, 64),
        (128, 128, 128),
        (256, 256, 64),
        (512, 512, 64),
        (1024, 1024, 64),
        (2048, 2048, 64),
    ]
    for s in llm_shapes:
        shapes.add(s)

    # --- 13. Non-power-of-2 common sizes ---
    for m in [48, 96, 192, 384, 576, 768, 1152, 1536, 2304, 3072, 4608, 6144]:
        shapes.add((m, m, m))
        shapes.add((m, 4096, 4096))

    sorted_shapes = sorted(shapes)
    return sorted_shapes


def run_shape_batch(bin_dir, shapes, out_file, warmup=3, repeat=10):
    """Run all kernels against a batch of shapes, writing streaming log output."""
    executables = sorted(Path(bin_dir).glob("benchmark_gemm_universal_fp8_rcr_*"))
    if not executables:
        print(f"ERROR: No executables found in {bin_dir}", file=sys.stderr)
        return 0

    total_benchmarks = 0

    for shape_idx, (m, n, k) in enumerate(shapes):
        out_file.write("\n========================================\n")
        out_file.write(
            f"Shape {shape_idx + 1}: M={m} N={n} K={k} dtype=fp8 layout=rcr\n"
        )
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
                # Extract JSON block from output
                json_start = output.find("{")
                json_end = output.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_block = output[json_start:json_end]
                    try:
                        json.loads(json_block)
                        out_file.write(json_block + "\n")
                        total_benchmarks += 1
                    except json.JSONDecodeError:
                        pass
            except (subprocess.TimeoutExpired, Exception):
                pass

        out_file.flush()
        elapsed_kernels = len(executables)
        print(
            f"  Shape {shape_idx + 1}/{len(shapes)}: M={m} N={n} K={k} "
            f"({elapsed_kernels} kernels)",
            file=sys.stderr,
            flush=True,
        )

    return total_benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="Generate wide-coverage benchmark data"
    )
    parser.add_argument(
        "--bin_dir",
        default="/workspace/ck_tile/bin",
        help="Directory with benchmark executables",
    )
    parser.add_argument("--out_dir", default="data", help="Output directory")
    parser.add_argument(
        "--batch_size", type=int, default=25, help="Shapes per output file"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--max_shapes", type=int, default=None, help="Limit total shapes (for testing)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes = generate_shape_list()
    if args.max_shapes:
        shapes = shapes[: args.max_shapes]

    print(f"Generated {len(shapes)} unique shapes", file=sys.stderr, flush=True)
    print(f"Bin dir: {args.bin_dir}", file=sys.stderr, flush=True)
    print(f"Output dir: {args.out_dir}", file=sys.stderr, flush=True)
    print(f"Batch size: {args.batch_size}", file=sys.stderr, flush=True)

    total = 0
    batch_idx = 0
    for i in range(0, len(shapes), args.batch_size):
        batch = shapes[i : i + args.batch_size]
        batch_idx += 1
        out_path = out_dir / f"wide_coverage_batch_{batch_idx:03d}.log"

        print(
            f"\nBatch {batch_idx}: shapes {i + 1}-{i + len(batch)} -> {out_path}",
            file=sys.stderr,
            flush=True,
        )

        with open(out_path, "w") as f:
            f.write(f"CK Tile Wide Coverage Benchmark Batch {batch_idx}\n")
            f.write("GPU ID: 0\n")
            f.write("Implementation: gemm_universal\n\n")
            count = run_shape_batch(
                args.bin_dir, batch, f, warmup=args.warmup, repeat=args.repeat
            )
            total += count

        print(
            f"  Batch {batch_idx} complete: {count} benchmarks",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"\nTotal: {total} benchmarks across {len(shapes)} shapes",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
