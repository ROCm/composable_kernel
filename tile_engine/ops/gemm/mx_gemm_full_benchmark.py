#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full MX GEMM (fp4/fp8 microscaling) benchmark sweep driven through the
Dispatcher bridge.

Same three phases as gemm_full_benchmark.py, whose multi-GPU Phase 3 runner it
reuses:
  Phase 1: Expand the Tile Engine MX sweep JSON and compile every kernel to a
           .so (parallel, no GPU)
  Phase 2: Load problems (M, N, K shapes)
  Phase 3: Benchmark via subprocess isolation across all visible GPUs

The sweep JSON is the one the Old-TE mx_gemm CMake build reads
(mx_gemm/configs/, tile_config/trait_config schema). The kernel list comes from
the Old-TE MxGemmKernelBuilder, so a bridge sweep runs exactly the kernels of an
Old-TE build of the same JSON. Kernels need gfx950 or gfx1250.

Examples:
    # Default: fp4, the arch's CI sweep config + example problems, all GPUs.
    python mx_gemm_full_benchmark.py

    # Full fp8 sweep with verification on 4 GPUs:
    python mx_gemm_full_benchmark.py mx_gemm/configs/default_config.json \\
        --dtype fp8 --verify --devices 4 --csv mx_out.csv

A bare config file name is looked up in mx_gemm/configs/. When no config is
given the driver uses bridge_ci_config_<arch>.json, a small sweep covering every
MX pipeline of the arch; default_config[_<arch>].json is the Old-TE full sweep.
"""

import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _THIS_DIR / "mx_gemm" / "configs"

import gemm_full_benchmark as gfb  # noqa: E402  (also sets up sys.path)
from dispatcher_common import normalize_arch  # noqa: E402
from mx_gemm_utils import (  # noqa: E402
    _get_arch,
    expand_mx_sweep,
    setup_multiple_mx_gemm_dispatchers,
)

SUPPORTED_DTYPES = ("fp4", "fp8")
EXAMPLE_PROBLEMS = _CONFIG_DIR / "example_problems.json"


def resolve_configs(configs, arch):
    """Positional configs (paths or bare names in mx_gemm/configs/) -> paths."""
    if not configs:
        names = (f"bridge_ci_config_{arch}.json", "default_ci_config.json")
        return [next(_CONFIG_DIR / n for n in names if (_CONFIG_DIR / n).exists())]
    return [Path(c) if Path(c).exists() else _CONFIG_DIR / c for c in configs]


def main():
    parser = argparse.ArgumentParser(
        description="MX GEMM Benchmark Sweep (via Dispatcher)"
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="TE MX sweep config JSON files (default: bridge_ci_config_<arch>.json)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="gfx950 or gfx1250. Auto-detected via rocminfo when omitted.",
    )
    parser.add_argument("--dtype", default="fp4", choices=SUPPORTED_DTYPES)
    parser.add_argument(
        "--problems",
        default=str(EXAMPLE_PROBLEMS),
        help="JSON file of M,N,K problems (default: mx_gemm/configs/example_problems.json)",
    )
    parser.add_argument("--csv", type=str, default="mx_gemm_results.csv")
    parser.add_argument("--workers", type=int, default=8, help="Parallel build workers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Build dir for generated headers and .so files; reusing it skips "
        "rebuilding kernels (default: a fresh temp dir)",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="GPUs to use: int count (e.g. 4) or comma-list (e.g. 0,2,5); "
        "bare digit is a count, not an id; default auto-detects all visible",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Kernels per subprocess (overhead vs fault isolation)",
    )
    parser.add_argument(
        "--kernel-timeout", type=int, default=60, help="Per-kernel timeout (s)"
    )
    parser.add_argument(
        "--max-kernels", type=int, default=0, help="Limit to first N kernels (0=all)"
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Sample at most N kernels per config, like the Old-TE "
        "MX_GEMM_MAX_INSTANCES CMake option (same Sobol/LHS sampler)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Sampling seed for --max-instances"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check each kernel's output against the numpy microscaled "
        "reference (global max|out-ref|/max|ref|); a mismatch counts as a failure",
    )
    parser.add_argument(
        "--verify-tol",
        type=float,
        default=gfb.DEFAULT_VERIFY_TOL,
        help=f"Relative tolerance for --verify (default {gfb.DEFAULT_VERIFY_TOL})",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Run Phase 1 only (compile all kernels, skip benchmarking)",
    )
    args = parser.parse_args()

    arch = normalize_arch(args.arch) if args.arch else _get_arch()
    config_paths = resolve_configs(args.configs, arch)
    devices = gfb.resolve_devices(args.devices)
    if not args.compile_only and not devices:
        parser.error("No GPUs detected; set --devices explicitly or use --compile-only")

    # ========================================================================
    # Phase 1: Compile kernels (parallel, no GPU)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile MX GEMM kernels")
    print(f"{'=' * 80}")
    print(f"  Configs: {', '.join(map(str, config_paths))}")
    print(f"  Arch: {arch}  Dtype: {args.dtype}  Layout: rcr")

    all_configs = []
    for cfg_path in config_paths:
        all_configs.extend(
            expand_mx_sweep(
                cfg_path,
                arch,
                dtype=args.dtype,
                max_instances=args.max_instances,
                seed=args.seed,
            )
        )
    if args.max_kernels > 0:
        all_configs = all_configs[: args.max_kernels]

    print(f"  Expanded configs: {len(all_configs)}")
    if not all_configs:
        print(f"  ERROR: 0 configs expanded for {args.dtype} on {arch}")
        return 1
    print(f"  Build workers: {args.workers}")

    t0 = time.perf_counter()
    lib_paths = setup_multiple_mx_gemm_dispatchers(
        all_configs,
        output_dir=args.output_dir,
        gfx_arch=arch,
        max_workers=args.workers,
    )
    build_time = time.perf_counter() - t0
    built_kernels, duplicate_count = gfb.unique_built_kernels(all_configs, lib_paths)
    n_build_fail = sum(lib is None for lib in lib_paths)

    print(
        f"\n  Built {len(all_configs)} configs -> {len(built_kernels)} unique kernels "
        f"({duplicate_count} duplicates filtered, {n_build_fail} build failures) "
        f"in {build_time:.0f}s"
    )
    if not built_kernels:
        print("  ERROR: No kernels built successfully")
        return 1
    if args.compile_only:
        print("\n  --compile-only: skipping benchmark phases")
        return 1 if n_build_fail else 0

    # ========================================================================
    # Phase 2: Load problems
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 2: Load test problems")
    print(f"{'=' * 80}")

    problems = gfb.load_problems(args.problems, gfb.DEFAULT_VARIANT)
    print(f"  Problems: {len(problems)}")
    print(
        f"  Total measurements: {len(built_kernels)} x {len(problems)} = "
        f"{len(built_kernels) * len(problems)}"
    )

    base_env = os.environ.copy()
    base_env["GEMM_PYPATH"] = os.pathsep.join(
        [str(gfb._DISPATCHER_ROOT / "python"), str(_THIS_DIR)]
    )
    csv_path = Path(args.csv)
    stats = gfb.benchmark_on_devices(
        problems,
        [range(len(built_kernels))] * len(problems),
        built_kernels,
        devices,
        args,
        _THIS_DIR / "run_one_mx_gemm_kernel.py",
        base_env,
        csv_path,
        extra={"dtype": args.dtype, "arch": arch},
    )
    gfb.print_summary(build_time, stats, devices, csv_path)
    if n_build_fail:
        print(f"  Build failures: {n_build_fail}")
    return 1 if stats["failures"] or n_build_fail else 0


if __name__ == "__main__":
    sys.exit(main())
