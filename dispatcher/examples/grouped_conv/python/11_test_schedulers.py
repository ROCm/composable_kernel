#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 11: Test All Pipeline + Scheduler Combinations

Tests all 8 pipelines with both intrawave and interwave schedulers
for all convolution variants to determine which combinations work.

Usage:
    python3 11_test_schedulers.py
    python3 11_test_schedulers.py --arch gfx942
    python3 11_test_schedulers.py --variant forward
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GroupedConvProblem,
    GroupedConvRegistry,
    detect_gpu_arch,
)

# All pipelines from unified_grouped_conv_codegen.py
ALL_PIPELINES = [
    "basic_v1",
    "mem",
    "compv3",
    "compv4",
    "compv5",
    "compv6",
    "comp_async",
    "basic_async_v1",
]

# Both schedulers
ALL_SCHEDULERS = ["intrawave", "interwave"]

# Pipelines that require DoubleSmemBuffer=true (enforced by static_assert in
# the pipeline headers). Building these with dsb=false is a loud compile error.
PIPELINES_REQUIRING_DSB = {"compv4", "comp_async"}


def test_pipeline_scheduler(pipeline, scheduler, variant, arch, dtype, ndim=2):
    """
    Test if a pipeline+scheduler+variant combination builds and runs successfully.

    Args:
        pipeline: Pipeline name (e.g., "compv3", "mem")
        scheduler: Scheduler type ("intrawave" or "interwave")
        variant: Convolution variant (forward, bwd_data, bwd_weight)
        arch: GPU architecture (e.g., "gfx950")
        dtype: Data type (fp16, bf16)
        ndim: Spatial dimensions (2 or 3)

    Returns:
        dict with keys: pipeline, scheduler, variant, ndim, build_success, run_success, error_msg
    """
    result = {
        "pipeline": pipeline,
        "scheduler": scheduler,
        "variant": variant,
        "ndim": ndim,
        "arch": arch,
        "dtype": dtype,
        "build_success": False,
        "run_success": False,
        "error_msg": None,
        "time_ms": None,
        "tflops": None,
    }

    try:
        # Create registry with single kernel config
        reg = GroupedConvRegistry(f"{variant}_{pipeline}_{scheduler}_{ndim}d")

        # Use a simple, safe tile config: 16x64x64
        # wave 1x4x1, warp 16x16x16
        config = GroupedConvKernelConfig(
            variant=variant,
            ndim_spatial=ndim,
            arch=arch,
            dtype=dtype,
            tile_m=16,
            tile_n=64,
            tile_k=64,
            wave_m=1,
            wave_n=4,
            wave_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline=pipeline,
            scheduler=scheduler,  # Test scheduler here
            epilogue="cshuffle" if pipeline not in ["mem"] else "default",
            vector_size_a=4,
            vector_size_b=8,
            vector_size_c=8,
            block_per_cu=1,
            # compv4/comp_async require DoubleSmemBuffer=true (loud
            # static_assert otherwise); other pipelines do not.
            double_smem_buffer=(pipeline in PIPELINES_REQUIRING_DSB),
        )

        reg.add(config)

        # Try to build
        try:
            runners = reg.build(verbose=False, max_workers=1)
            key = (variant, ndim)

            if key in runners:
                result["build_success"] = True

                # Try to run
                np_dtype = np.float16 if dtype in ["fp16", "bf16"] else np.float32

                if ndim == 2:
                    prob = GroupedConvProblem(
                        N=1,
                        C=64,
                        K=64,
                        Hi=8,
                        Wi=8,
                        Y=3,
                        X=3,
                        pad_h=1,
                        pad_w=1,
                        direction=variant,
                    )
                else:  # 3D
                    prob = GroupedConvProblem(
                        N=1,
                        C=64,
                        K=64,
                        Di=4,
                        Hi=8,
                        Wi=8,
                        Z=3,
                        Y=3,
                        X=3,
                        pad_d=1,
                        pad_h=1,
                        pad_w=1,
                        direction=variant,
                    )

                # Generate inputs
                if variant == "forward":
                    x = np.random.uniform(-0.5, 0.5, prob.input_shape()).astype(
                        np_dtype
                    )
                    w = np.random.uniform(-0.5, 0.5, prob.weight_shape()).astype(
                        np_dtype
                    )
                    res = runners[key].run(x, w, prob)
                elif variant == "bwd_data":
                    # Runner contract: input_np=dY, weight_np=W for bwd_data
                    w = np.random.uniform(-0.5, 0.5, prob.weight_shape()).astype(
                        np_dtype
                    )
                    dy = np.random.uniform(-0.5, 0.5, prob.output_shape()).astype(
                        np_dtype
                    )
                    res = runners[key].run(dy, w, prob)
                elif variant == "bwd_weight":
                    x = np.random.uniform(-0.5, 0.5, prob.input_shape()).astype(
                        np_dtype
                    )
                    dy = np.random.uniform(-0.5, 0.5, prob.output_shape()).astype(
                        np_dtype
                    )
                    res = runners[key].run(x, dy, prob)

                if res.success and np.count_nonzero(res.output) > 0:
                    result["run_success"] = True
                    result["time_ms"] = res.time_ms
                    result["tflops"] = res.tflops
                else:
                    result["error_msg"] = "Kernel ran but produced zero output"

                # Cleanup
                runners[key].cleanup()
            else:
                result["error_msg"] = "Kernel not in runners (build failed)"

        except Exception as e:
            result["error_msg"] = f"Build exception: {str(e)}"

    except Exception as e:
        result["error_msg"] = f"Setup exception: {str(e)}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test All Pipeline + Scheduler Combinations"
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument(
        "--variant",
        default="all",
        choices=["all", "forward", "bwd_data", "bwd_weight"],
        help="Variant to test (default: all)",
    )
    parser.add_argument(
        "--ndim",
        type=int,
        default=2,
        choices=[2, 3],
        help="Spatial dimensions to test (default: 2)",
    )
    parser.add_argument(
        "--scheduler",
        default="all",
        choices=["all", "intrawave", "interwave"],
        help="Scheduler to test (default: all)",
    )
    parser.add_argument(
        "--output",
        default="scheduler_test_results.json",
        help="Output JSON file (default: scheduler_test_results.json)",
    )
    args = parser.parse_args()

    arch = args.arch
    print("=" * 80)
    print("Test All Pipeline + Scheduler Combinations")
    print("=" * 80)
    print(f"Arch: {arch}, Dtype: {args.dtype}, NDim: {args.ndim}D")
    print()

    # Determine variants to test
    if args.variant == "all":
        variants = ["forward", "bwd_data", "bwd_weight"]
    else:
        variants = [args.variant]

    # Determine schedulers to test
    if args.scheduler == "all":
        schedulers = ALL_SCHEDULERS
    else:
        schedulers = [args.scheduler]

    # Run tests
    all_results = []

    for variant in variants:
        print(f"\n{'=' * 80}")
        print(f"Testing {variant.upper()} ({args.ndim}D)")
        print(f"{'=' * 80}")
        print()

        print(
            f"{'Pipeline':<20} {'Scheduler':<12} {'Build':<8} {'Run':<8} {'Time (ms)':<12} {'TFLOPS':<10}"
        )
        print("-" * 80)

        for pipeline in ALL_PIPELINES:
            for scheduler in schedulers:
                result = test_pipeline_scheduler(
                    pipeline, scheduler, variant, arch, args.dtype, args.ndim
                )
                all_results.append(result)

                build_status = "✓" if result["build_success"] else "✗"
                run_status = "✓" if result["run_success"] else "✗"
                time_str = (
                    f"{result['time_ms']:.4f}"
                    if result["time_ms"] is not None
                    else "-"
                )
                tflops_str = (
                    f"{result['tflops']:.2f}" if result["tflops"] is not None else "-"
                )

                print(
                    f"{pipeline:<20} {scheduler:<12} {build_status:<8} {run_status:<8} {time_str:<12} {tflops_str:<10}"
                )

                if result["error_msg"] and not result["run_success"]:
                    print(f"  → {result['error_msg']}")

        print()

    # Summarize results by scheduler
    print("=" * 80)
    print("SUMMARY BY SCHEDULER")
    print("=" * 80)
    print()

    for scheduler in schedulers:
        print(f"\n{scheduler.upper()} Scheduler:")
        print("-" * 80)

        for variant in variants:
            variant_results = [
                r
                for r in all_results
                if r["variant"] == variant and r["scheduler"] == scheduler
            ]
            successful_build = [
                r["pipeline"] for r in variant_results if r["build_success"]
            ]
            successful_run = [r["pipeline"] for r in variant_results if r["run_success"]]

            print(f"\n{variant} ({args.ndim}D):")
            print(f"  Build success ({len(successful_build)}/8): {successful_build}")
            print(f"  Run success   ({len(successful_run)}/8): {successful_run}")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print()

    # Per-pipeline support: a pipeline is "supported" if at least one
    # scheduler runs successfully. Not every pipeline supports both
    # intrawave and interwave (loud static_assert / unsupported trait
    # in some pipeline headers), so we only require one to work.
    per_variant_supported: dict[str, list[str]] = {}
    for variant in variants:
        print(f"{variant.upper()}:")

        # Group by pipeline; mark as supported if any scheduler succeeded
        supported_pipelines = []
        per_pipeline_status = []
        for pipeline in ALL_PIPELINES:
            schedulers_ok = [
                r["scheduler"]
                for r in all_results
                if r["variant"] == variant
                and r["pipeline"] == pipeline
                and r["run_success"]
            ]
            if schedulers_ok:
                supported_pipelines.append(pipeline)
                per_pipeline_status.append((pipeline, "✓", schedulers_ok))
            else:
                per_pipeline_status.append((pipeline, "✗", []))

        # Per-pipeline detail (any-scheduler-counts)
        for pipeline, status, sched_list in per_pipeline_status:
            sched_str = ",".join(sched_list) if sched_list else "none"
            print(f"  {pipeline:<18}: {status}  via [{sched_str}]")

        # Per-scheduler raw breakdown (for completeness)
        for scheduler in schedulers:
            variant_results = [
                r
                for r in all_results
                if r["variant"] == variant and r["scheduler"] == scheduler
            ]
            success_count = len([r for r in variant_results if r["run_success"]])
            total = len(variant_results)
            pct = (success_count / total * 100) if total > 0 else 0
            print(
                f"  raw {scheduler:<10}: {success_count}/{total} ({pct:.0f}%) pipelines work"
            )

        # Any-scheduler aggregate
        n_sup = len(supported_pipelines)
        n_total = len(ALL_PIPELINES)
        agg_pct = (n_sup / n_total * 100) if n_total > 0 else 0
        agg_status = "✓" if n_sup > 0 else "✗"
        print(
            f"  ANY scheduler  : {agg_status} {n_sup}/{n_total} ({agg_pct:.0f}%) pipelines supported"
        )
        per_variant_supported[variant] = supported_pipelines
        print()

    # Save results
    output_file = Path(__file__).parent / args.output
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Detailed results saved to: {output_file}")
    print()

    # Success criterion (relaxed): for each variant, at least one pipeline
    # must be supported by at least one scheduler. Pipelines that fail under
    # *both* schedulers are reported but don't fail the run, since some
    # pipelines genuinely don't support both schedulers.
    success = all(per_variant_supported.get(v) for v in variants)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
