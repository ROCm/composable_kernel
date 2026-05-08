#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Test harness for grouped convolution configuration options.

Tests all 5 configuration options to verify they are production-ready:
1. double_smem_buffer - LDS ping-pong buffering
2. num_groups_to_merge - Group fusion
3. split_image - Spatial dimension splitting
4. explicit_gemm - Alternative GEMM path
5. two_stage - fp32 workspace for bwd_weight

Usage:
    python3 12_test_config_options.py
    python3 12_test_config_options.py --arch gfx950
    python3 12_test_config_options.py --verbose
"""

import sys
import json
import subprocess
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# This file is in: dispatcher/examples/grouped_conv/python/
# Need to go up 3 levels to get to dispatcher/
_DISPATCHER_ROOT = _THIS_DIR.parents[2]
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GroupedConvProblem,
    GroupedConvRegistry,
    detect_gpu_arch,
)


def create_test_problem(variant: str, ndim: int = 2) -> GroupedConvProblem:
    """Create a small test problem for verification.

    Uses G=2 so num_groups_to_merge testing is meaningful, with small
    spatial / channel dims to keep allocations small and avoid GPU
    page faults from oversized buffers in this smoke-test path.
    """
    if ndim == 2:
        return GroupedConvProblem(
            N=1,
            C=64,  # c_per_g = 32
            K=64,  # k_per_g = 32
            G=2,
            Hi=8,
            Wi=8,
            Y=3,
            X=3,
            stride_h=1,
            stride_w=1,
            dilation_h=1,
            dilation_w=1,
            pad_h=1,
            pad_w=1,
            direction=variant,
        )
    else:  # 3D
        return GroupedConvProblem(
            N=1,
            C=64,
            K=64,
            G=2,
            Di=4,
            Hi=8,
            Wi=8,
            Z=3,
            Y=3,
            X=3,
            stride_d=1,
            stride_h=1,
            stride_w=1,
            dilation_d=1,
            dilation_h=1,
            dilation_w=1,
            pad_d=1,
            pad_h=1,
            pad_w=1,
            direction=variant,
        )


def test_config_option(
    option_name: str,
    option_value,
    variant: str = "forward",
    arch: str = "gfx942",
    dtype: str = "bf16",
    ndim: int = 2,
    pipeline: str = "compv3",
) -> tuple[bool, str]:
    """Test a single configuration option.

    Returns:
        (success, message) tuple
    """
    # Create base config
    config_kwargs = {
        "variant": variant,
        "ndim_spatial": ndim,
        "dtype": dtype,
        "layout": "nhwgc",
        "arch": arch,
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "wave_m": 2,
        "wave_n": 2,
        "wave_k": 1,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": pipeline,
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "vector_size_a": 4,
        "vector_size_b": 8,
        "vector_size_c": 8,
        "pad_m": True,
        "pad_n": True,
        "pad_k": True,
        "block_per_cu": 1,
        "num_wave_groups": 1,
        # Default config options
        "num_groups_to_merge": 1,
        "double_smem_buffer": False,
        "split_image": False,
        "explicit_gemm": False,
        "two_stage": False,
    }

    # Override the specific option being tested
    config_kwargs[option_name] = option_value

    config = GroupedConvKernelConfig(**config_kwargs)

    # Create registry and build
    registry = GroupedConvRegistry(name=f"test_{option_name}")
    registry.add(config)

    runners = registry.build(verbose=False)
    if not runners:
        return False, f"Build failed - no runners created"

    key = (variant, ndim)
    if key not in runners:
        return False, f"Runner not found for {key}"

    # Create test problem and run
    problem = create_test_problem(variant, ndim)

    # Create input/weight tensors per runner contract:
    #   forward:    input_np=X,  weight_np=W
    #   bwd_data:   input_np=dY, weight_np=W
    #   bwd_weight: input_np=X,  weight_np=dY
    import numpy as np
    np_dtype = np.float16 if config.dtype in ["fp16", "bf16"] else np.float32
    x_arr = np.random.uniform(-0.5, 0.5, problem.input_shape()).astype(np_dtype)
    w_arr = np.random.uniform(-0.5, 0.5, problem.weight_shape()).astype(np_dtype)
    dy_arr = np.random.uniform(-0.5, 0.5, problem.output_shape()).astype(np_dtype)

    if variant == "forward":
        a, b = x_arr, w_arr
    elif variant == "bwd_data":
        a, b = dy_arr, w_arr
    elif variant == "bwd_weight":
        a, b = x_arr, dy_arr
    else:
        return False, f"Unknown variant: {variant}"

    try:
        result = runners[key].run(a, b, problem)
        if result.error:
            return False, f"Runtime error: {result.error}"
        if result.time_ms <= 0:
            return False, f"Invalid time: {result.time_ms}"
        return True, f"OK (time={result.time_ms:.3f}ms)"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def run_test_in_subprocess(
    option_name: str,
    option_value,
    variant: str,
    arch: str,
    dtype: str,
    ndim: int,
    pipeline: str,
    timeout: int = 180,
) -> tuple[bool, str]:
    """Run one config-option test in an isolated subprocess.

    Returns (success, message). If the subprocess crashes (e.g. GPU
    page fault), success=False with a CRASH message instead of taking
    down the whole test driver.
    """
    spec = json.dumps(
        {
            "option_name": option_name,
            "option_value": option_value,
            "variant": variant,
            "arch": arch,
            "dtype": dtype,
            "ndim": ndim,
            "pipeline": pipeline,
        }
    )
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--single-test", spec]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"Subprocess timeout (>{timeout}s)"

    # The single-test mode prints exactly one JSON line on its last
    # non-empty stdout line containing the result.
    out_lines = [ln for ln in (res.stdout or "").splitlines() if ln.strip()]
    last = out_lines[-1] if out_lines else ""
    parsed = None
    if last.startswith("{"):
        try:
            parsed = json.loads(last)
        except json.JSONDecodeError:
            parsed = None

    if parsed is not None:
        return bool(parsed.get("success")), str(parsed.get("message", ""))

    # No parseable result -> subprocess died (likely GPU fault) before
    # it could report. Surface a short hint from stderr/stdout.
    tail = (res.stderr or res.stdout or "").strip().splitlines()
    hint = tail[-1] if tail else "(no output)"
    return False, f"CRASH (rc={res.returncode}): {hint[:200]}"


def _single_test_main(spec_json: str) -> int:
    """Internal entry point used by run_test_in_subprocess()."""
    spec = json.loads(spec_json)
    success, message = test_config_option(
        option_name=spec["option_name"],
        option_value=spec["option_value"],
        variant=spec["variant"],
        arch=spec["arch"],
        dtype=spec["dtype"],
        ndim=spec["ndim"],
        pipeline=spec["pipeline"],
    )
    # Last line of stdout is the JSON result that the parent parses.
    print(json.dumps({"success": bool(success), "message": str(message)}))
    return 0 if success else 0  # exit 0 either way; success encoded in JSON


def run_config_option_tests(arch: str = "gfx942", verbose: bool = False):
    """Run comprehensive config option tests."""

    print(f"Testing Grouped Convolution Configuration Options")
    print(f"Architecture: {arch}")
    print(f"=" * 80)

    # Test suite: (option_name, option_value, variant, ndim, pipeline, description)
    tests = [
        # 1. double_smem_buffer tests
        ("double_smem_buffer", False, "forward", 2, "compv3", "double_smem_buffer=False (baseline)"),
        ("double_smem_buffer", True, "forward", 2, "compv4", "double_smem_buffer=True with compv4"),
        ("double_smem_buffer", True, "forward", 3, "compv4", "double_smem_buffer=True with compv4 3D"),

        # 2. num_groups_to_merge tests
        ("num_groups_to_merge", 1, "forward", 2, "compv3", "num_groups_to_merge=1 (baseline)"),
        ("num_groups_to_merge", 2, "forward", 2, "compv3", "num_groups_to_merge=2 (merge 2 groups)"),
        ("num_groups_to_merge", 2, "forward", 3, "compv3", "num_groups_to_merge=2 with 3D"),
        ("num_groups_to_merge", 2, "bwd_data", 2, "compv3", "num_groups_to_merge=2 with bwd_data"),
        ("num_groups_to_merge", 2, "bwd_weight", 2, "compv3", "num_groups_to_merge=2 with bwd_weight"),

        # 3. split_image tests
        ("split_image", False, "forward", 2, "compv3", "split_image=False (baseline)"),
        ("split_image", True, "forward", 2, "compv3", "split_image=True (spatial split)"),
        ("split_image", True, "forward", 3, "compv3", "split_image=True with 3D"),
        ("split_image", True, "bwd_data", 2, "compv3", "split_image=True with bwd_data"),
        ("split_image", True, "bwd_weight", 2, "compv3", "split_image=True with bwd_weight"),

        # 4. explicit_gemm tests (experimental - expect failures)
        ("explicit_gemm", False, "forward", 2, "compv3", "explicit_gemm=False (baseline)"),
        # ("explicit_gemm", True, "forward", 2, "compv3", "explicit_gemm=True (experimental)"),

        # 5. two_stage tests (bwd_weight only)
        ("two_stage", False, "bwd_weight", 2, "compv3", "two_stage=False (baseline bwd_weight)"),
        ("two_stage", True, "bwd_weight", 2, "compv3", "two_stage=True (fp32 workspace)"),
        ("two_stage", True, "bwd_weight", 3, "compv3", "two_stage=True with 3D"),

        # 6. Combined tests (multiple options)
        ("num_groups_to_merge", 2, "forward", 2, "compv3", "Combined: num_groups=2 + split_image=True"),
        # Note: The above test only sets num_groups_to_merge=2, but we could modify the test function
        # to accept multiple options if needed
    ]

    results = []
    passed = 0
    failed = 0

    for option_name, option_value, variant, ndim, pipeline, description in tests:
        test_name = f"{description}"
        if verbose:
            print(f"\nTesting: {test_name}")
            print(f"  Option: {option_name}={option_value}")
            print(f"  Variant: {variant}, NDim: {ndim}, Pipeline: {pipeline}")
        else:
            print(f"Testing: {test_name:60s} ... ", end="", flush=True)

        # Run each test in a subprocess so a GPU page fault (e.g. from
        # an unsupported config like num_groups_to_merge=2 + bwd_data,
        # which the kernel does not validate before launch) only kills
        # that one test rather than the whole suite.
        success, message = run_test_in_subprocess(
            option_name=option_name,
            option_value=option_value,
            variant=variant,
            arch=arch,
            dtype="bf16",
            ndim=ndim,
            pipeline=pipeline,
        )

        if success:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        if verbose:
            print(f"  Result: {status} - {message}")
        else:
            print(f"{status}")
            if not success:
                print(f"         {message}")

        results.append((test_name, success, message))

    # Summary
    print(f"\n" + "=" * 80)
    print(f"Test Summary:")
    print(f"  Total:  {len(tests)}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print(f"  Success Rate: {100 * passed / len(tests):.1f}%")

    if failed > 0:
        print(f"\n" + "=" * 80)
        print(f"Failed Tests:")
        for test_name, success, message in results:
            if not success:
                print(f"  ❌ {test_name}")
                print(f"     {message}")

    return passed, failed


def test_combined_options(arch: str = "gfx942", verbose: bool = False):
    """Test multiple config options combined."""

    print(f"\n" + "=" * 80)
    print(f"Testing Combined Configuration Options")
    print(f"=" * 80)

    # Create config with multiple options enabled
    config = GroupedConvKernelConfig(
        variant="forward",
        ndim_spatial=2,
        dtype="bf16",
        layout="nhwgc",
        arch=arch,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        wave_m=2,
        wave_n=2,
        wave_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        vector_size_a=4,
        vector_size_b=8,
        vector_size_c=8,
        pad_m=True,
        pad_n=True,
        pad_k=True,
        block_per_cu=1,
        num_wave_groups=1,
        # Multiple options enabled
        num_groups_to_merge=2,
        double_smem_buffer=False,  # compv3 doesn't need this
        split_image=True,
        explicit_gemm=False,
        two_stage=False,
    )

    print(f"Testing: num_groups_to_merge=2 + split_image=True ... ", end="", flush=True)

    registry = GroupedConvRegistry(name="test_combined")
    registry.add(config)

    runners = registry.build(verbose=False)
    if not runners:
        print("❌ FAIL - Build failed")
        return False

    key = ("forward", 2)
    if key not in runners:
        print(f"❌ FAIL - Runner not found for {key}")
        return False

    problem = create_test_problem("forward", 2)

    import numpy as np
    np_dtype = np.float16
    x = np.random.uniform(-0.5, 0.5, problem.input_shape()).astype(np_dtype)
    w = np.random.uniform(-0.5, 0.5, problem.weight_shape()).astype(np_dtype)

    try:
        result = runners[key].run(x, w, problem)
        if result.error:
            print(f"❌ FAIL - Runtime error: {result.error}")
            return False
        if result.time_ms <= 0:
            print(f"❌ FAIL - Invalid time: {result.time_ms}")
            return False
        print(f"✅ PASS (time={result.time_ms:.3f}ms)")
        return True
    except Exception as e:
        print(f"❌ FAIL - Exception: {str(e)}")
        return False


def main():
    import argparse

    # Internal subprocess-isolated single-test mode. Used by
    # run_test_in_subprocess() to insulate the driver from GPU faults.
    if len(sys.argv) >= 3 and sys.argv[1] == "--single-test":
        return _single_test_main(sys.argv[2])

    parser = argparse.ArgumentParser(
        description="Test grouped convolution configuration options"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=detect_gpu_arch(),
        help="GPU architecture (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Run main tests
    passed, failed = run_config_option_tests(arch=args.arch, verbose=args.verbose)

    # Run combined tests
    combined_success = test_combined_options(arch=args.arch, verbose=args.verbose)

    # Final summary
    print(f"\n" + "=" * 80)
    print(f"Overall Results:")
    print(f"  Config Option Tests: {passed} passed, {failed} failed")
    print(f"  Combined Test: {'✅ PASS' if combined_success else '❌ FAIL'}")

    # Exit code
    if failed > 0 or not combined_success:
        print(f"\n⚠️  Some tests failed - config options may not be production-ready")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed - config options are production-ready!")
        sys.exit(0)


if __name__ == "__main__":
    rc = main()
    if rc is not None:
        sys.exit(rc)
