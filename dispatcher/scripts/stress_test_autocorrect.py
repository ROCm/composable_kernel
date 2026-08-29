#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Stress Test for Auto-Correction and Codegen

This script tests the robustness of:
1. GEMM auto-correction (Python)
2. Conv auto-correction (Python)
3. C++ kernel declaration validation and wildcard expansion
4. Architecture filtering

Usage:
    python3 scripts/stress_test_autocorrect.py [--arch gfx942] [--samples 50] [--verbose]
"""

import argparse
import random
import sys
from pathlib import Path

# Add paths for imports
dispatcher_root = Path(__file__).parent.parent
sys.path.insert(0, str(dispatcher_root / "python"))
sys.path.insert(0, str(dispatcher_root / "codegen"))
sys.path.insert(0, str(dispatcher_root / "scripts"))

from ctypes_utils import auto_correct_kernel_config, KernelConfig  # noqa: E402

# Import validation/expansion functions from compile scripts
from compile_gemm_examples import (  # noqa: E402
    validate_kernel_config,
    expand_declaration_with_arch_filter,
)
from compile_grouped_conv_examples import (  # noqa: E402
    validate_grouped_conv_kernel_config as validate_conv_kernel_config,
    expand_grouped_conv_declaration_with_arch_filter as expand_conv_declaration_with_arch_filter,
)


# =============================================================================
# TEST PARAMETERS
# =============================================================================

# Valid dtypes
DTYPES = ["fp16", "bf16", "fp32", "fp8", "bf8", "int8"]

# Valid layouts
LAYOUTS = ["rcr", "rrr", "crr", "ccr"]

# Tile sizes (some valid, some invalid)
TILE_SIZES = [
    (32, 32, 16),
    (64, 64, 32),
    (128, 128, 32),
    (256, 256, 64),
    (128, 256, 32),
    (256, 128, 32),
    # Invalid sizes to test auto-correction
    (100, 100, 50),
    (17, 17, 17),
    (512, 512, 128),
]

# Wave configs (some valid, some invalid)
WAVE_CONFIGS = [
    (1, 1, 1),
    (1, 2, 1),
    (2, 1, 1),
    (2, 2, 1),
    (1, 4, 1),
    (4, 1, 1),
    (2, 4, 1),
    (4, 2, 1),
    # Invalid configs to test auto-correction
    (3, 3, 1),
    (5, 5, 1),
    (1, 1, 2),
]

# Warp tile sizes (some valid, some invalid)
WARP_TILES = [
    (16, 16, 16),
    (16, 16, 32),
    (32, 32, 8),
    (32, 32, 16),
    # Invalid tiles to test auto-correction
    (48, 48, 24),
    (64, 64, 32),
]

# Pipelines and schedulers
PIPELINES = ["compv3", "compv4", "flatmma", "invalid_pipeline"]
SCHEDULERS = ["intrawave", "interwave", "invalid_scheduler"]

# Architectures
ARCHS = ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1200", "gfx1201"]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def generate_random_gemm_config():
    """Generate a random GEMM configuration (may be invalid)."""
    dtype = random.choice(DTYPES)
    layout = random.choice(LAYOUTS)
    tile = random.choice(TILE_SIZES)
    wave = random.choice(WAVE_CONFIGS)
    warp = random.choice(WARP_TILES)
    pipeline = random.choice(PIPELINES)
    scheduler = random.choice(SCHEDULERS)
    arch = random.choice(ARCHS)

    return {
        "name": f"test_{dtype}_{layout}_{tile[0]}x{tile[1]}x{tile[2]}",
        "dtype_a": dtype,
        "dtype_b": dtype,
        "dtype_c": dtype,
        "dtype_acc": "fp32",
        "layout": layout,
        "tile_m": tile[0],
        "tile_n": tile[1],
        "tile_k": tile[2],
        "wave_m": wave[0],
        "wave_n": wave[1],
        "wave_k": wave[2],
        "warp_m": warp[0],
        "warp_n": warp[1],
        "warp_k": warp[2],
        "pipeline": pipeline,
        "scheduler": scheduler,
        "arch": arch,
    }


def generate_random_conv_config():
    """Generate a random Conv configuration (may be invalid)."""
    dtype = random.choice(["fp16", "bf16"])
    tile_k = random.choice([64, 128, 256])
    tile_c = random.choice([64, 128, 256])
    wave = random.choice(WAVE_CONFIGS)
    warp = random.choice(WARP_TILES)
    pipeline = random.choice(["compv3", "compv4"])
    scheduler = random.choice(["intrawave"])
    arch = random.choice(ARCHS)

    return {
        "name": f"test_conv_{dtype}_{tile_k}x{tile_c}",
        "dtype": dtype,
        "layout": "nhwgc",
        "conv_type": "forward",
        "tile_k": tile_k,
        "tile_c": tile_c,
        "wave_m": wave[0],
        "wave_n": wave[1],
        "wave_k": wave[2],
        "warp_m": warp[0],
        "warp_n": warp[1],
        "warp_k": warp[2],
        "pipeline": pipeline,
        "scheduler": scheduler,
        "arch": arch,
    }


def test_gemm_validation(config, verbose=False):
    """Test GEMM validation and auto-correction."""
    arch = config.get("arch", "gfx942")
    is_valid, error_msg = validate_kernel_config(config, arch)

    result = {
        "config": config,
        "is_valid": is_valid,
        "error_msg": error_msg,
        "expanded": [],
        "auto_corrected": None,
    }

    if not is_valid:
        # Try wildcard expansion
        wildcard_config = config.copy()
        wildcard_config["wave_m"] = -1
        wildcard_config["wave_n"] = -1
        wildcard_config["warp_m"] = -1
        wildcard_config["warp_n"] = -1
        wildcard_config["pipeline"] = "*"
        wildcard_config["scheduler"] = "*"

        expanded = expand_declaration_with_arch_filter(wildcard_config, arch)
        result["expanded"] = expanded

    if verbose:
        print(f"\n  Config: {config['name']}")
        print(f"    Valid: {is_valid}")
        if not is_valid:
            print(f"    Error: {error_msg[:80]}...")
            print(f"    Expanded to: {len(result['expanded'])} configurations")

    return result


def test_python_autocorrect(verbose=False):
    """Test Python auto-correction for GEMM KernelConfig."""
    print("\n" + "=" * 70)
    print("  PYTHON AUTO-CORRECTION TEST (GEMM KernelConfig)")
    print("=" * 70)

    test_cases = [
        # Valid config
        {
            "name": "valid_fp16",
            "dtype_a": "fp16",
            "dtype_b": "fp16",
            "dtype_c": "fp16",
            "dtype_acc": "fp32",
            "layout": "rcr",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 32,
            "wave_m": 2,
            "wave_n": 2,
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
            "gfx_arch": "gfx942",
        },
        # Invalid wave config
        {
            "name": "invalid_wave",
            "dtype_a": "fp16",
            "dtype_b": "fp16",
            "dtype_c": "fp16",
            "dtype_acc": "fp32",
            "layout": "rcr",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 32,
            "wave_m": 1,
            "wave_n": 1,
            "wave_k": 1,  # Invalid for gfx942
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
            "gfx_arch": "gfx942",
        },
        # Invalid scheduler
        {
            "name": "invalid_scheduler",
            "dtype_a": "fp16",
            "dtype_b": "fp16",
            "dtype_c": "fp16",
            "dtype_acc": "fp32",
            "layout": "rcr",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 32,
            "wave_m": 2,
            "wave_n": 2,
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "interwave",  # May not be valid for all archs
            "gfx_arch": "gfx942",
        },
    ]

    results = {"passed": 0, "failed": 0, "details": []}

    for tc in test_cases:
        try:
            config = KernelConfig()
            config.dtype_a = tc["dtype_a"]
            config.dtype_b = tc["dtype_b"]
            config.dtype_c = tc["dtype_c"]
            config.dtype_acc = tc["dtype_acc"]
            config.tile_m = tc["tile_m"]
            config.tile_n = tc["tile_n"]
            config.tile_k = tc["tile_k"]
            config.wave_m = tc["wave_m"]
            config.wave_n = tc["wave_n"]
            config.wave_k = tc["wave_k"]
            config.warp_m = tc["warp_m"]
            config.warp_n = tc["warp_n"]
            config.warp_k = tc["warp_k"]
            config.pipeline = tc["pipeline"]
            config.scheduler = tc["scheduler"]
            config.gfx_arch = tc["gfx_arch"]

            corrected, was_modified, corrections = auto_correct_kernel_config(
                config, verbose=verbose
            )

            results["passed"] += 1
            results["details"].append(
                {
                    "name": tc["name"],
                    "status": "PASS",
                    "was_modified": was_modified,
                    "corrections": corrections,
                }
            )

            if verbose:
                print(f"\n  {tc['name']}: PASS")
                if was_modified:
                    print(f"    Modified: {len(corrections)} correction(s)")
                    for c in corrections:
                        print(f"      - {c}")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(
                {"name": tc["name"], "status": "FAIL", "error": str(e)}
            )
            if verbose:
                print(f"\n  {tc['name']}: FAIL - {e}")

    print(f"\n  Summary: {results['passed']} passed, {results['failed']} failed")
    return results


def run_stress_test(arch, num_samples, verbose):
    """Run the full stress test."""
    print("\n" + "=" * 70)
    print("  DISPATCHER AUTO-CORRECTION & CODEGEN STRESS TEST")
    print("=" * 70)
    print(f"  Target Architecture: {arch}")
    print(f"  Number of Samples:   {num_samples}")
    print("=" * 70)

    # Test 1: GEMM Validation
    print("\n" + "-" * 70)
    print("  TEST 1: GEMM Validation & Wildcard Expansion")
    print("-" * 70)

    gemm_results = {"valid": 0, "invalid": 0, "expanded": 0, "expansion_failed": 0}

    for i in range(num_samples):
        config = generate_random_gemm_config()
        config["arch"] = arch  # Override with target arch

        result = test_gemm_validation(config, verbose)

        if result["is_valid"]:
            gemm_results["valid"] += 1
        else:
            gemm_results["invalid"] += 1
            if result["expanded"]:
                gemm_results["expanded"] += 1
            else:
                gemm_results["expansion_failed"] += 1

    print("\n  GEMM Results:")
    print(f"    Valid configs:         {gemm_results['valid']}")
    print(f"    Invalid configs:       {gemm_results['invalid']}")
    print(f"    Successfully expanded: {gemm_results['expanded']}")
    print(f"    Expansion failed:      {gemm_results['expansion_failed']}")

    # Test 2: Conv Validation
    print("\n" + "-" * 70)
    print("  TEST 2: Conv Validation & Wildcard Expansion")
    print("-" * 70)

    conv_results = {"valid": 0, "invalid": 0, "expanded": 0, "expansion_failed": 0}

    for i in range(num_samples):
        config = generate_random_conv_config()
        config["arch"] = arch  # Override with target arch

        is_valid, error_msg = validate_conv_kernel_config(config, arch)

        if is_valid:
            conv_results["valid"] += 1
        else:
            conv_results["invalid"] += 1
            # Try wildcard expansion
            wildcard_config = config.copy()
            wildcard_config["wave_m"] = -1
            wildcard_config["wave_n"] = -1
            wildcard_config["warp_m"] = -1
            wildcard_config["warp_n"] = -1

            expanded = expand_conv_declaration_with_arch_filter(wildcard_config, arch)
            if expanded:
                conv_results["expanded"] += 1
            else:
                conv_results["expansion_failed"] += 1

    print("\n  Conv Results:")
    print(f"    Valid configs:         {conv_results['valid']}")
    print(f"    Invalid configs:       {conv_results['invalid']}")
    print(f"    Successfully expanded: {conv_results['expanded']}")
    print(f"    Expansion failed:      {conv_results['expansion_failed']}")

    # Test 3: Python Auto-Correction
    print("\n" + "-" * 70)
    print("  TEST 3: Python Auto-Correction (KernelConfig)")
    print("-" * 70)

    py_results = test_python_autocorrect(verbose)

    # Test 4: Architecture-specific tests
    print("\n" + "-" * 70)
    print("  TEST 4: Architecture-Specific Validation")
    print("-" * 70)

    arch_test_configs = [
        # fp16 should work on all archs
        {"dtype": "fp16", "expected_archs": ARCHS},
        # bf16 works on all archs that have bf16_bf16_fp32 in warp_tile_combos
        {
            "dtype": "bf16",
            "expected_archs": [
                "gfx908",
                "gfx90a",
                "gfx942",
                "gfx950",
                "gfx1100",
                "gfx1200",
                "gfx1201",
            ],
        },
        # fp8 works on archs that have fp8_fp8_fp32 in warp_tile_combos
        {
            "dtype": "fp8",
            "expected_archs": ["gfx90a", "gfx942", "gfx950", "gfx1200", "gfx1201"],
        },
    ]

    for test in arch_test_configs:
        dtype = test["dtype"]
        print(f"\n  Testing {dtype}:")

        for test_arch in ARCHS:
            config = {
                "name": f"arch_test_{dtype}_{test_arch}",
                "dtype_a": dtype,
                "dtype_b": dtype,
                "dtype_c": dtype,
                "dtype_acc": "fp32",
                "layout": "rcr",
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 32,
                "wave_m": -1,  # Wildcard
                "wave_n": -1,
                "wave_k": 1,
                "warp_m": -1,
                "warp_n": -1,
                "warp_k": -1,
                "pipeline": "*",
                "scheduler": "*",
                "arch": test_arch,
            }

            expanded = expand_declaration_with_arch_filter(config, test_arch)
            status = "OK" if expanded else "FAIL"
            expected = test_arch in test["expected_archs"]
            match = "OK" if (bool(expanded) == expected) else "MISMATCH"

            if verbose or match == "MISMATCH":
                print(f"    {test_arch}: {status} ({len(expanded)} configs) [{match}]")

    # Summary
    print("\n" + "=" * 70)
    print("  STRESS TEST SUMMARY")
    print("=" * 70)
    print(
        f"  GEMM: {gemm_results['valid'] + gemm_results['expanded']}/{num_samples} handled"
    )
    print(
        f"  Conv: {conv_results['valid'] + conv_results['expanded']}/{num_samples} handled"
    )
    print(
        f"  Python Auto-Correct: {py_results['passed']}/{py_results['passed'] + py_results['failed']} passed"
    )

    total_success = (
        gemm_results["valid"]
        + gemm_results["expanded"]
        + conv_results["valid"]
        + conv_results["expanded"]
        + py_results["passed"]
    )
    total_tests = num_samples * 2 + py_results["passed"] + py_results["failed"]

    print(f"\n  Overall: {total_success}/{total_tests} tests handled successfully")
    print("=" * 70)

    return (
        gemm_results["expansion_failed"] == 0 and conv_results["expansion_failed"] == 0
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stress test auto-correction and codegen"
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        choices=ARCHS,
        help="Target GPU architecture (default: gfx942)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of random samples to test (default: 50)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    success = run_stress_test(args.arch, args.samples, args.verbose)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
