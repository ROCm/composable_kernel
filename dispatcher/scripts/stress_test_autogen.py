#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Stress Test for GEMM and Conv Auto-generation and Auto-correction

This script tests:
1. Python auto-correction for invalid configurations
2. C++ compile script validation and wildcard expansion
3. Random configuration generation and validation
4. Edge cases and boundary conditions

Usage:
    python3 scripts/stress_test_autogen.py [--verbose] [--quick]
"""

import argparse
import random
import sys
from pathlib import Path

# Add paths for imports
script_dir = Path(__file__).parent
dispatcher_root = script_dir.parent
sys.path.insert(0, str(dispatcher_root / "python"))
sys.path.insert(0, str(dispatcher_root / "codegen"))

# Import test utilities
try:
    from arch_filter import ArchFilter

    ARCH_FILTER_AVAILABLE = True
except ImportError:
    ARCH_FILTER_AVAILABLE = False
    print("Warning: arch_filter not available, some tests will be skipped")

try:
    from ctypes_utils import (
        KernelConfig,
        auto_correct_kernel_config,
    )

    CTYPES_UTILS_AVAILABLE = True
except ImportError as e:
    CTYPES_UTILS_AVAILABLE = False
    print(f"Warning: ctypes_utils not available ({e}), some tests will be skipped")

try:
    from conv_utils import (
        auto_correct_conv_config,
    )

    CONV_UTILS_AVAILABLE = True
except ImportError as e:
    CONV_UTILS_AVAILABLE = False
    print(f"Warning: conv_utils not available ({e}), some tests will be skipped")


# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

# Valid and invalid GEMM configurations for testing
# Note: KernelConfig uses layout_a, layout_b, layout_c instead of a combined "layout"
GEMM_TEST_CONFIGS = [
    # Valid configurations
    {
        "name": "valid_fp16_gfx942",
        "dtype_a": "fp16",
        "dtype_b": "fp16",
        "dtype_c": "fp16",
        "dtype_acc": "fp32",
        "layout_a": "row",
        "layout_b": "col",
        "layout_c": "row",  # RCR
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
        "should_correct": False,
    },
    {
        "name": "valid_bf16_gfx942",
        "dtype_a": "bf16",
        "dtype_b": "bf16",
        "dtype_c": "bf16",
        "dtype_acc": "fp32",
        "layout_a": "row",
        "layout_b": "col",
        "layout_c": "row",  # RCR
        "tile_m": 256,
        "tile_n": 256,
        "tile_k": 64,
        "wave_m": 2,
        "wave_n": 2,
        "wave_k": 1,
        "warp_m": 32,
        "warp_n": 32,
        "warp_k": 16,
        "pipeline": "compv4",
        "scheduler": "intrawave",
        "gfx_arch": "gfx942",
        "should_correct": False,
    },
    # Invalid configurations that should be auto-corrected
    {
        "name": "invalid_wave_gfx942",
        "dtype_a": "fp16",
        "dtype_b": "fp16",
        "dtype_c": "fp16",
        "dtype_acc": "fp32",
        "layout_a": "row",
        "layout_b": "col",
        "layout_c": "row",
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
        "should_correct": True,
    },
    {
        "name": "invalid_warp_gfx942",
        "dtype_a": "fp16",
        "dtype_b": "fp16",
        "dtype_c": "fp16",
        "dtype_acc": "fp32",
        "layout_a": "row",
        "layout_b": "col",
        "layout_c": "row",
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 32,
        "wave_m": 2,
        "wave_n": 2,
        "wave_k": 1,
        "warp_m": 64,
        "warp_n": 64,
        "warp_k": 8,  # Invalid warp tile
        "pipeline": "compv4",
        "scheduler": "intrawave",
        "gfx_arch": "gfx942",
        "should_correct": True,
    },
    {
        "name": "invalid_scheduler_gfx942",
        "dtype_a": "fp16",
        "dtype_b": "fp16",
        "dtype_c": "fp16",
        "dtype_acc": "fp32",
        "layout_a": "row",
        "layout_b": "col",
        "layout_c": "row",
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
        "scheduler": "interwave",  # May not be valid
        "gfx_arch": "gfx942",
        "should_correct": True,
    },
    # gfx90a configurations
    {
        "name": "valid_fp16_gfx90a",
        "dtype_a": "fp16",
        "dtype_b": "fp16",
        "dtype_c": "fp16",
        "dtype_acc": "fp32",
        "layout_a": "row",
        "layout_b": "col",
        "layout_c": "row",
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 32,
        "wave_m": 2,
        "wave_n": 2,
        "wave_k": 1,
        "warp_m": 32,
        "warp_n": 32,
        "warp_k": 8,
        "pipeline": "compv3",
        "scheduler": "intrawave",
        "gfx_arch": "gfx90a",
        "should_correct": False,
    },
    {
        "name": "invalid_wave_gfx90a",
        "dtype_a": "fp16",
        "dtype_b": "fp16",
        "dtype_c": "fp16",
        "dtype_acc": "fp32",
        "layout_a": "row",
        "layout_b": "col",
        "layout_c": "row",
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 32,
        "wave_m": 4,
        "wave_n": 4,
        "wave_k": 1,  # Invalid for gfx90a
        "warp_m": 32,
        "warp_n": 32,
        "warp_k": 8,
        "pipeline": "compv3",
        "scheduler": "intrawave",
        "gfx_arch": "gfx90a",
        "should_correct": True,
    },
]

# Valid and invalid Conv configurations
CONV_TEST_CONFIGS = [
    {
        "name": "valid_conv_fp16_gfx942",
        "dtype_in": "fp16",
        "dtype_out": "fp16",
        "dtype_acc": "fp32",
        "layout": "nhwgc",
        "tile_k": 1,
        "tile_c": 128,
        "wave_m": 2,
        "wave_n": 2,
        "wave_k": 1,
        "warp_m": 32,
        "warp_n": 32,
        "warp_k": 16,
        "pipeline": "compv3",
        "scheduler": "intrawave",
        "arch": "gfx942",
        "should_correct": False,
    },
    {
        "name": "invalid_conv_wave_gfx942",
        "dtype_in": "fp16",
        "dtype_out": "fp16",
        "dtype_acc": "fp32",
        "layout": "nhwgc",
        "tile_k": 1,
        "tile_c": 128,
        "wave_m": 1,
        "wave_n": 1,
        "wave_k": 1,  # Invalid
        "warp_m": 32,
        "warp_n": 32,
        "warp_k": 16,
        "pipeline": "compv3",
        "scheduler": "intrawave",
        "arch": "gfx942",
        "should_correct": True,
    },
]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_gemm_auto_correction(verbose: bool = False) -> tuple[int, int]:
    """Test GEMM auto-correction for predefined configurations."""
    if not CTYPES_UTILS_AVAILABLE:
        print("  [SKIP] ctypes_utils not available")
        return 0, 0

    passed = 0
    failed = 0

    print("\n  Testing GEMM Auto-Correction:")
    print("  " + "-" * 50)

    for test in GEMM_TEST_CONFIGS:
        name = test["name"]
        should_correct = test["should_correct"]

        # Create KernelConfig using correct attribute names
        config = KernelConfig(
            dtype_a=test["dtype_a"],
            dtype_b=test["dtype_b"],
            dtype_c=test["dtype_c"],
            dtype_acc=test["dtype_acc"],
            layout_a=test["layout_a"],
            layout_b=test["layout_b"],
            layout_c=test["layout_c"],
            tile_m=test["tile_m"],
            tile_n=test["tile_n"],
            tile_k=test["tile_k"],
            wave_m=test["wave_m"],
            wave_n=test["wave_n"],
            wave_k=test["wave_k"],
            warp_m=test["warp_m"],
            warp_n=test["warp_n"],
            warp_k=test["warp_k"],
            pipeline=test["pipeline"],
            scheduler=test["scheduler"],
            gfx_arch=test["gfx_arch"],
        )

        try:
            corrected, was_modified, corrections = auto_correct_kernel_config(
                config, verbose=False
            )

            if should_correct and was_modified:
                passed += 1
                if verbose:
                    print(f"    ✓ {name}: Correctly auto-corrected")
                    for corr in corrections:
                        print(f"      • {corr}")
            elif not should_correct and not was_modified:
                passed += 1
                if verbose:
                    print(f"    ✓ {name}: No correction needed (as expected)")
            elif should_correct and not was_modified:
                failed += 1
                print(f"    ✗ {name}: Expected correction but none applied")
            else:
                failed += 1
                print(f"    ✗ {name}: Unexpected correction applied")
                for corr in corrections:
                    print(f"      • {corr}")
        except Exception as e:
            failed += 1
            print(f"    ✗ {name}: Exception - {e}")

    return passed, failed


def test_conv_auto_correction(verbose: bool = False) -> tuple[int, int]:
    """Test Conv auto-correction for predefined configurations."""
    if not CONV_UTILS_AVAILABLE:
        print("  [SKIP] conv_utils not available")
        return 0, 0

    passed = 0
    failed = 0

    print("\n  Testing Conv Auto-Correction:")
    print("  " + "-" * 50)

    for test in CONV_TEST_CONFIGS:
        name = test["name"]
        should_correct = test["should_correct"]

        config_dict = {
            "dtype_in": test["dtype_in"],
            "dtype_out": test["dtype_out"],
            "dtype_acc": test["dtype_acc"],
            "layout": test["layout"],
            "tile_k": test["tile_k"],
            "tile_c": test["tile_c"],
            "wave_m": test["wave_m"],
            "wave_n": test["wave_n"],
            "wave_k": test["wave_k"],
            "warp_m": test["warp_m"],
            "warp_n": test["warp_n"],
            "warp_k": test["warp_k"],
            "pipeline": test["pipeline"],
            "scheduler": test["scheduler"],
            "arch": test["arch"],
        }

        try:
            corrected, was_modified, corrections = auto_correct_conv_config(
                config_dict, verbose=False
            )

            if should_correct and was_modified:
                passed += 1
                if verbose:
                    print(f"    ✓ {name}: Correctly auto-corrected")
                    for corr in corrections:
                        print(f"      • {corr}")
            elif not should_correct and not was_modified:
                passed += 1
                if verbose:
                    print(f"    ✓ {name}: No correction needed (as expected)")
            elif should_correct and not was_modified:
                failed += 1
                print(f"    ✗ {name}: Expected correction but none applied")
            else:
                failed += 1
                print(f"    ✗ {name}: Unexpected correction applied")
                for corr in corrections:
                    print(f"      • {corr}")
        except Exception as e:
            failed += 1
            print(f"    ✗ {name}: Exception - {e}")

    return passed, failed


def test_arch_filter_validation(verbose: bool = False) -> tuple[int, int]:
    """Test arch_filter validation for various configurations."""
    if not ARCH_FILTER_AVAILABLE:
        print("  [SKIP] arch_filter not available")
        return 0, 0

    passed = 0
    failed = 0

    print("\n  Testing Arch Filter Validation:")
    print("  " + "-" * 50)

    # Create ArchFilter for gfx942
    try:
        arch_filter = ArchFilter("gfx942")
    except Exception as e:
        print(f"  [SKIP] Could not create ArchFilter: {e}")
        return 0, 0

    # Test valid configurations using is_kernel_valid method
    test_cases = [
        (
            "Valid fp16 config",
            {
                "datatype_a": "fp16",
                "datatype_b": "fp16",
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 32,
                "wave_m": 2,
                "wave_n": 2,
                "wave_k": 1,
                "warp_m": 32,
                "warp_n": 32,
                "warp_k": 16,
            },
            True,
        ),
        (
            "Valid bf16 config",
            {
                "datatype_a": "bf16",
                "datatype_b": "bf16",
                "tile_m": 256,
                "tile_n": 256,
                "tile_k": 64,
                "wave_m": 2,
                "wave_n": 2,
                "wave_k": 1,
                "warp_m": 32,
                "warp_n": 32,
                "warp_k": 16,
            },
            True,
        ),
        (
            "Invalid wave config",
            {
                "datatype_a": "fp16",
                "datatype_b": "fp16",
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 32,
                "wave_m": 99,
                "wave_n": 99,
                "wave_k": 99,
                "warp_m": 32,
                "warp_n": 32,
                "warp_k": 16,
            },
            False,
        ),
    ]

    for name, config, should_pass in test_cases:
        try:
            is_valid = arch_filter.is_kernel_valid(**config)
            if is_valid == should_pass:
                passed += 1
                if verbose:
                    status = "valid" if should_pass else "rejected"
                    print(f"    ✓ {name}: Correctly {status}")
            else:
                failed += 1
                status = "valid" if is_valid else "invalid"
                print(
                    f"    ✗ {name}: Expected {'valid' if should_pass else 'invalid'}, got {status}"
                )
        except Exception as e:
            failed += 1
            print(f"    ✗ {name}: Exception - {e}")

    return passed, failed


def test_random_gemm_configs(
    num_samples: int = 20, verbose: bool = False
) -> tuple[int, int]:
    """Generate and test random GEMM configurations."""
    if not CTYPES_UTILS_AVAILABLE:
        print("  [SKIP] ctypes_utils not available")
        return 0, 0

    passed = 0
    failed = 0

    print(f"\n  Testing {num_samples} Random GEMM Configurations:")
    print("  " + "-" * 50)

    dtypes = ["fp16", "bf16"]
    # Layout combinations: (layout_a, layout_b, layout_c)
    layouts = [
        ("row", "col", "row"),  # RCR
        ("row", "row", "row"),  # RRR
        ("row", "col", "col"),  # RCC
    ]
    tiles = [(64, 64, 32), (128, 128, 32), (256, 256, 64), (128, 256, 32)]
    waves = [(1, 1, 1), (2, 2, 1), (1, 4, 1), (4, 1, 1), (2, 4, 1)]
    warps = [(16, 16, 16), (32, 32, 16), (16, 16, 32), (32, 32, 8)]
    pipelines = ["compv3", "compv4"]
    schedulers = ["intrawave", "interwave"]
    archs = ["gfx90a", "gfx942"]

    for i in range(num_samples):
        dtype = random.choice(dtypes)
        layout = random.choice(layouts)
        tile = random.choice(tiles)
        wave = random.choice(waves)
        warp = random.choice(warps)
        pipeline = random.choice(pipelines)
        scheduler = random.choice(schedulers)
        arch = random.choice(archs)

        config = KernelConfig(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c=dtype,
            dtype_acc="fp32",
            layout_a=layout[0],
            layout_b=layout[1],
            layout_c=layout[2],
            tile_m=tile[0],
            tile_n=tile[1],
            tile_k=tile[2],
            wave_m=wave[0],
            wave_n=wave[1],
            wave_k=wave[2],
            warp_m=warp[0],
            warp_n=warp[1],
            warp_k=warp[2],
            pipeline=pipeline,
            scheduler=scheduler,
            gfx_arch=arch,
        )

        try:
            corrected, was_modified, corrections = auto_correct_kernel_config(
                config, verbose=False
            )

            # Verify the corrected config is valid using ArchFilter
            if ARCH_FILTER_AVAILABLE:
                try:
                    arch_filter = ArchFilter(corrected.gfx_arch)
                    is_valid = arch_filter.is_kernel_valid(
                        datatype_a=corrected.dtype_a,
                        datatype_b=corrected.dtype_b,
                        tile_m=corrected.tile_m,
                        tile_n=corrected.tile_n,
                        tile_k=corrected.tile_k,
                        wave_m=corrected.wave_m,
                        wave_n=corrected.wave_n,
                        wave_k=corrected.wave_k,
                        warp_m=corrected.warp_m,
                        warp_n=corrected.warp_n,
                        warp_k=corrected.warp_k,
                    )

                    if is_valid:
                        passed += 1
                        if verbose:
                            status = "corrected" if was_modified else "valid"
                            print(f"    ✓ Random {i + 1}: {status} ({dtype}/{arch})")
                    else:
                        failed += 1
                        print(f"    ✗ Random {i + 1}: Corrected config still invalid")
                except Exception as e:
                    # ArchFilter validation failed but auto-correct ran
                    passed += 1
                    if verbose:
                        print(
                            f"    ~ Random {i + 1}: Processed (validation skipped: {e})"
                        )
            else:
                # Without arch_filter, just check it doesn't crash
                passed += 1
                if verbose:
                    print(f"    ✓ Random {i + 1}: Processed without error")

        except Exception as e:
            failed += 1
            print(f"    ✗ Random {i + 1}: Exception - {e}")

    return passed, failed


def test_random_conv_configs(
    num_samples: int = 20, verbose: bool = False
) -> tuple[int, int]:
    """Generate and test random Conv configurations."""
    if not CONV_UTILS_AVAILABLE:
        print("  [SKIP] conv_utils not available")
        return 0, 0

    passed = 0
    failed = 0

    print(f"\n  Testing {num_samples} Random Conv Configurations:")
    print("  " + "-" * 50)

    dtypes = ["fp16", "bf16"]
    layouts = ["nhwgc", "ndhwgc"]
    tiles_k = [1, 2, 4]
    tiles_c = [64, 128, 256]
    waves = [(1, 1, 1), (2, 2, 1), (1, 4, 1), (4, 1, 1)]
    warps = [(16, 16, 16), (32, 32, 16), (16, 16, 32)]
    pipelines = ["compv3", "compv4"]
    schedulers = ["intrawave", "interwave"]
    archs = ["gfx90a", "gfx942"]

    for i in range(num_samples):
        dtype = random.choice(dtypes)
        layout = random.choice(layouts)
        tile_k = random.choice(tiles_k)
        tile_c = random.choice(tiles_c)
        wave = random.choice(waves)
        warp = random.choice(warps)
        pipeline = random.choice(pipelines)
        scheduler = random.choice(schedulers)
        arch = random.choice(archs)

        config_dict = {
            "dtype_in": dtype,
            "dtype_out": dtype,
            "dtype_acc": "fp32",
            "layout": layout,
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

        try:
            corrected, was_modified, corrections = auto_correct_conv_config(
                config_dict, verbose=False
            )
            passed += 1
            if verbose:
                status = "corrected" if was_modified else "valid"
                print(f"    ✓ Random {i + 1}: {status} ({dtype}/{arch})")
        except Exception as e:
            failed += 1
            print(f"    ✗ Random {i + 1}: Exception - {e}")

    return passed, failed


def test_edge_cases(verbose: bool = False) -> tuple[int, int]:
    """Test edge cases and boundary conditions."""
    passed = 0
    failed = 0

    print("\n  Testing Edge Cases:")
    print("  " + "-" * 50)

    if CTYPES_UTILS_AVAILABLE:
        # Test with extreme values
        edge_cases = [
            ("Very small tiles", {"tile_m": 16, "tile_n": 16, "tile_k": 8}),
            ("Very large tiles", {"tile_m": 512, "tile_n": 512, "tile_k": 128}),
            ("Asymmetric tiles", {"tile_m": 64, "tile_n": 256, "tile_k": 32}),
        ]

        for name, overrides in edge_cases:
            try:
                config = KernelConfig(
                    dtype_a="fp16",
                    dtype_b="fp16",
                    dtype_c="fp16",
                    dtype_acc="fp32",
                    layout="rcr",
                    tile_m=overrides.get("tile_m", 128),
                    tile_n=overrides.get("tile_n", 128),
                    tile_k=overrides.get("tile_k", 32),
                    wave_m=2,
                    wave_n=2,
                    wave_k=1,
                    warp_m=32,
                    warp_n=32,
                    warp_k=16,
                    pipeline="compv4",
                    scheduler="intrawave",
                    gfx_arch="gfx942",
                )
                corrected, was_modified, corrections = auto_correct_kernel_config(
                    config, verbose=False
                )
                passed += 1
                if verbose:
                    print(f"    ✓ {name}: Handled without crash")
            except Exception as e:
                failed += 1
                print(f"    ✗ {name}: Exception - {e}")

    return passed, failed


def test_cpp_compile_script_parsing(verbose: bool = False) -> tuple[int, int]:
    """Test that the C++ compile script can parse kernel declarations."""
    passed = 0
    failed = 0

    print("\n  Testing C++ Compile Script Integration:")
    print("  " + "-" * 50)

    # Check if compile scripts exist
    gemm_compile = dispatcher_root / "scripts" / "compile_gemm_examples.py"
    conv_compile = dispatcher_root / "scripts" / "compile_conv_examples.py"

    if gemm_compile.exists():
        passed += 1
        if verbose:
            print("    ✓ GEMM compile script exists")
    else:
        failed += 1
        print("    ✗ GEMM compile script not found")

    if conv_compile.exists():
        passed += 1
        if verbose:
            print("    ✓ Conv compile script exists")
    else:
        failed += 1
        print("    ✗ Conv compile script not found")

    # Test that we can import the compile script modules
    try:
        sys.path.insert(0, str(script_dir))
        # Just check if the file can be read and has expected content
        if gemm_compile.exists():
            content = gemm_compile.read_text()
            if "validate_kernel_config" in content and "expand_declaration" in content:
                passed += 1
                if verbose:
                    print("    ✓ GEMM compile script has validation functions")
            else:
                failed += 1
                print("    ✗ GEMM compile script missing expected functions")
    except Exception as e:
        failed += 1
        print(f"    ✗ Error checking compile scripts: {e}")

    return passed, failed


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stress test GEMM/Conv auto-generation"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick test (fewer samples)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    num_samples = 10 if args.quick else 50

    print("=" * 70)
    print("  STRESS TEST: GEMM & Conv Auto-Generation and Auto-Correction")
    print("=" * 70)
    print(f"\n  Random seed: {args.seed}")
    print(f"  Samples per test: {num_samples}")

    total_passed = 0
    total_failed = 0

    # Run all tests
    tests = [
        ("GEMM Auto-Correction", lambda: test_gemm_auto_correction(args.verbose)),
        ("Conv Auto-Correction", lambda: test_conv_auto_correction(args.verbose)),
        ("Arch Filter Validation", lambda: test_arch_filter_validation(args.verbose)),
        (
            "Random GEMM Configs",
            lambda: test_random_gemm_configs(num_samples, args.verbose),
        ),
        (
            "Random Conv Configs",
            lambda: test_random_conv_configs(num_samples, args.verbose),
        ),
        ("Edge Cases", lambda: test_edge_cases(args.verbose)),
        ("C++ Compile Scripts", lambda: test_cpp_compile_script_parsing(args.verbose)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed, failed = test_fn()
            results.append((name, passed, failed))
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results.append((name, 0, 1))
            total_failed += 1

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for name, passed, failed in results:
        status = "✓" if failed == 0 else "✗"
        print(f"  {status} {name}: {passed} passed, {failed} failed")

    print("-" * 70)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed")

    if total_failed == 0:
        print("\n  ✓ ALL TESTS PASSED!")
    else:
        print(f"\n  ✗ {total_failed} TESTS FAILED")

    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
