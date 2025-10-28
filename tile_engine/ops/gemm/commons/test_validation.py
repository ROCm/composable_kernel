#!/usr/bin/env python
"""
Test script to verify that the validation logic is working correctly.
"""

from validation_utils import (
    is_tile_config_valid,
    is_trait_combination_valid,
    validate_warp_tile_combination,
)


def test_warp_tile_validation():
    """Test warp tile combination validation"""
    print("Testing warp tile combination validation...")

    # Get GPU name
    gpu_name = "gfx90a"

    # Test cases for fp16
    test_cases = [
        # (warp_tile_m, warp_tile_n, warp_tile_k, expected_valid)
        ([4, 64, 8], False),  # Invalid - not in supported list
        ([4, 64, 16], True),  # Valid
        ([32, 32, 8], True),  # Valid
        ([16, 16, 16], True),  # Valid
        ([32, 32, 16], True),  # Valid
        ([16, 16, 32], True),  # Valid
        ([64, 4, 16], True),  # Valid
        ([128, 128, 128], False),  # Invalid - too large
    ]

    print("\nTesting fp16 warp tile combinations:")
    for (warp_tile_m, warp_tile_n, warp_tile_k), expected in test_cases:
        valid, msg = validate_warp_tile_combination(
            warp_tile_m, warp_tile_n, warp_tile_k, "fp16", "fp16", "fp16", gpu_name
        )
        status = "PASS" if valid == expected else "FAIL"
        print(f"  [{warp_tile_m}, {warp_tile_n}, {warp_tile_k}]: {valid} - {status}")
        if not valid and msg:
            print(f"    Reason: {msg}")


def test_trait_combinations():
    """Test trait combination validation"""
    print("\n\nTesting trait combination validation...")

    test_cases = [
        # (pipeline, epilogue, scheduler, expected_valid)
        ("mem", "default", "intrawave", True),
        ("mem", "cshuffle", "intrawave", True),
        ("compv3", "default", "interwave", False),  # Invalid combination
        ("compv3", "cshuffle", "interwave", False),  # Invalid combination
        ("compv4", "default", "interwave", False),  # Invalid combination
        ("compv4", "cshuffle", "interwave", False),  # Invalid combination
        ("compv3", "default", "intrawave", True),
        ("compv4", "cshuffle", "intrawave", True),
    ]

    print("\nTesting trait combinations:")
    for pipeline, epilogue, scheduler, expected in test_cases:
        valid = is_trait_combination_valid(pipeline, epilogue, scheduler)
        status = "PASS" if valid == expected else "FAIL"
        print(f"  {pipeline}-{epilogue}-{scheduler}: {valid} - {status}")


def test_full_tile_config_validation():
    """Test full tile configuration validation"""
    print("\n\nTesting full tile configuration validation...")

    # Test case that was failing in the build
    tile_m, tile_n, tile_k = 256, 256, 32
    warp_m, warp_n, warp_k = 1, 4, 1
    warp_tile_m, warp_tile_n, warp_tile_k = 4, 64, 8

    print("\nTesting problematic configuration:")
    print(f"  Tile: {tile_m}x{tile_n}x{tile_k}")
    print(f"  Warp: {warp_m}x{warp_n}x{warp_k}")
    print(f"  WarpTile: {warp_tile_m}x{warp_tile_n}x{warp_tile_k}")

    valid = is_tile_config_valid(
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        "fp16",
        "fp16",
        "fp16",
        "mem",
    )

    print(f"  Valid: {valid}")
    print("  Expected: False (warp tile [4, 64, 8] is not supported for fp16)")

    # Test a valid configuration
    warp_tile_k = 16  # Change to valid value
    print("\nTesting corrected configuration:")
    print(f"  WarpTile: {warp_tile_m}x{warp_tile_n}x{warp_tile_k}")

    valid = is_tile_config_valid(
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        "fp16",
        "fp16",
        "fp16",
        "mem",
    )

    print(f"  Valid: {valid}")
    print("  Expected: True")


def main():
    """Run all tests"""
    print("=" * 60)
    print("GEMM Validation Test Suite")
    print("=" * 60)

    test_warp_tile_validation()
    test_trait_combinations()
    test_full_tile_config_validation()

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
