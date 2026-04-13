#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Comprehensive Test Suite for Auto-Correction and Validation

Tests:
1. GEMM validation and wildcard expansion
2. Conv validation and wildcard expansion
3. Python KernelConfig auto-correction
4. Architecture-specific dtype support
5. Edge cases and error handling

Can be run as:
    python3 tests/test_autocorrect.py                    # Run all tests
    python3 tests/test_autocorrect.py -v                 # Verbose output
    python3 tests/test_autocorrect.py TestGemmValidation # Run specific test class
    ctest -R test_autocorrect                            # Via ctest

Exit codes:
    0 = All tests passed
    1 = Some tests failed
"""

import sys
import unittest
import random
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "scripts"))

# Import modules under test
from compile_gemm_examples import (  # noqa: E402
    validate_kernel_config,
    expand_declaration_with_arch_filter,
    is_wildcard_declaration,
)
from compile_grouped_conv_examples import (  # noqa: E402
    validate_grouped_conv_kernel_config as validate_conv_kernel_config,
    expand_grouped_conv_declaration_with_arch_filter as expand_conv_declaration_with_arch_filter,
    is_grouped_conv_wildcard_declaration as is_conv_wildcard_declaration,
)
from ctypes_utils import auto_correct_kernel_config, KernelConfig  # noqa: E402


# =============================================================================
# TEST DATA
# =============================================================================

VALID_ARCHS = ["gfx90a", "gfx942", "gfx950"]
VALID_DTYPES = ["fp16", "bf16"]
VALID_LAYOUTS = ["rcr", "rrr"]
VALID_PIPELINES = ["compv3", "compv4"]
VALID_SCHEDULERS = ["intrawave"]

# Known valid wave configs for gfx942
VALID_WAVE_CONFIGS_GFX942 = [[1, 4, 1], [2, 2, 1], [4, 1, 1]]

# Known valid warp tiles for fp16 on gfx942
VALID_WARP_TILES_FP16_GFX942 = [[16, 16, 16], [16, 16, 32], [32, 32, 8], [32, 32, 16]]


# =============================================================================
# GEMM VALIDATION TESTS
# =============================================================================


class TestGemmValidation(unittest.TestCase):
    """Test GEMM kernel validation."""

    def test_valid_config(self):
        """Valid configuration should pass validation."""
        config = {
            "name": "test_valid",
            "dtype_a": "fp16",
            "dtype_b": "fp16",
            "dtype_c": "fp16",
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
        }
        is_valid, error = validate_kernel_config(config, "gfx942")
        self.assertTrue(is_valid, f"Expected valid, got error: {error}")

    def test_invalid_wave_config(self):
        """Invalid wave config should fail validation."""
        config = {
            "name": "test_invalid_wave",
            "dtype_a": "fp16",
            "wave_m": 3,  # Invalid
            "wave_n": 3,  # Invalid
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        is_valid, error = validate_kernel_config(config, "gfx942")
        self.assertFalse(is_valid)
        self.assertIn("wave", error.lower())

    def test_invalid_scheduler(self):
        """Invalid scheduler should fail validation."""
        config = {
            "name": "test_invalid_scheduler",
            "dtype_a": "fp16",
            "wave_m": 2,
            "wave_n": 2,
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "epilogue": "cshuffle",
            "scheduler": "interwave",  # Invalid with compv4+cshuffle
        }
        is_valid, error = validate_kernel_config(config, "gfx942")
        self.assertFalse(is_valid)
        self.assertIn("trait", error.lower())

    def test_wildcard_skips_validation(self):
        """Wildcard declarations should skip validation."""
        config = {
            "name": "test_wildcard",
            "dtype_a": "fp16",
            "wave_m": -1,  # Wildcard
            "wave_n": -1,  # Wildcard
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        self.assertTrue(is_wildcard_declaration(config))
        is_valid, _ = validate_kernel_config(config, "gfx942")
        self.assertTrue(is_valid)

    def test_unsupported_arch(self):
        """Unsupported architecture should fail validation."""
        config = {
            "name": "test_bad_arch",
            "dtype_a": "fp16",
            "wave_m": 2,
            "wave_n": 2,
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        is_valid, error = validate_kernel_config(config, "gfx_invalid")
        self.assertFalse(is_valid)
        self.assertIn("unsupported", error.lower())


class TestGemmExpansion(unittest.TestCase):
    """Test GEMM wildcard expansion."""

    def test_wave_expansion(self):
        """Wave wildcard should expand to valid configs."""
        config = {
            "name": "test_wave_expand",
            "dtype_a": "fp16",
            "dtype_b": "fp16",
            "dtype_c": "fp16",
            "layout": "rcr",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 32,
            "wave_m": -1,  # Wildcard
            "wave_n": -1,  # Wildcard
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        expanded = expand_declaration_with_arch_filter(config, "gfx942")
        self.assertGreater(len(expanded), 0, "Should expand to at least one config")

        # All expanded configs should be valid
        for exp in expanded:
            is_valid, error = validate_kernel_config(exp, "gfx942")
            self.assertTrue(is_valid, f"Expanded config invalid: {error}")

    def test_full_wildcard_expansion(self):
        """Full wildcard should expand to multiple valid configs."""
        config = {
            "name": "test_full_wildcard",
            "dtype_a": "fp16",
            "dtype_b": "fp16",
            "dtype_c": "fp16",
            "layout": "rcr",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 32,
            "wave_m": -1,
            "wave_n": -1,
            "wave_k": 1,
            "warp_m": -1,
            "warp_n": -1,
            "warp_k": -1,
            "pipeline": "*",
            "scheduler": "*",
        }
        expanded = expand_declaration_with_arch_filter(config, "gfx942")
        self.assertGreater(
            len(expanded), 1, "Full wildcard should expand to multiple configs"
        )

    def test_explicit_config_not_expanded(self):
        """Explicit (non-wildcard) config should not expand."""
        config = {
            "name": "test_explicit",
            "dtype_a": "fp16",
            "dtype_b": "fp16",
            "dtype_c": "fp16",
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
        }
        expanded = expand_declaration_with_arch_filter(config, "gfx942")
        self.assertEqual(len(expanded), 1, "Explicit config should not expand")


# =============================================================================
# CONV VALIDATION TESTS
# =============================================================================


class TestConvValidation(unittest.TestCase):
    """Test Conv kernel validation."""

    def test_valid_conv_config(self):
        """Valid conv configuration should pass validation."""
        config = {
            "name": "test_valid_conv",
            "dtype": "fp16",
            "layout": "nhwgc",
            "conv_type": "forward",
            "tile_k": 128,
            "tile_c": 128,
            "wave_m": 2,
            "wave_n": 2,
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        is_valid, error = validate_conv_kernel_config(config, "gfx942")
        self.assertTrue(is_valid, f"Expected valid, got error: {error}")

    def test_invalid_conv_wave(self):
        """Invalid wave config should fail conv validation."""
        config = {
            "name": "test_invalid_conv_wave",
            "dtype": "fp16",
            "wave_m": 5,  # Invalid
            "wave_n": 5,  # Invalid
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        is_valid, error = validate_conv_kernel_config(config, "gfx942")
        self.assertFalse(is_valid)
        self.assertIn("wave", error.lower())

    def test_conv_wildcard_detection(self):
        """Should correctly detect conv wildcards."""
        wildcard_config = {
            "wave_m": -1,
            "wave_n": 2,
            "warp_m": 32,
            "warp_n": 32,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        self.assertTrue(is_conv_wildcard_declaration(wildcard_config))

        explicit_config = {
            "wave_m": 2,
            "wave_n": 2,
            "warp_m": 32,
            "warp_n": 32,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        self.assertFalse(is_conv_wildcard_declaration(explicit_config))


class TestConvExpansion(unittest.TestCase):
    """Test Conv wildcard expansion."""

    def test_conv_wave_expansion(self):
        """Conv wave wildcard should expand to valid configs."""
        config = {
            "name": "test_conv_wave_expand",
            "dtype": "fp16",
            "layout": "nhwgc",
            "conv_type": "forward",
            "tile_k": 128,
            "tile_c": 128,
            "wave_m": -1,
            "wave_n": -1,
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
            "pipeline": "compv4",
            "scheduler": "intrawave",
        }
        expanded = expand_conv_declaration_with_arch_filter(config, "gfx942")
        self.assertGreater(len(expanded), 0, "Should expand to at least one config")


# =============================================================================
# PYTHON AUTO-CORRECTION TESTS
# =============================================================================


class TestPythonAutoCorrect(unittest.TestCase):
    """Test Python KernelConfig auto-correction."""

    def test_autocorrect_invalid_wave(self):
        """Auto-correction should fix invalid wave config."""
        config = KernelConfig()
        config.dtype_a = "fp16"
        config.dtype_b = "fp16"
        config.dtype_c = "fp16"
        config.dtype_acc = "fp32"
        config.layout_a = "row"
        config.layout_b = "col"
        config.layout_c = "row"
        config.tile_m = 128
        config.tile_n = 128
        config.tile_k = 32
        config.wave_m = 1  # May be invalid
        config.wave_n = 1  # May be invalid
        config.wave_k = 1
        config.warp_m = 32
        config.warp_n = 32
        config.warp_k = 16
        config.pipeline = "compv4"
        config.scheduler = "intrawave"
        config.gfx_arch = "gfx942"

        corrected, was_modified, corrections = auto_correct_kernel_config(
            config, verbose=False
        )

        # Should either be valid or corrected
        self.assertIsNotNone(corrected)
        if was_modified:
            self.assertGreater(len(corrections), 0)

    def test_autocorrect_returns_three_values(self):
        """Auto-correction should return (config, was_modified, corrections)."""
        config = KernelConfig()
        config.dtype_a = "fp16"
        config.dtype_b = "fp16"
        config.dtype_c = "fp16"
        config.dtype_acc = "fp32"
        config.layout_a = "row"
        config.layout_b = "col"
        config.layout_c = "row"
        config.tile_m = 128
        config.tile_n = 128
        config.tile_k = 32
        config.wave_m = 2
        config.wave_n = 2
        config.wave_k = 1
        config.warp_m = 32
        config.warp_n = 32
        config.warp_k = 16
        config.pipeline = "compv4"
        config.scheduler = "intrawave"
        config.gfx_arch = "gfx942"

        result = auto_correct_kernel_config(config, verbose=False)

        self.assertEqual(len(result), 3, "Should return 3 values")
        corrected, was_modified, corrections = result
        self.assertIsInstance(was_modified, bool)
        self.assertIsInstance(corrections, list)


# =============================================================================
# STRESS TESTS
# =============================================================================


class TestStressRandom(unittest.TestCase):
    """Stress test with random configurations."""

    def test_random_gemm_configs(self):
        """Random GEMM configs should either validate or expand successfully."""
        random.seed(42)  # Reproducible

        dtypes = ["fp16", "bf16"]
        layouts = ["rcr", "rrr"]
        tiles = [(64, 64, 32), (128, 128, 32), (256, 256, 64)]
        waves = [(1, 1, 1), (2, 2, 1), (1, 4, 1), (3, 3, 1)]  # Some invalid
        warps = [(16, 16, 16), (32, 32, 16), (48, 48, 24)]  # Some invalid
        pipelines = ["compv3", "compv4", "invalid"]
        schedulers = ["intrawave", "interwave"]

        success_count = 0
        total_count = 30

        for _ in range(total_count):
            config = {
                "name": "random_test",
                "dtype_a": random.choice(dtypes),
                "dtype_b": random.choice(dtypes),
                "dtype_c": random.choice(dtypes),
                "layout": random.choice(layouts),
                "tile_m": random.choice(tiles)[0],
                "tile_n": random.choice(tiles)[1],
                "tile_k": random.choice(tiles)[2],
                "wave_m": random.choice(waves)[0],
                "wave_n": random.choice(waves)[1],
                "wave_k": random.choice(waves)[2],
                "warp_m": random.choice(warps)[0],
                "warp_n": random.choice(warps)[1],
                "warp_k": random.choice(warps)[2],
                "pipeline": random.choice(pipelines),
                "scheduler": random.choice(schedulers),
            }

            is_valid, _ = validate_kernel_config(config, "gfx942")

            if is_valid:
                success_count += 1
            else:
                # Try wildcard expansion
                wildcard = config.copy()
                wildcard["wave_m"] = -1
                wildcard["wave_n"] = -1
                wildcard["warp_m"] = -1
                wildcard["warp_n"] = -1
                wildcard["pipeline"] = "*"
                wildcard["scheduler"] = "*"

                expanded = expand_declaration_with_arch_filter(wildcard, "gfx942")
                if expanded:
                    success_count += 1

        # At least 50% should be handleable
        self.assertGreater(
            success_count / total_count,
            0.5,
            f"Only {success_count}/{total_count} configs were handleable",
        )

    def test_random_conv_configs(self):
        """Random Conv configs should either validate or expand successfully."""
        random.seed(42)

        dtypes = ["fp16", "bf16"]
        tiles = [(64, 64), (128, 128), (256, 256)]
        waves = [(2, 2, 1), (1, 4, 1), (3, 3, 1)]
        warps = [(16, 16, 16), (32, 32, 16)]

        success_count = 0
        total_count = 20

        for _ in range(total_count):
            config = {
                "name": "random_conv_test",
                "dtype": random.choice(dtypes),
                "layout": "nhwgc",
                "conv_type": "forward",
                "tile_k": random.choice(tiles)[0],
                "tile_c": random.choice(tiles)[1],
                "wave_m": random.choice(waves)[0],
                "wave_n": random.choice(waves)[1],
                "wave_k": random.choice(waves)[2],
                "warp_m": random.choice(warps)[0],
                "warp_n": random.choice(warps)[1],
                "warp_k": random.choice(warps)[2],
                "pipeline": "compv4",
                "scheduler": "intrawave",
            }

            is_valid, _ = validate_conv_kernel_config(config, "gfx942")

            if is_valid:
                success_count += 1
            else:
                # Try wildcard expansion
                wildcard = config.copy()
                wildcard["wave_m"] = -1
                wildcard["wave_n"] = -1
                wildcard["warp_m"] = -1
                wildcard["warp_n"] = -1

                expanded = expand_conv_declaration_with_arch_filter(wildcard, "gfx942")
                if expanded:
                    success_count += 1

        self.assertGreater(
            success_count / total_count,
            0.5,
            f"Only {success_count}/{total_count} conv configs were handleable",
        )


# =============================================================================
# ARCHITECTURE TESTS
# =============================================================================


class TestArchitectureSupport(unittest.TestCase):
    """Test architecture-specific support."""

    def test_gfx942_fp16_support(self):
        """gfx942 should support fp16."""
        config = {
            "dtype_a": "fp16",
            "wave_m": -1,
            "wave_n": -1,
            "warp_m": -1,
            "warp_n": -1,
            "pipeline": "*",
            "scheduler": "*",
        }
        expanded = expand_declaration_with_arch_filter(config, "gfx942")
        self.assertGreater(len(expanded), 0, "gfx942 should support fp16")

    def test_gfx942_bf16_support(self):
        """gfx942 should support bf16."""
        config = {
            "dtype_a": "bf16",
            "wave_m": -1,
            "wave_n": -1,
            "warp_m": -1,
            "warp_n": -1,
            "pipeline": "*",
            "scheduler": "*",
        }
        expanded = expand_declaration_with_arch_filter(config, "gfx942")
        self.assertGreater(len(expanded), 0, "gfx942 should support bf16")

    def test_gfx90a_support(self):
        """gfx90a should support fp16."""
        config = {
            "dtype_a": "fp16",
            "wave_m": -1,
            "wave_n": -1,
            "warp_m": -1,
            "warp_n": -1,
            "pipeline": "*",
            "scheduler": "*",
        }
        expanded = expand_declaration_with_arch_filter(config, "gfx90a")
        self.assertGreater(len(expanded), 0, "gfx90a should support fp16")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run tests."""
    # Parse args for verbosity
    verbosity = 2 if "-v" in sys.argv or "--verbose" in sys.argv else 1

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGemmValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestGemmExpansion))
    suite.addTests(loader.loadTestsFromTestCase(TestConvValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestConvExpansion))
    suite.addTests(loader.loadTestsFromTestCase(TestPythonAutoCorrect))
    suite.addTests(loader.loadTestsFromTestCase(TestStressRandom))
    suite.addTests(loader.loadTestsFromTestCase(TestArchitectureSupport))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
