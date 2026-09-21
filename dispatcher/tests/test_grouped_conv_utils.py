#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
TDD tests for python/grouped_conv_utils.py -- grouped convolution Python utilities.

Phase 1 TDD: tests written BEFORE implementation exists.
Run: python3 -m pytest tests/test_grouped_conv_utils.py -v
"""

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from dispatcher_common import ValidationResultBase  # noqa: E402
from grouped_conv_utils import (  # noqa: E402
    GroupedConvValidationResult,
    validate_grouped_conv_config,
    auto_correct_grouped_conv_config,
    get_grouped_conv_default_config,
    GroupedConvDataType,
    format_grouped_conv_summary,
)


# =============================================================================
# VALID CONFIG FIXTURES
# =============================================================================


def make_valid_grouped_conv_config():
    """Return a valid grouped conv config dict for gfx942."""
    return {
        "tile_config": {
            "tile_k": 128,
            "tile_c": 128,
            "wave_m": 2,
            "wave_n": 2,
            "wave_k": 1,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 16,
        },
        "trait_config": {
            "pipeline": "compv4",
            "epilogue": "cshuffle",
            "scheduler": "intrawave",
        },
        "variant": "2d_fwd",
        "ndim_spatial": 2,
        "arch": "gfx942",
        "layout": "nhwgc",
        "dtype": "fp16",
    }


# =============================================================================
# TestGroupedConvValidationResult
# =============================================================================


class TestGroupedConvValidationResult(unittest.TestCase):
    """Tests for GroupedConvValidationResult dataclass."""

    def test_inherits_from_validation_result_base(self):
        """GroupedConvValidationResult should inherit from ValidationResultBase."""
        self.assertTrue(
            issubclass(GroupedConvValidationResult, ValidationResultBase),
            "GroupedConvValidationResult must inherit from ValidationResultBase",
        )

    def test_valid_result_has_is_valid(self):
        """Valid result has is_valid=True."""
        vr = GroupedConvValidationResult(is_valid=True)
        self.assertTrue(vr.is_valid)

    def test_invalid_result_has_is_valid_false(self):
        """Invalid result has is_valid=False."""
        vr = GroupedConvValidationResult(is_valid=False, errors=["bad config"])
        self.assertFalse(vr.is_valid)

    def test_has_errors_list(self):
        """Result has errors list."""
        vr = GroupedConvValidationResult(
            is_valid=False,
            errors=["invalid wave", "invalid trait"],
        )
        self.assertEqual(len(vr.errors), 2)
        self.assertIn("invalid wave", vr.errors)
        self.assertIn("invalid trait", vr.errors)

    def test_has_warnings_list(self):
        """Result has warnings list."""
        vr = GroupedConvValidationResult(
            is_valid=True,
            warnings=["deprecated option"],
        )
        self.assertEqual(len(vr.warnings), 1)
        self.assertIn("deprecated option", vr.warnings)

    def test_has_suggested_fixes_dict(self):
        """Result has suggested_fixes dict."""
        vr = GroupedConvValidationResult(
            is_valid=False,
            suggested_fixes={"wave_m": 2, "wave_n": 2},
        )
        self.assertIn("wave_m", vr.suggested_fixes)
        self.assertEqual(vr.suggested_fixes["wave_m"], 2)
        self.assertIn("wave_n", vr.suggested_fixes)
        self.assertEqual(vr.suggested_fixes["wave_n"], 2)

    def test_default_empty_errors_warnings_fixes(self):
        """Default result has empty errors, warnings, suggested_fixes."""
        vr = GroupedConvValidationResult(is_valid=True)
        self.assertEqual(vr.errors, [])
        self.assertEqual(vr.warnings, [])
        self.assertEqual(vr.suggested_fixes, {})


# =============================================================================
# TestValidateGroupedConvConfig
# =============================================================================


class TestValidateGroupedConvConfig(unittest.TestCase):
    """Tests for validate_grouped_conv_config."""

    def test_valid_config_passes(self):
        """Valid config should pass validation."""
        config = make_valid_grouped_conv_config()
        result = validate_grouped_conv_config(config)
        self.assertTrue(result.is_valid, f"Expected valid, got errors: {result.errors}")
        self.assertEqual(result.errors, [])

    def test_invalid_wave_config_fails(self):
        """Invalid wave config should fail validation."""
        config = make_valid_grouped_conv_config()
        config["tile_config"]["wave_m"] = 3
        config["tile_config"]["wave_n"] = 3
        result = validate_grouped_conv_config(config)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        error_str = " ".join(result.errors).lower()
        self.assertIn("wave", error_str)

    def test_invalid_trait_fails(self):
        """Invalid trait combination should fail validation."""
        config = make_valid_grouped_conv_config()
        config["trait_config"]["pipeline"] = "compv4"
        config["trait_config"]["epilogue"] = "cshuffle"
        config["trait_config"]["scheduler"] = "interwave"  # Invalid combo
        result = validate_grouped_conv_config(config)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        error_str = " ".join(result.errors).lower()
        self.assertIn("trait", error_str)

    def test_missing_fields_fails(self):
        """Config with missing required fields should fail validation."""
        config = {"arch": "gfx942"}  # Missing tile_config, trait_config, etc.
        result = validate_grouped_conv_config(config)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)


# =============================================================================
# TestAutoCorrectGroupedConvConfig
# =============================================================================


class TestAutoCorrectGroupedConvConfig(unittest.TestCase):
    """Tests for auto_correct_grouped_conv_config."""

    def test_invalid_wave_gets_corrected(self):
        """Invalid wave config should be auto-corrected."""
        config = make_valid_grouped_conv_config()
        config["tile_config"]["wave_m"] = 3
        config["tile_config"]["wave_n"] = 3
        corrected, result = auto_correct_grouped_conv_config(config)
        self.assertIsInstance(corrected, dict)
        self.assertIsInstance(result, GroupedConvValidationResult)
        # Corrected wave should be valid for arch
        wave_m = corrected.get("tile_config", {}).get("wave_m")
        wave_n = corrected.get("tile_config", {}).get("wave_n")
        self.assertIn(wave_m, [1, 2, 4])
        self.assertIn(wave_n, [1, 2, 4])

    def test_invalid_trait_gets_corrected(self):
        """Invalid trait combination should be auto-corrected."""
        config = make_valid_grouped_conv_config()
        config["trait_config"]["scheduler"] = "interwave"
        config["trait_config"]["pipeline"] = "compv4"
        config["trait_config"]["epilogue"] = "cshuffle"
        corrected, result = auto_correct_grouped_conv_config(config)
        self.assertIsInstance(corrected, dict)
        self.assertIsInstance(result, GroupedConvValidationResult)
        # Scheduler should be corrected to intrawave for compv4+cshuffle
        scheduler = corrected.get("trait_config", {}).get("scheduler")
        self.assertEqual(scheduler, "intrawave")


# =============================================================================
# TestGetGroupedConvDefaultConfig
# =============================================================================


class TestGetGroupedConvDefaultConfig(unittest.TestCase):
    """Tests for get_grouped_conv_default_config."""

    def test_returns_config(self):
        """Should return a GroupedConvKernelConfig (or dict via to_dict)."""
        config = get_grouped_conv_default_config("2d_fwd")
        # Accepts both dataclass and dict
        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertIsInstance(d, dict)

    def test_has_tile_config(self):
        """Returned config has tile_config key."""
        config = get_grouped_conv_default_config("2d_fwd")
        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertIn("tile_config", d)
        self.assertIsInstance(d["tile_config"], dict)

    def test_has_trait_config(self):
        """Returned config has trait_config key."""
        config = get_grouped_conv_default_config("2d_fwd")
        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertIn("trait_config", d)
        self.assertIsInstance(d["trait_config"], dict)

    def test_has_variant(self):
        """Returned config has variant."""
        config = get_grouped_conv_default_config("2d_fwd")
        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertIn("variant", d)

    def test_has_ndim_spatial(self):
        """Returned config has ndim_spatial."""
        config = get_grouped_conv_default_config("2d_fwd")
        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertIn("ndim_spatial", d)

    def test_has_arch(self):
        """Returned config has arch."""
        config = get_grouped_conv_default_config("2d_fwd")
        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertIn("arch", d)

    def test_has_layout(self):
        """Returned config has layout."""
        config = get_grouped_conv_default_config("2d_fwd")
        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertIn("layout", d)


# =============================================================================
# TestGroupedConvDataType
# =============================================================================


class TestGroupedConvDataType(unittest.TestCase):
    """Tests for GroupedConvDataType enum."""

    def test_fp16_exists(self):
        """GroupedConvDataType has FP16."""
        self.assertIsNotNone(GroupedConvDataType.FP16)

    def test_bf16_exists(self):
        """GroupedConvDataType has BF16."""
        self.assertIsNotNone(GroupedConvDataType.BF16)

    def test_fp32_exists(self):
        """GroupedConvDataType has FP32."""
        self.assertIsNotNone(GroupedConvDataType.FP32)

    def test_fp8_exists(self):
        """GroupedConvDataType has FP8."""
        self.assertIsNotNone(GroupedConvDataType.FP8)

    def test_bf8_exists(self):
        """GroupedConvDataType has BF8."""
        self.assertIsNotNone(GroupedConvDataType.BF8)

    def test_int8_exists(self):
        """GroupedConvDataType has INT8."""
        self.assertIsNotNone(GroupedConvDataType.INT8)

    def test_enum_values_unique(self):
        """All enum values should be unique."""
        values = [
            GroupedConvDataType.FP16,
            GroupedConvDataType.BF16,
            GroupedConvDataType.FP32,
            GroupedConvDataType.FP8,
            GroupedConvDataType.BF8,
            GroupedConvDataType.INT8,
        ]
        self.assertEqual(len(values), len(set(values)))


# =============================================================================
# TestFormatGroupedConvSummary
# =============================================================================


class TestFormatGroupedConvSummary(unittest.TestCase):
    """Tests for format_grouped_conv_summary."""

    def test_returns_non_empty_string(self):
        """Should return a non-empty string."""
        config = make_valid_grouped_conv_config()
        summary = format_grouped_conv_summary(config)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_contains_key_info(self):
        """Summary should contain key config info (variant, arch, layout, dtype)."""
        config = make_valid_grouped_conv_config()
        summary = format_grouped_conv_summary(config)
        # Should mention at least some of: variant, arch, layout, dtype
        summary_lower = summary.lower()
        has_key_info = (
            "2d" in summary_lower
            or "fwd" in summary_lower
            or "gfx" in summary_lower
            or "nhwgc" in summary_lower
            or "fp16" in summary_lower
        )
        self.assertTrue(
            has_key_info,
            f"Summary should contain key info, got: {summary}",
        )

    def test_empty_config_returns_something(self):
        """Empty or minimal config should still return a string."""
        summary = format_grouped_conv_summary({})
        self.assertIsInstance(summary, str)
        self.assertGreaterEqual(len(summary), 0)


if __name__ == "__main__":
    unittest.main()
