#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for depthwise convolution rules in codegen/tile_math.py.

Ground truth: all 20 profiler depthwise configs extracted from
configs/grouped_conv/forward/profiler/ngchw_fp16.json (identical
across fp16/bf16/fp32).

Run:
    cd projects/composablekernel/dispatcher
    python3 -m unittest tests/test_depthwise_tile_math.py -v
"""

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from grouped_conv.tile_math import (  # noqa: E402
    DepthwiseConfig,
    is_valid_depthwise_config,
    get_valid_depthwise_configs,
)

# =============================================================================
# Reference configs from JSON ground truth
# =============================================================================
# Tuple order: (tile_h, tile_w, filt, str_h, str_w, pad_h, pad_w,
#               nbatch, sub_h, sub_w, in_vec, out_vec)

DEPTHWISE_PROFILER_CONFIGS = [
    (8, 8, 3, 1, 1, 1, 1, 8, 2, 2, 2, 2),
    (16, 16, 3, 1, 1, 1, 1, 8, 1, 4, 8, 8),
    (16, 16, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2),
    (28, 28, 3, 1, 1, 1, 1, 1, 4, 4, 8, 8),
    (32, 32, 3, 1, 1, 1, 1, 1, 4, 4, 8, 8),
    (16, 16, 3, 2, 2, 1, 1, 2, 1, 4, 8, 8),
    (16, 16, 3, 2, 2, 1, 1, 1, 1, 4, 8, 8),
    (16, 16, 3, 2, 2, 1, 1, 1, 2, 2, 8, 8),
    (16, 16, 3, 2, 2, 1, 1, 1, 2, 2, 2, 2),
    (14, 28, 3, 2, 2, 1, 1, 1, 2, 4, 8, 8),
    (32, 32, 3, 2, 2, 1, 1, 2, 4, 4, 8, 8),
    (32, 32, 3, 2, 2, 1, 1, 1, 4, 4, 4, 4),
    (32, 32, 3, 2, 2, 1, 1, 1, 4, 4, 8, 8),
    (32, 32, 3, 2, 2, 1, 1, 1, 2, 8, 8, 8),
    (8, 8, 5, 1, 1, 2, 2, 1, 1, 1, 1, 1),
    (8, 8, 5, 1, 1, 2, 2, 8, 2, 2, 2, 2),
    (16, 16, 5, 1, 1, 2, 2, 1, 1, 4, 8, 8),
    (16, 16, 5, 1, 1, 2, 2, 8, 1, 4, 8, 8),
    (28, 28, 5, 1, 1, 2, 2, 8, 4, 4, 8, 8),
    (32, 32, 5, 1, 1, 2, 2, 4, 4, 4, 8, 8),
]

DEPTHWISE_TEST_CONFIGS = [
    (8, 8, 3, 1, 1, 1, 1, 8, 2, 2, 2, 2),
    (32, 32, 3, 1, 1, 1, 1, 1, 4, 4, 8, 8),
    (16, 16, 3, 2, 2, 1, 1, 2, 1, 4, 8, 8),
    (32, 32, 3, 2, 2, 1, 1, 1, 2, 8, 8, 8),
    (8, 8, 5, 1, 1, 2, 2, 1, 1, 1, 1, 1),
    (32, 32, 5, 1, 1, 2, 2, 4, 4, 4, 8, 8),
]

# Tile/filter/stride space matching grouped_config_rules_default.py
TILE_SIZES = [(8, 8), (14, 28), (16, 16), (28, 28), (32, 32)]
FILTER_SIZES = [3, 5]
STRIDES = [(1, 1), (2, 2)]


def _tuple_to_cfg(t):
    """Convert a 12-tuple to a DepthwiseConfig."""
    return DepthwiseConfig(*t)


def _cfg_to_tuple(c):
    """Convert a DepthwiseConfig to a 12-tuple."""
    return (c.tile_h, c.tile_w, c.filt, c.str_h, c.str_w,
            c.pad_h, c.pad_w, c.nbatch, c.sub_h, c.sub_w,
            c.in_vec, c.out_vec)


# =============================================================================
# Tests
# =============================================================================

class TestIsValidDepthwiseConfig(unittest.TestCase):
    """Tests for is_valid_depthwise_config()."""

    def test_all_reference_configs_valid(self):
        """Every JSON reference config must pass validation."""
        for t in DEPTHWISE_PROFILER_CONFIGS:
            cfg = _tuple_to_cfg(t)
            self.assertTrue(
                is_valid_depthwise_config(cfg),
                f"Reference config should be valid: {t}",
            )

    def test_all_reference_configs_valid_fp32(self):
        """Reference configs must also be valid with fp32 dtype_size=4."""
        for t in DEPTHWISE_PROFILER_CONFIGS:
            cfg = _tuple_to_cfg(t)
            self.assertTrue(is_valid_depthwise_config(cfg, dtype_size=4))

    def test_odd_filter_required(self):
        """Even filter size must be rejected."""
        cfg = DepthwiseConfig(8, 8, 4, 1, 1, 1, 1, 8, 2, 2, 2, 2)
        self.assertFalse(is_valid_depthwise_config(cfg))

    def test_pad_w_must_be_positive(self):
        """PadW=0 must be rejected."""
        cfg = DepthwiseConfig(8, 8, 3, 1, 1, 1, 0, 8, 2, 2, 2, 2)
        self.assertFalse(is_valid_depthwise_config(cfg))

    def test_subtile_exceeds_tile(self):
        """sub_h > tile_h must be rejected."""
        cfg = DepthwiseConfig(8, 8, 3, 1, 1, 1, 1, 8, 16, 2, 2, 2)
        self.assertFalse(is_valid_depthwise_config(cfg))

    def test_total_subtiles_exceeds_64(self):
        """Config with too many subtiles must be rejected."""
        # sub_h=1, sub_w=1 on tile 16x16 → 256 subtiles > 64
        cfg = DepthwiseConfig(16, 16, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.assertFalse(is_valid_depthwise_config(cfg))

    def test_nbatch_divisibility(self):
        """nbatch must be divisible by tile_per_wave."""
        # tile_h=8, tile_w=8, sub_h=2, sub_w=2 → 4×4=16 subtiles
        # tile_per_wave = 64//16 = 4, so nbatch=3 is invalid
        cfg = DepthwiseConfig(8, 8, 3, 1, 1, 1, 1, 3, 2, 2, 2, 2)
        self.assertFalse(is_valid_depthwise_config(cfg))

    def test_vec_not_power_of_2(self):
        """Non-power-of-2 vector size must be rejected."""
        cfg = DepthwiseConfig(8, 8, 3, 1, 1, 1, 1, 8, 2, 2, 3, 2)
        self.assertFalse(is_valid_depthwise_config(cfg))

    def test_stride_w_constraint(self):
        """Odd StrideW != 1 must be rejected."""
        cfg = DepthwiseConfig(8, 8, 3, 1, 3, 1, 1, 8, 2, 2, 2, 2)
        self.assertFalse(is_valid_depthwise_config(cfg))


class TestGetValidDepthwiseConfigs(unittest.TestCase):
    """Tests for get_valid_depthwise_configs()."""

    @classmethod
    def setUpClass(cls):
        """Generate configs once for all tests."""
        cls.generated = get_valid_depthwise_configs(
            TILE_SIZES, FILTER_SIZES, STRIDES,
        )
        cls.generated_set = {_cfg_to_tuple(c) for c in cls.generated}

    def test_no_false_negatives(self):
        """Every JSON profiler reference config must be generated by rules."""
        missing = []
        for ref in DEPTHWISE_PROFILER_CONFIGS:
            if ref not in self.generated_set:
                missing.append(ref)
        if missing:
            lines = [f"  {m}" for m in missing]
            self.fail(
                f"{len(missing)}/20 profiler configs missing:\n"
                + "\n".join(lines)
            )

    def test_test_configs_are_subset(self):
        """Test configs must also be in the generated set."""
        for ref in DEPTHWISE_TEST_CONFIGS:
            self.assertIn(ref, self.generated_set,
                          f"Test config missing: {ref}")

    def test_all_generated_are_valid(self):
        """Every generated config must pass all constraint checks."""
        invalid = []
        for cfg in self.generated:
            if not is_valid_depthwise_config(cfg):
                invalid.append(_cfg_to_tuple(cfg))
        if invalid:
            self.fail(f"{len(invalid)} generated configs are invalid")

    def test_no_duplicates(self):
        """Generated list must not contain duplicates."""
        self.assertEqual(
            len(self.generated), len(self.generated_set),
            "Duplicate configs found in generated list",
        )

    def test_generates_nonzero_configs(self):
        """Must generate at least the 20 reference configs."""
        self.assertGreaterEqual(len(self.generated), 20)

    def test_coverage_rate(self):
        """Log coverage statistics (informational)."""
        n_ref = len(DEPTHWISE_PROFILER_CONFIGS)
        n_covered = sum(1 for r in DEPTHWISE_PROFILER_CONFIGS
                        if r in self.generated_set)
        n_total = len(self.generated)
        print(f"\n[depthwise coverage] {n_covered}/{n_ref} reference covered, "
              f"{n_total} total generated")
        self.assertEqual(n_covered, n_ref,
                         f"Coverage {n_covered}/{n_ref} < 100%")

    def test_empty_input(self):
        """Empty tile_sizes should return no configs."""
        self.assertEqual(
            get_valid_depthwise_configs([], [3], [(1, 1)]),
            [],
        )

    def test_smem_constraint_rejects_huge_tile_fp32(self):
        """Very large tiles with fp32 should produce fewer configs due to SmemSize."""
        # fp32 (dtype_size=4) doubles SmemSize, pruning more configs
        fp16 = get_valid_depthwise_configs([(32, 32)], [5], [(1, 1)], dtype_size=2)
        fp32 = get_valid_depthwise_configs([(32, 32)], [5], [(1, 1)], dtype_size=4)
        self.assertLess(len(fp32), len(fp16),
                        "fp32 should produce fewer valid configs than fp16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
