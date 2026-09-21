#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for codegen/codegen_common.py -- shared infrastructure for GEMM and grouped conv codegen.

Phase 1a TDD: these tests are written BEFORE the implementation exists.
Run: python3 -m pytest tests/test_codegen_common.py -v
"""

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from codegen_common import (  # noqa: E402
    TileConfig,
    TraitConfigBase,
    CommonTypeMappings,
    generate_cpp_compilation_unit,
    parallel_generate,
    valid_wave_configs,
    valid_warp_configs,
    valid_trait_configs,
    needs_wave_expansion,
    needs_warp_expansion,
    needs_pipeline_expansion,
)


class TestTileConfig(unittest.TestCase):
    """TileConfig dataclass tests."""

    def test_valid_config(self):
        tc = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertTrue(tc.is_valid())

    def test_zero_tile_invalid(self):
        tc = TileConfig(0, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertFalse(tc.is_valid())

    def test_non_divisible_invalid(self):
        tc = TileConfig(127, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertFalse(tc.is_valid())

    def test_all_fields_accessible(self):
        tc = TileConfig(256, 128, 64, 4, 1, 1, 32, 32, 16)
        self.assertEqual(tc.tile_m, 256)
        self.assertEqual(tc.tile_n, 128)
        self.assertEqual(tc.tile_k, 64)
        self.assertEqual(tc.warp_m, 4)
        self.assertEqual(tc.warp_n, 1)
        self.assertEqual(tc.warp_k, 1)
        self.assertEqual(tc.warp_tile_m, 32)
        self.assertEqual(tc.warp_tile_n, 32)
        self.assertEqual(tc.warp_tile_k, 16)

    def test_small_valid_config(self):
        tc = TileConfig(16, 16, 16, 1, 1, 1, 16, 16, 16)
        self.assertTrue(tc.is_valid())


class TestTraitConfigBase(unittest.TestCase):
    """TraitConfigBase dataclass tests."""

    def test_valid_intrawave(self):
        tc = TraitConfigBase("compv3", "cshuffle", "intrawave", False, False, False)
        self.assertTrue(tc.is_valid())

    def test_invalid_interwave_compv3(self):
        tc = TraitConfigBase("compv3", "cshuffle", "interwave", False, False, False)
        self.assertFalse(tc.is_valid())

    def test_invalid_interwave_compv4(self):
        tc = TraitConfigBase("compv4", "cshuffle", "interwave", False, False, False)
        self.assertFalse(tc.is_valid())

    def test_valid_mem_interwave(self):
        tc = TraitConfigBase("mem", "cshuffle", "interwave", False, False, False)
        self.assertTrue(tc.is_valid())

    def test_valid_mem_intrawave(self):
        tc = TraitConfigBase("mem", "cshuffle", "intrawave", False, False, False)
        self.assertTrue(tc.is_valid())

    def test_padding_fields(self):
        tc = TraitConfigBase("compv3", "cshuffle", "intrawave", True, True, True)
        self.assertTrue(tc.pad_m)
        self.assertTrue(tc.pad_n)
        self.assertTrue(tc.pad_k)


class TestCommonTypeMappings(unittest.TestCase):
    """CommonTypeMappings tests."""

    def test_dtype_to_ck(self):
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["fp16"], "fp16_t")
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["bf16"], "bf16_t")
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["fp32"], "float")
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["fp8"], "fp8_t")

    def test_pipeline_to_ck(self):
        self.assertEqual(
            CommonTypeMappings.PIPELINE_TO_CK["mem"], "GemmPipelineAgBgCrMem"
        )
        self.assertIn("compv3", CommonTypeMappings.PIPELINE_TO_CK)
        self.assertIn("compv4", CommonTypeMappings.PIPELINE_TO_CK)

    def test_pipeline_to_base(self):
        self.assertIn("mem", CommonTypeMappings.PIPELINE_TO_BASE)
        self.assertIn("compv3", CommonTypeMappings.PIPELINE_TO_BASE)
        self.assertIn("compv4", CommonTypeMappings.PIPELINE_TO_BASE)

    def test_scheduler_to_ck(self):
        self.assertIn("intrawave", CommonTypeMappings.SCHEDULER_TO_CK)
        self.assertIn("interwave", CommonTypeMappings.SCHEDULER_TO_CK)

    def test_epilogue_to_dispatcher(self):
        self.assertIn("cshuffle", CommonTypeMappings.EPILOGUE_TO_DISPATCHER)
        self.assertIn("default", CommonTypeMappings.EPILOGUE_TO_DISPATCHER)

    def test_layout_to_ck(self):
        self.assertIn("r", CommonTypeMappings.LAYOUT_TO_CK)
        self.assertIn("c", CommonTypeMappings.LAYOUT_TO_CK)

    def test_get_output_dtype(self):
        self.assertEqual(CommonTypeMappings.get_output_dtype("fp8"), "fp16")
        self.assertEqual(CommonTypeMappings.get_output_dtype("bf8"), "fp16")
        self.assertEqual(CommonTypeMappings.get_output_dtype("fp16"), "fp16")
        self.assertEqual(CommonTypeMappings.get_output_dtype("fp32"), "fp32")


class TestGenerateCppCompilationUnit(unittest.TestCase):
    """Tests for generate_cpp_compilation_unit."""

    def test_includes_kernel_header(self):
        result = generate_cpp_compilation_unit("my_kernel")
        self.assertIn('#include "my_kernel.hpp"', result)

    def test_contains_pragma_once_or_guard(self):
        result = generate_cpp_compilation_unit("test_kernel")
        self.assertIn("test_kernel", result)

    def test_different_names_different_output(self):
        a = generate_cpp_compilation_unit("kernel_a")
        b = generate_cpp_compilation_unit("kernel_b")
        self.assertNotEqual(a, b)


class TestParallelGenerate(unittest.TestCase):
    """Tests for parallel_generate helper."""

    def _dummy_generate(self, item):
        return f"generated_{item}"

    def test_parallel_returns_all(self):
        items = ["a", "b", "c", "d"]
        results = parallel_generate(self._dummy_generate, items, parallel=True)
        self.assertEqual(len(results), 4)
        for item in items:
            self.assertIn(f"generated_{item}", results)

    def test_sequential_returns_all(self):
        items = ["x", "y", "z"]
        results = parallel_generate(self._dummy_generate, items, parallel=False)
        self.assertEqual(len(results), 3)
        for item in items:
            self.assertIn(f"generated_{item}", results)

    def test_empty_items(self):
        results = parallel_generate(self._dummy_generate, [], parallel=True)
        self.assertEqual(len(results), 0)

    def test_logs_per_kernel_progress(self):
        items = ["k1", "k2"]
        with self.assertLogs(level="INFO") as cm:
            parallel_generate(self._dummy_generate, items, parallel=False)
        log_output = "\n".join(cm.output)
        self.assertIn("k1", log_output)
        self.assertIn("k2", log_output)


class TestArchAwareExpansion(unittest.TestCase):
    """Tests for arch-aware expansion helpers (best-of-conv)."""

    def test_valid_wave_configs_gfx942(self):
        configs = valid_wave_configs("gfx942")
        self.assertIsInstance(configs, list)
        self.assertIn([2, 2, 1], configs)
        self.assertIn([1, 4, 1], configs)

    def test_valid_wave_configs_unknown_arch(self):
        configs = valid_wave_configs("gfx_unknown")
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)

    def test_valid_warp_configs_gfx942_fp16(self):
        configs = valid_warp_configs("gfx942", "fp16")
        self.assertIsInstance(configs, list)
        self.assertIn([32, 32, 16], configs)

    def test_valid_warp_configs_unknown_arch(self):
        configs = valid_warp_configs("gfx_unknown", "fp16")
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)

    def test_valid_trait_configs_excludes_interwave_compute(self):
        configs = valid_trait_configs()
        self.assertIsInstance(configs, list)
        self.assertNotIn(("compv3", "cshuffle", "interwave"), configs)
        self.assertNotIn(("compv4", "cshuffle", "interwave"), configs)

    def test_valid_trait_configs_includes_mem_interwave(self):
        configs = valid_trait_configs()
        has_mem_interwave = any(p == "mem" and s == "interwave" for p, s in configs)
        self.assertTrue(has_mem_interwave)

    def test_needs_wave_expansion_wildcard(self):
        self.assertTrue(needs_wave_expansion({"wave_m": -1, "wave_n": 2}))
        self.assertTrue(needs_wave_expansion({"wave_m": 2, "wave_n": -1}))

    def test_needs_wave_expansion_explicit(self):
        self.assertFalse(needs_wave_expansion({"wave_m": 2, "wave_n": 2}))

    def test_needs_warp_expansion_wildcard(self):
        self.assertTrue(needs_warp_expansion({"warp_m": -1, "warp_n": 32}))

    def test_needs_warp_expansion_explicit(self):
        self.assertFalse(needs_warp_expansion({"warp_m": 32, "warp_n": 32}))

    def test_needs_pipeline_expansion_wildcard(self):
        self.assertTrue(needs_pipeline_expansion({"pipeline": "*"}))

    def test_needs_pipeline_expansion_explicit(self):
        self.assertFalse(needs_pipeline_expansion({"pipeline": "compv4"}))


if __name__ == "__main__":
    unittest.main()
