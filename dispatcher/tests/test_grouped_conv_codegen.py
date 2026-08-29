#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
TDD tests for codegen/unified_grouped_conv_codegen.py -- grouped convolution code generator.

These tests are written BEFORE the implementation exists.
Run: python3 -m pytest dispatcher/tests/test_grouped_conv_codegen.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from codegen_common import TileConfig, TraitConfigBase  # noqa: E402

from unified_grouped_conv_codegen import (  # noqa: E402
    GroupedConvVariant,
    GroupedConvLayout,
    GroupedConvKernelConfig,
    GroupedConvTypeMappings,
    GroupedConvTraitConfig,
    CKTileGroupedConvKernelGenerator,
    GroupedConvDispatcherWrapperGenerator,
    UnifiedGroupedConvCodegen,
)


# =============================================================================
# TestGroupedConvVariant
# =============================================================================


class TestGroupedConvVariant(unittest.TestCase):
    """Test GroupedConvVariant enum values."""

    def test_forward_value(self):
        self.assertEqual(GroupedConvVariant.FORWARD.value, "forward")

    def test_backward_data_value(self):
        self.assertEqual(GroupedConvVariant.BACKWARD_DATA.value, "bwd_data")

    def test_backward_weight_value(self):
        self.assertEqual(GroupedConvVariant.BACKWARD_WEIGHT.value, "bwd_weight")

    def test_all_variants_exist(self):
        self.assertIn(GroupedConvVariant.FORWARD, GroupedConvVariant)
        self.assertIn(GroupedConvVariant.BACKWARD_DATA, GroupedConvVariant)
        self.assertIn(GroupedConvVariant.BACKWARD_WEIGHT, GroupedConvVariant)


# =============================================================================
# TestGroupedConvLayout
# =============================================================================


class TestGroupedConvLayout(unittest.TestCase):
    """Test GroupedConvLayout enum for 1D/2D/3D layouts."""

    def test_nhwgc_value(self):
        self.assertEqual(GroupedConvLayout.NHWGC.value, "NHWGC")

    def test_gkyxc_value(self):
        self.assertEqual(GroupedConvLayout.GKYXC.value, "GKYXC")

    def test_nhwgk_value(self):
        self.assertEqual(GroupedConvLayout.NHWGK.value, "NHWGK")

    def test_1d_layouts_exist(self):
        """1D conv layouts (e.g., NWGC, GYXC, NWGK)."""
        layouts_1d = [
            lay
            for lay in GroupedConvLayout
            if "W" in lay.value and "H" not in lay.value
        ]
        self.assertGreater(len(layouts_1d), 0)

    def test_2d_layouts_exist(self):
        """2D conv layouts (e.g., NHWGC, GKYXC, NHWGK)."""
        layouts_2d = [lay for lay in GroupedConvLayout if "HW" in lay.value]
        self.assertGreater(len(layouts_2d), 0)

    def test_3d_layouts_exist(self):
        """3D conv layouts (e.g., NDHWGC, GDKYXC)."""
        layouts_3d = [
            lay for lay in GroupedConvLayout if "D" in lay.value or "DHW" in lay.value
        ]
        self.assertGreater(len(layouts_3d), 0)


# =============================================================================
# TestGroupedConvKernelConfig
# =============================================================================


class TestGroupedConvKernelConfig(unittest.TestCase):
    """Test GroupedConvKernelConfig dataclass."""

    def _make_tile(self):
        return TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)

    def _make_trait(self):
        return GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )

    def test_name_contains_grouped_conv_fwd(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        name = config.name("fp16")
        self.assertIn("grouped_conv_fwd", name)

    def test_name_backward_data_contains_bwd_data(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.BACKWARD_DATA,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        name = config.name("fp16")
        self.assertIn("bwd_data", name)

    def test_is_valid_for_arch_supported(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        self.assertTrue(config.is_valid_for_arch("gfx942"))

    def test_is_valid_for_arch_unsupported(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        self.assertFalse(config.is_valid_for_arch("gfx600"))


# =============================================================================
# TestGroupedConvTypeMappings
# =============================================================================


class TestGroupedConvTypeMappings(unittest.TestCase):
    """Test GroupedConvTypeMappings class."""

    def test_dtype_to_ck_fp16(self):
        self.assertEqual(GroupedConvTypeMappings.DTYPE_TO_CK["fp16"], "half_t")

    def test_dtype_to_ck_bf16(self):
        self.assertIn("bf16", GroupedConvTypeMappings.DTYPE_TO_CK)

    def test_dtype_to_ck_fp32(self):
        self.assertIn("fp32", GroupedConvTypeMappings.DTYPE_TO_CK)

    def test_get_layouts_2d_has_in_wei_out_keys(self):
        layouts = GroupedConvTypeMappings.get_layouts(2)
        self.assertIn("in", layouts)
        self.assertIn("wei", layouts)
        self.assertIn("out", layouts)

    def test_get_layouts_2d_returns_dict(self):
        layouts = GroupedConvTypeMappings.get_layouts(2)
        self.assertIsInstance(layouts, dict)

    def test_get_layouts_1d(self):
        layouts = GroupedConvTypeMappings.get_layouts(1)
        self.assertIn("in", layouts)
        self.assertIn("wei", layouts)
        self.assertIn("out", layouts)

    def test_get_layouts_3d(self):
        layouts = GroupedConvTypeMappings.get_layouts(3)
        self.assertIn("in", layouts)
        self.assertIn("wei", layouts)
        self.assertIn("out", layouts)


# =============================================================================
# TestCKTileGroupedConvKernelGenerator
# =============================================================================


class TestCKTileGroupedConvKernelGenerator(unittest.TestCase):
    """Test CKTileGroupedConvKernelGenerator.generate()."""

    def _make_config(self):
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )
        return GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )

    def test_generate_contains_pragma_once(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIn("#pragma once", result)

    def test_generate_contains_forward_kernel_include(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIn("grouped_convolution_forward_kernel.hpp", result)

    def test_generate_returns_non_empty_string(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_generate_valid_cpp_structure(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIn("#include", result)
        self.assertIn("ck_tile", result)


# =============================================================================
# TestGroupedConvDispatcherWrapperGenerator
# =============================================================================


class TestGroupedConvDispatcherWrapperGenerator(unittest.TestCase):
    """Test GroupedConvDispatcherWrapperGenerator.generate()."""

    def _make_config(self):
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )
        return GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )

    def test_generate_contains_dispatcher_registration(self):
        gen = GroupedConvDispatcherWrapperGenerator("fp16")
        config = self._make_config()
        kernel_path = DISPATCHER_DIR / "build" / "generated" / "test_kernel.hpp"
        output_dir = DISPATCHER_DIR / "build" / "generated"
        result = gen.generate(config, kernel_path, output_dir)
        self.assertIn("dispatcher", result)
        self.assertIn("KernelKey", result)
        self.assertIn("KernelInstancePtr", result)

    def test_generate_contains_pragma_once(self):
        gen = GroupedConvDispatcherWrapperGenerator("fp16")
        config = self._make_config()
        kernel_path = DISPATCHER_DIR / "build" / "generated" / "test_kernel.hpp"
        output_dir = DISPATCHER_DIR / "build" / "generated"
        result = gen.generate(config, kernel_path, output_dir)
        self.assertIn("#pragma once", result)

    def test_generate_valid_cpp(self):
        gen = GroupedConvDispatcherWrapperGenerator("fp16")
        config = self._make_config()
        kernel_path = DISPATCHER_DIR / "build" / "generated" / "test_kernel.hpp"
        output_dir = DISPATCHER_DIR / "build" / "generated"
        result = gen.generate(config, kernel_path, output_dir)
        self.assertIn("#include", result)
        self.assertIn("namespace", result)


# =============================================================================
# TestUnifiedGroupedConvCodegen
# =============================================================================


class TestUnifiedGroupedConvCodegen(unittest.TestCase):
    """Test UnifiedGroupedConvCodegen.generate_all()."""

    def test_generate_all_returns_dict_with_expected_keys(self):
        output_dir = DISPATCHER_DIR / "build" / "generated" / "grouped_conv"
        output_dir.mkdir(parents=True, exist_ok=True)
        codegen = UnifiedGroupedConvCodegen(
            output_dir=output_dir,
            datatype="fp16",
            ndim_spatial=2,
            gpu_target="gfx942",
        )
        with patch.object(
            codegen,
            "_get_configs",
            return_value=[],  # Mock empty config list for fast test
        ):
            results = codegen.generate_all(parallel=False)
        self.assertIn("kernels", results)
        self.assertIn("wrappers", results)
        self.assertIn("failed", results)
        self.assertIsInstance(results["kernels"], list)
        self.assertIsInstance(results["wrappers"], list)
        self.assertIsInstance(results["failed"], list)

    def test_generate_all_with_mock_config_produces_output(self):
        output_dir = DISPATCHER_DIR / "build" / "generated" / "grouped_conv_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        codegen = UnifiedGroupedConvCodegen(
            output_dir=output_dir,
            datatype="fp16",
            ndim_spatial=2,
            gpu_target="gfx942",
        )
        # Use a real config - patch the config source to return one config
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )
        config = GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )

        with patch.object(codegen, "_get_configs", return_value=[config]):
            results = codegen.generate_all(parallel=False)
        self.assertIsInstance(results, dict)
        self.assertIn("kernels", results)


# =============================================================================
# TestSharedImports
# =============================================================================


class TestSharedImports(unittest.TestCase):
    """Verify TileConfig from codegen_common and GroupedConvTraitConfig extends TraitConfigBase."""

    def test_tile_config_has_expected_fields(self):
        """TileConfig from codegen_common has tile_m, tile_n, tile_k, etc."""
        tc = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertEqual(tc.tile_m, 128)
        self.assertEqual(tc.tile_n, 128)
        self.assertEqual(tc.tile_k, 32)
        self.assertEqual(tc.warp_m, 2)
        self.assertEqual(tc.warp_n, 2)
        self.assertEqual(tc.warp_k, 1)
        self.assertEqual(tc.warp_tile_m, 32)
        self.assertEqual(tc.warp_tile_n, 32)
        self.assertEqual(tc.warp_tile_k, 16)

    def test_tile_config_is_from_codegen_common(self):
        """TileConfig used by grouped conv is the same as codegen_common.TileConfig."""
        tc = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertTrue(tc.is_valid())

    def test_grouped_conv_trait_config_extends_trait_config_base(self):
        """GroupedConvTraitConfig extends TraitConfigBase."""
        self.assertTrue(issubclass(GroupedConvTraitConfig, TraitConfigBase))

    def test_grouped_conv_trait_config_has_double_smem_buffer(self):
        """GroupedConvTraitConfig has double_smem_buffer field."""
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=True,
            num_groups_to_merge=2,
        )
        self.assertTrue(trait.double_smem_buffer)
        self.assertEqual(trait.num_groups_to_merge, 2)

    def test_grouped_conv_trait_config_has_num_groups_to_merge(self):
        """GroupedConvTraitConfig has num_groups_to_merge field."""
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=4,
        )
        self.assertEqual(trait.num_groups_to_merge, 4)

    def test_grouped_conv_trait_config_inherits_base_fields(self):
        """GroupedConvTraitConfig inherits pipeline, epilogue, scheduler from base."""
        trait = GroupedConvTraitConfig(
            "compv4",
            "cshuffle",
            "intrawave",
            True,
            True,
            True,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )
        self.assertEqual(trait.pipeline, "compv4")
        self.assertEqual(trait.epilogue, "cshuffle")
        self.assertEqual(trait.scheduler, "intrawave")
        self.assertTrue(trait.pad_m)
        self.assertTrue(trait.pad_n)
        self.assertTrue(trait.pad_k)


# =============================================================================
# TestTwoStageBwdWeightCodegen
# =============================================================================


def _make_two_stage_config():
    """Helper: create a two-stage bwd_weight config."""
    return GroupedConvKernelConfig(
        tile=TileConfig(16, 64, 64, 1, 4, 1, 16, 16, 32),
        trait=GroupedConvTraitConfig(
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            two_stage=True,
        ),
        variant=GroupedConvVariant.BACKWARD_WEIGHT,
        ndim_spatial=2,
        arch="gfx942",
    )


class TestTwoStageBwdWeightCodegen(unittest.TestCase):
    """Tests for two-stage backward weight kernel generation."""

    def test_kernel_name_contains_2stage(self):
        config = _make_two_stage_config()
        name = config.name("fp16")
        self.assertIn("_2stage", name)
        self.assertIn("bwd_weight", name)

    def test_single_stage_name_has_no_2stage(self):
        config = _make_two_stage_config()
        config.trait.two_stage = False
        name = config.name("fp16")
        self.assertNotIn("_2stage", name)

    def test_generate_contains_elementwise_include(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("elementwise.hpp", code)

    def test_generate_contains_workspace_type(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("WorkspaceDataType", code)

    def test_generate_contains_elementwise_kernel(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("ElementWiseKernel", code)

    def test_generate_contains_launch_kernel_time_mask(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("launch_kernel_time_mask", code)

    def test_two_stage_uses_fp32_workspace_vector_size_c(self):
        # Two-stage writes the GEMM result to an fp32 workspace, so it uses the
        # configured VectorSizeC directly instead of forcing it to 1.
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("WorkspaceDataType = float", code)
        self.assertIn("Config::VectorSizeC", code)
        self.assertNotIn("VectorSizeC_TwoStage", code)

    def test_generate_contains_workspace_memset(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("hipMemsetAsync", code)

    def test_single_stage_does_not_contain_workspace(self):
        config = _make_two_stage_config()
        config.trait.two_stage = False
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertNotIn("WorkspaceDataType", code)
        self.assertNotIn("ElementWiseKernel", code)

    def test_default_configs_include_two_stage(self):
        from unified_grouped_conv_codegen import get_default_configs

        configs = get_default_configs(
            arch="gfx942",
            variants=[GroupedConvVariant.BACKWARD_WEIGHT],
            ndims=[2],
        )
        two_stage = [c for c in configs if c.trait.two_stage]
        single_stage = [c for c in configs if not c.trait.two_stage]
        self.assertGreater(len(two_stage), 0, "Should have two-stage configs")
        self.assertGreater(
            len(single_stage), 0, "Should still have single-stage configs"
        )


if __name__ == "__main__":
    unittest.main()
