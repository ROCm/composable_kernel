# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for GemmABQuantKernelBuilder (gemm_abquant operator)."""

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gemm_abquant_instance_builder import GemmABQuantKernelBuilder  # noqa: E402

_MINIMAL_CONFIG = {
    "tile_config": {
        "tile_m": {"values": [64]},
        "tile_n": {"values": [64]},
        "tile_k": {"values": [128]},
        "warp_m": {"values": [2]},
        "warp_n": {"values": [2]},
        "warp_k": {"values": [1]},
        "warp_tile_m": {"values": [16]},
        "warp_tile_n": {"values": [16]},
        "warp_tile_k": {"values": [64]},
    },
    "trait_config": {
        "pipeline": {"values": ["compv3"]},
        "scheduler": {"values": ["intrawave"]},
        "epilogue": {"values": ["default"]},
        "pad_m": {"values": [False]},
        "pad_n": {"values": [False]},
        "pad_k": {"values": [False]},
        "a_preshuffle_quant": {"values": [False]},
        "b_preshuffle_quant": {"values": [False, True]},
    },
    "k_block_per_cu": 1,
    "group_size_k": 128,
    "group_size_n": {"values": [1, 128]},
}


def _make_builder(tmpdir, config=None, **kwargs):
    cfg = config if config is not None else _MINIMAL_CONFIG
    cfg_path = os.path.join(tmpdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return GemmABQuantKernelBuilder(
        kernel_name_prefix="gemm_abquant",
        working_path=tmpdir,
        gpu_target=kwargs.get("gpu_target", "gfx942"),
        datatype=kwargs.get("datatype", "fp8"),
        layout=kwargs.get("layout", "rcr"),
        config_json=cfg_path,
    )


class TestGemmABQuantBuilderInit(unittest.TestCase):
    def test_group_size_k_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            self.assertEqual(builder.group_size_k, 128)

    def test_group_size_n_dict_parsed(self):
        """group_size_n specified as {"values": [...]} should expand to a list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            self.assertIsInstance(builder.group_size_n_values, list)
            self.assertIn(1, builder.group_size_n_values)
            self.assertIn(128, builder.group_size_n_values)

    def test_group_size_n_scalar_parsed(self):
        """group_size_n specified as a plain int should become a single-element list."""
        cfg = json.loads(json.dumps(_MINIMAL_CONFIG))
        cfg["group_size_n"] = 64
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir, config=cfg)
            self.assertEqual(builder.group_size_n_values, [64])

    def test_kernel_name_prefix_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            self.assertEqual(builder.kernel_name_prefix, "gemm_abquant")

    def test_working_path_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, "workdir")
            cfg_path = os.path.join(tmpdir, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(_MINIMAL_CONFIG, f)
            GemmABQuantKernelBuilder("gemm_abquant", sub, "gfx942", "fp8", "rcr", cfg_path)
            self.assertTrue(os.path.isdir(sub))


class TestGemmABQuantTraitCombinations(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.builder = _make_builder(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_trait_combinations_non_empty(self):
        combos = self.builder._generate_trait_combinations()
        self.assertGreater(len(combos), 0)

    def test_trait_combo_is_8_tuple(self):
        """ABQuant uses 8-tuples:
        (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, a_preshuffle_quant, b_preshuffle_quant)
        """
        combos = self.builder._generate_trait_combinations()
        for combo in combos:
            self.assertEqual(
                len(combo), 8, f"Expected 8-tuple, got {len(combo)}: {combo}"
            )

    def test_pipeline_is_compv3(self):
        combos = self.builder._generate_trait_combinations()
        for combo in combos:
            self.assertEqual(combo[0], "compv3")

    def test_b_preshuffle_quant_values(self):
        combos = self.builder._generate_trait_combinations()
        b_preshuffle_vals = {c[7] for c in combos}
        self.assertIn(True, b_preshuffle_vals)
        self.assertIn(False, b_preshuffle_vals)

    def test_a_preshuffle_quant_default_false(self):
        """Config only lists False for a_preshuffle_quant, so all combos should have False."""
        combos = self.builder._generate_trait_combinations()
        for combo in combos:
            self.assertFalse(combo[6], "a_preshuffle_quant should be False per minimal config")


class TestGemmABQuantTileConfigs(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.builder = _make_builder(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_tile_configs_non_empty(self):
        configs = self.builder._get_tile_configs()
        self.assertGreater(len(configs), 0)

    def test_tile_config_has_required_keys(self):
        required = {
            "tile_m", "tile_n", "tile_k",
            "warp_m", "warp_n", "warp_k",
            "warp_tile_m", "warp_tile_n", "warp_tile_k",
        }
        for cfg in self.builder._get_tile_configs():
            self.assertTrue(required.issubset(cfg.keys()))

    def test_tile_config_values_are_positive(self):
        for cfg in self.builder._get_tile_configs():
            for key, val in cfg.items():
                self.assertGreater(val, 0, f"{key}={val} must be positive")


class TestGemmABQuantTileConfigStr(unittest.TestCase):
    def test_tile_config_to_str_format(self):
        tile_config = {
            "tile_m": 128, "tile_n": 128, "tile_k": 64,
            "warp_m": 2, "warp_n": 2, "warp_k": 1,
            "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16,
        }
        result = GemmABQuantKernelBuilder._tile_config_to_str(tile_config)
        self.assertEqual(result, "128x128x64_2x2x1_32x32x16")


class TestGemmABQuantGroupSizeN(unittest.TestCase):
    def test_group_size_n_values_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            gsn = builder._get_group_size_n_values()
            self.assertIsInstance(gsn, list)
            self.assertGreater(len(gsn), 0)

    def test_group_size_n_values_match_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            gsn = builder._get_group_size_n_values()
            self.assertIn(1, gsn)
            self.assertIn(128, gsn)


class TestGemmABQuantLayoutVariants(unittest.TestCase):
    LAYOUTS = ["rcr", "rrr", "ccr", "crr"]

    def test_all_layouts_construct(self):
        for layout in self.LAYOUTS:
            with self.subTest(layout=layout):
                with tempfile.TemporaryDirectory() as tmpdir:
                    builder = _make_builder(tmpdir, layout=layout)
                    self.assertEqual(builder.layout, layout)


class TestGemmABQuantDataTypes(unittest.TestCase):
    DTYPES = ["fp8", "bf8"]

    def test_all_dtypes_construct(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                with tempfile.TemporaryDirectory() as tmpdir:
                    builder = _make_builder(tmpdir, datatype=dtype)
                    self.assertEqual(builder.datatype, dtype)


class TestGemmABQuantMaxInstances(unittest.TestCase):
    def test_max_instances_none_returns_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(_MINIMAL_CONFIG, f)
            builder = GemmABQuantKernelBuilder(
                "gemm_abquant", tmpdir, "gfx942", "fp8", "rcr",
                config_json=cfg_path, max_instances=None,
            )
            kernel_list = [
                {"tile_config": {"tile_m": 64, "tile_n": 64, "tile_k": 128,
                                 "warp_m": 2, "warp_n": 2, "warp_k": 1,
                                 "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 64},
                 "trait_combo": ("compv3", "default", "intrawave",
                                 False, False, False, False, False),
                 "group_size_n": 1}
            ]
            result = builder._apply_sampling(kernel_list)
            self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
