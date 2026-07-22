# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for GemmBQuantKernelBuilder (gemm_bquant operator)."""

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gemm_bquant_instance_builder import GemmBQuantKernelBuilder  # noqa: E402

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
        "b_preshuffle_quant": {"values": [False, True]},
    },
    "k_block_per_cu": 1,
    "group_size_k": 128,
}


def _make_builder(tmpdir, config=None, **kwargs):
    cfg = config if config is not None else _MINIMAL_CONFIG
    cfg_path = os.path.join(tmpdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return GemmBQuantKernelBuilder(
        kernel_name_prefix="gemm_bquant",
        working_path=tmpdir,
        gpu_target=kwargs.get("gpu_target", "gfx942"),
        datatype=kwargs.get("datatype", "fp8"),
        layout=kwargs.get("layout", "rcr"),
        config_json=cfg_path,
    )


class TestGemmBQuantBuilderInit(unittest.TestCase):
    def test_group_size_k_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            self.assertEqual(builder.group_size_k, 128)

    def test_group_size_k_custom(self):
        cfg = json.loads(json.dumps(_MINIMAL_CONFIG))
        cfg["group_size_k"] = 32
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir, config=cfg)
            self.assertEqual(builder.group_size_k, 32)

    def test_kernel_name_prefix_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            self.assertEqual(builder.kernel_name_prefix, "gemm_bquant")

    def test_working_path_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, "workdir")
            cfg_path = os.path.join(tmpdir, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(_MINIMAL_CONFIG, f)
            GemmBQuantKernelBuilder("gemm_bquant", sub, "gfx942", "fp8", "rcr", cfg_path)
            self.assertTrue(os.path.isdir(sub))


class TestGemmBQuantTraitCombinations(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.builder = _make_builder(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_trait_combinations_non_empty(self):
        combos = self.builder._generate_trait_combinations()
        self.assertGreater(len(combos), 0)

    def test_trait_combo_is_7_tuple(self):
        """BQuant trait tuple: (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, b_preshuffle_quant)."""
        combos = self.builder._generate_trait_combinations()
        for combo in combos:
            self.assertEqual(len(combo), 7, f"Expected 7-tuple, got {len(combo)}: {combo}")

    def test_pipeline_is_compv3(self):
        combos = self.builder._generate_trait_combinations()
        for combo in combos:
            self.assertEqual(combo[0], "compv3")

    def test_preshuffle_quant_values(self):
        combos = self.builder._generate_trait_combinations()
        preshuffle_vals = {c[6] for c in combos}
        self.assertIn(True, preshuffle_vals)
        self.assertIn(False, preshuffle_vals)

    def test_scheduler_intrawave(self):
        combos = self.builder._generate_trait_combinations()
        schedulers = {c[2] for c in combos}
        self.assertIn("intrawave", schedulers)


class TestGemmBQuantTileConfigs(unittest.TestCase):
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

    def test_tile_m_matches_config(self):
        configs = self.builder._get_tile_configs()
        tile_m_vals = {c["tile_m"] for c in configs}
        self.assertIn(64, tile_m_vals)


class TestGemmBQuantLayoutVariants(unittest.TestCase):
    LAYOUTS = ["rcr", "rrr", "ccr", "crr"]

    def test_all_layouts_construct(self):
        for layout in self.LAYOUTS:
            with self.subTest(layout=layout):
                with tempfile.TemporaryDirectory() as tmpdir:
                    builder = _make_builder(tmpdir, layout=layout)
                    self.assertEqual(builder.layout, layout)


class TestGemmBQuantDataTypes(unittest.TestCase):
    DTYPES = ["fp8", "bf8"]

    def test_all_dtypes_construct(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                with tempfile.TemporaryDirectory() as tmpdir:
                    builder = _make_builder(tmpdir, datatype=dtype)
                    self.assertEqual(builder.datatype, dtype)


class TestGemmBQuantMaxInstances(unittest.TestCase):
    def test_max_instances_none_returns_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(_MINIMAL_CONFIG, f)
            builder = GemmBQuantKernelBuilder(
                "gemm_bquant", tmpdir, "gfx942", "fp8", "rcr",
                config_json=cfg_path, max_instances=None,
            )
            kernel_list = [
                {"tile_config": {"tile_m": 64, "tile_n": 64, "tile_k": 128,
                                 "warp_m": 2, "warp_n": 2, "warp_k": 1,
                                 "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 64},
                 "trait_combo": ("compv3", "default", "intrawave", False, False, False, False)}
            ]
            result = builder._apply_sampling(kernel_list)
            self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
