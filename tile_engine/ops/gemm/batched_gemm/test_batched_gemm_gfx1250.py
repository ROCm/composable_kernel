# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU unit tests for the gfx1250 batched GEMM Tile Engine configuration:
comp_async / comp_tdm / comp_tdm_v2 enablement, weight-preshuffle rejection,
config lint and trait parsing. No GPU is required."""

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEMM_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _GEMM_DIR)

import gemm_validation_utils as vu  # noqa: E402
from batched_gemm_instance_builder import (  # noqa: E402
    BATCHED_GEMM_PRESHUFFLE_ERROR,
    BATCHED_GEMM_UNSUPPORTED_PIPELINES,
    BatchedGemmKernelBuilder,
    check_batched_gemm_pipelines,
    split_trait,
)

_CONFIG_DIR = os.path.join(_HERE, "configs")
_FULL_CONFIG = os.path.join(_CONFIG_DIR, "default_config_gfx1250.json")
_CI_CONFIG = os.path.join(_CONFIG_DIR, "default_ci_config_gfx1250.json")
_LEGACY_CONFIGS = ("default_config.json", "default_ci_config.json")
_LDS_CAPACITY_GFX1250 = 327680
_NEW_PIPELINES = ("comp_async", "comp_tdm", "comp_tdm_v2")
_TDM_PIPELINES = ("comp_tdm", "comp_tdm_v2")
# Warp layouts of the gfx1250 row in the dispatcher arch specs.
_GFX1250_WARP_LAYOUTS = {
    (2, 4, 1),
    (1, 8, 1),
    (8, 1, 1),
    (4, 2, 1),
    (2, 1, 1),
    (1, 2, 2),
    (4, 1, 1),
    (1, 4, 1),
    (2, 2, 1),
}


def _builder(tmp, config_path, gpu_target="gfx1250"):
    return BatchedGemmKernelBuilder(tmp, gpu_target, "fp16", "rcr", config_path)


def _write_config(tmp, pipelines, epilogues):
    with open(_CI_CONFIG) as f:
        cfg = json.load(f)
    cfg["trait_config"]["pipeline"]["values"] = list(pipelines)
    cfg["trait_config"]["epilogue"]["values"] = list(epilogues)
    path = os.path.join(tmp, "cfg.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _kernels(config_path, gpu_target="gfx1250"):
    with tempfile.TemporaryDirectory() as tmp:
        return _builder(tmp, config_path, gpu_target)._get_sampled_kernel_list()


class TestTraitParsing(unittest.TestCase):
    def test_split_trait_round_trip(self):
        for trait in (
            "comp_tdm_v2_tdm_intrawave_false_false_false",
            "comp_tdm_tdm_intrawave_false_false_false",
            "comp_async_cshuffle_intrawave_false_false_false",
            "compv3_cshuffle_interwave_true_true_true",
        ):
            parts = split_trait(trait)
            self.assertEqual(len(parts), 6, trait)
            self.assertEqual("_".join(parts), trait)
        self.assertEqual(
            split_trait("comp_tdm_v2_tdm_intrawave_false_false_false")[0], "comp_tdm_v2"
        )
        self.assertEqual(
            split_trait("comp_async_cshuffle_intrawave_false_false_false")[1],
            "cshuffle",
        )


class TestPreshuffleRejection(unittest.TestCase):
    def test_check_rejects_preshuffle(self):
        for pipe in BATCHED_GEMM_UNSUPPORTED_PIPELINES:
            with self.assertRaises(ValueError) as ctx:
                check_batched_gemm_pipelines(["compv3", pipe])
            self.assertIn(BATCHED_GEMM_PRESHUFFLE_ERROR, str(ctx.exception))
            self.assertIn(pipe, str(ctx.exception))

    def test_check_accepts_supported(self):
        check_batched_gemm_pipelines(["mem", "compv3", "compv4"] + list(_NEW_PIPELINES))

    def test_error_mentions_b_offset(self):
        self.assertIn("batched_gemm_kernel.hpp", BATCHED_GEMM_PRESHUFFLE_ERROR)
        self.assertIn("unshuffled B", BATCHED_GEMM_PRESHUFFLE_ERROR)

    def test_pipeline_allowed_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = _builder(tmp, _CI_CONFIG)
            for pipe in BATCHED_GEMM_UNSUPPORTED_PIPELINES:
                with self.assertRaises(ValueError):
                    b._check_pipeline_allowed_for_op(pipe, "cshuffle")
            with self.assertRaisesRegex(ValueError, "pad_m=pad_n=pad_k=True"):
                b._check_pipeline_allowed_for_op("comp_async", "cshuffle")
            b._check_pipeline_allowed_for_op("comp_async", "cshuffle", True, True, True)
            b._check_pipeline_allowed_for_op("comp_tdm", "tdm")
            b._check_pipeline_allowed_for_op("comp_tdm_v2", "tdm")

    def test_config_with_preshuffle_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, ["compv3", "weight_preshuffle"], ["cshuffle"])
            b = _builder(tmp, path)
            with self.assertRaises(ValueError) as ctx:
                b._get_sampled_kernel_list()
            self.assertIn(BATCHED_GEMM_PRESHUFFLE_ERROR, str(ctx.exception))


class TestGfx1250Configs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = _kernels(_FULL_CONFIG)
        cls.ci = _kernels(_CI_CONFIG)

    def _all(self):
        return self.full + self.ci

    def test_configs_nonempty_and_cover_new_pipelines(self):
        for kernels in (self.full, self.ci):
            pipes = {k["trait_combo"][0] for k in kernels}
            for p in _NEW_PIPELINES:
                self.assertIn(p, pipes)

    def test_traits_valid(self):
        for k in self._all():
            pipe, epi, sched, pad_m, pad_n, pad_k, persistent = k["trait_combo"]
            self.assertTrue(
                vu.is_trait_combination_valid(
                    pipe,
                    epi,
                    sched,
                    persistent,
                    "batched_gemm",
                    "rcr",
                    pad_m=pad_m,
                    pad_n=pad_n,
                    pad_k=pad_k,
                ),
                k["name"],
            )
            if pipe in _TDM_PIPELINES:
                self.assertEqual((epi, sched), ("tdm", "intrawave"), k["name"])
                self.assertEqual((pad_m, pad_n, pad_k), (False,) * 3, k["name"])
            if pipe == "comp_async":
                self.assertEqual((epi, sched), ("cshuffle", "intrawave"), k["name"])
                self.assertEqual((pad_m, pad_n, pad_k), (True,) * 3, k["name"])
            if epi == "tdm":
                self.assertIn(pipe, _TDM_PIPELINES, k["name"])

    def test_warp_layout_and_warp_tile(self):
        for k in self._all():
            t = k["tile_config"]
            self.assertIn(
                (t["warp_m"], t["warp_n"], t["warp_k"]),
                _GFX1250_WARP_LAYOUTS,
                k["name"],
            )
            self.assertEqual(
                (t["warp_tile_m"], t["warp_tile_n"], t["warp_tile_k"]),
                (16, 16, 32),
                k["name"],
            )

    def test_comp_tdm_v2_four_waves(self):
        for k in self._all():
            if k["trait_combo"][0] == "comp_tdm_v2":
                t = k["tile_config"]
                self.assertEqual(t["warp_m"] * t["warp_n"] * t["warp_k"], 4, k["name"])

    def test_lds_double_buffer(self):
        for k in self._all():
            pipe = k["trait_combo"][0]
            if pipe not in _NEW_PIPELINES:
                continue
            t = k["tile_config"]
            ok, msg = vu.validate_lds_capacity(
                t["tile_m"], t["tile_n"], t["tile_k"], "fp16", "fp16", pipe, "gfx1250"
            )
            self.assertTrue(ok, msg)
            self.assertLessEqual(
                (t["tile_m"] + t["tile_n"]) * t["tile_k"] * 2,
                _LDS_CAPACITY_GFX1250 // 2,
            )

    def test_ci_warp_tile_values(self):
        with open(_CI_CONFIG) as f:
            tc = json.load(f)["tile_config"]
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32])

    def test_schema_matches_ci(self):
        with open(_FULL_CONFIG) as f:
            full = json.load(f)
        with open(_CI_CONFIG) as f:
            ci = json.load(f)
        self.assertEqual(set(full), set(ci))
        self.assertEqual(set(full["tile_config"]), set(ci["tile_config"]))
        self.assertEqual(set(full["trait_config"]), set(ci["trait_config"]))


class TestGeneratedHeaders(unittest.TestCase):
    def _gen(self, pipe, epi, warp=(2, 2, 1), pad=False):
        with tempfile.TemporaryDirectory() as tmp:
            b = _builder(tmp, _CI_CONFIG)
            tile = {
                "tile_m": 64,
                "tile_n": 64,
                "tile_k": 64,
                "warp_m": warp[0],
                "warp_n": warp[1],
                "warp_k": warp[2],
                "warp_tile_m": 16,
                "warp_tile_n": 16,
                "warp_tile_k": 32,
            }
            _, code = b._generate_kernel_instance(
                tile, (pipe, epi, "intrawave", pad, pad, pad, False)
            )
            return code

    def test_tdm_headers(self):
        for pipe, impl in (
            ("comp_tdm", "GemmPipelineAgBgCrCompTDMV1"),
            ("comp_tdm_v2", "GemmPipelineAgBgCrCompTDMV2"),
        ):
            code = self._gen(pipe, "tdm")
            self.assertIn(impl, code)
            self.assertIn("TdmEpilogue", code)
            self.assertIn("DoubleSmemBuffer = true", code)
            self.assertIn("k_batch != 1", code)
            self.assertIn("BatchedGemmKernel", code)

    def test_comp_async_header(self):
        code = self._gen("comp_async", "cshuffle", pad=True)
        self.assertIn("GemmPipelineAgBgCrCompAsync", code)
        self.assertIn("DoubleSmemBuffer = true", code)
        self.assertNotIn("TdmEpilogue", code)
        self.assertNotIn("k_batch != 1", code)

    def test_legacy_header_unchanged_traits(self):
        code = self._gen("compv3", "cshuffle")
        self.assertIn("DoubleSmemBuffer = false", code)
        self.assertNotIn("k_batch != 1", code)


class TestOtherArchesUnaffected(unittest.TestCase):
    def test_new_pipelines_not_listed_off_gfx1250(self):
        for arch in ("gfx942", "gfx950", "gfx90a"):
            for cfg in (_FULL_CONFIG, _CI_CONFIG):
                pipes = {k["trait_combo"][0] for k in _kernels(cfg, arch)}
                epis = {k["trait_combo"][1] for k in _kernels(cfg, arch)}
                self.assertFalse(pipes & set(_NEW_PIPELINES), (arch, cfg, pipes))
                self.assertNotIn("tdm", epis, (arch, cfg))

    def test_legacy_configs_have_no_gfx1250_pipelines(self):
        for name in _LEGACY_CONFIGS:
            with open(os.path.join(_CONFIG_DIR, name)) as f:
                traits = json.load(f)["trait_config"]
            self.assertFalse(
                set(traits["pipeline"]["values"]) & set(_NEW_PIPELINES), name
            )
            self.assertNotIn("tdm", traits["epilogue"]["values"], name)


class TestCMake(unittest.TestCase):
    def test_cmake_gfx1250_branch(self):
        with open(os.path.join(_HERE, "CMakeLists.txt")) as f:
            text = f.read()
        self.assertIn("gemm_trait_parse.cmake", text)
        self.assertIn("ck_tile_gemm_split_trait", text)
        self.assertIn("default_config_gfx1250.json", text)
        self.assertIn("comp_async;comp_tdm;comp_tdm_v2", text)
        self.assertNotIn("preshuffle_tdm", text)


if __name__ == "__main__":
    unittest.main()
