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
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEMM_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _GEMM_DIR)
_CK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_GEMM_DIR)))
sys.path.append(os.path.join(_CK_ROOT, "dispatcher", "codegen"))

import gemm_validation_utils as vu  # noqa: E402
from arch_specs_generated import WARP_TILE_SUPPORTED_COMBINATIONS  # noqa: E402
from batched_gemm_instance_builder import (  # noqa: E402
    BATCHED_GEMM_PRESHUFFLE_ERROR,
    BATCHED_GEMM_UNSUPPORTED_PIPELINES,
    BatchedGemmKernelBuilder,
    check_batched_gemm_pipelines,
    split_trait,
)
from batched_gemm_benchmark import GemmBenchmark  # noqa: E402

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


def _builder(tmp, config_path, gpu_target="gfx1250", dtype="fp16", layout="rcr"):
    return BatchedGemmKernelBuilder(tmp, gpu_target, dtype, layout, config_path)


def _write_config(tmp, pipelines, epilogues):
    with open(_CI_CONFIG) as f:
        cfg = json.load(f)
    cfg["trait_config"]["pipeline"]["values"] = list(pipelines)
    cfg["trait_config"]["epilogue"]["values"] = list(epilogues)
    path = os.path.join(tmp, "cfg.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _write_tile_config(tmp, **values):
    """CI config with the given tile_config entries replaced by value lists."""
    with open(_CI_CONFIG) as f:
        cfg = json.load(f)
    for key, vals in values.items():
        cfg["tile_config"][key] = {"values": list(vals)}
    path = os.path.join(tmp, "tile_cfg.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _kernels(config_path, gpu_target="gfx1250", dtype="fp16", layout="rcr"):
    with tempfile.TemporaryDirectory() as tmp:
        return _builder(
            tmp, config_path, gpu_target, dtype, layout
        )._get_sampled_kernel_list()


def _warp_tiles(kernels):
    return {
        tuple(k["tile_config"][f"warp_tile_{d}"] for d in "mnk") for k in kernels
    }


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


    def test_benchmark_driver_labels(self):
        """The benchmark driver must not truncate multi-token pipeline names."""
        driver = GemmBenchmark(".")
        for pipeline, epilogue in (
            ("compv4", "cshuffle"),
            ("comp_async", "cshuffle"),
            ("comp_tdm", "tdm"),
            ("comp_tdm_v2", "tdm"),
        ):
            name = (
                f"benchmark_batched_gemm_fp16_rcr_{pipeline}_{epilogue}_intrawave"
                "_False_False_False_256x256x64_2x2x1_16x16x32"
            )
            info = driver.extract_kernel_info(Path(name))
            self.assertEqual(
                (info["pipeline"], info["epilogue"], info["scheduler"]),
                (pipeline, epilogue, "intrawave"),
                name,
            )
            self.assertIn(pipeline, info["config_id"])


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
        # k=32 serves fp16/bf16 (16x16x32), k=4 fp32 (16x16x4), k=64 fp8/bf8
        # (16x16x64); the builder keeps only the WMMA tile of the requested datatype.
        self.assertEqual(tc["warp_tile_k"]["values"], [4, 32, 64])

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


class TestDtypeLayoutCoverage(unittest.TestCase):
    """fp16/bf16/fp32/fp8/bf8 x rcr/rrr/crr/ccr, each dtype on its own warp tiles."""

    def test_op_warp_tile_allowed(self):
        self.assertTrue(vu.op_warp_tile_allowed("gfx1250", "fp32", [16, 16, 4]))
        self.assertFalse(vu.op_warp_tile_allowed("gfx1250", "fp32", [16, 16, 32]))
        self.assertFalse(vu.op_warp_tile_allowed("gfx1250", "fp16", [16, 16, 4]))
        self.assertTrue(vu.op_warp_tile_allowed("gfx942", "fp32", [32, 32, 8]))
        self.assertFalse(vu.op_warp_tile_allowed("gfx950", "fp32", [16, 16, 32]))
        # fp32 has no WMMA rows on gfx12 (gfx1201); a mixed target list must
        # satisfy every target.
        self.assertFalse(vu.op_warp_tile_allowed("gfx1201", "fp32", [16, 16, 4]))
        self.assertFalse(
            vu.op_warp_tile_allowed("gfx942;gfx1201", "fp32", [32, 32, 8])
        )
        self.assertTrue(vu.op_warp_tile_allowed("gfx942;gfx950", "fp32", [32, 32, 8]))
        self.assertTrue(vu.op_warp_tile_allowed("gfx942;gfx1201", "fp16", [16, 16, 16]))
        # gfx1201 WMMA has only 16x16x16 for bf16/fp8/bf8.
        self.assertTrue(vu.op_warp_tile_allowed("gfx1201", "bf16", [16, 16, 16]))
        self.assertFalse(vu.op_warp_tile_allowed("gfx1201", "fp8", [16, 32, 8]))
        self.assertFalse(vu.op_warp_tile_allowed("gfx942;gfx1201", "bf16", [32, 32, 8]))
        # Other (arch, dtype) pairs are left to the shared table.
        self.assertTrue(vu.op_warp_tile_allowed("gfx942", "fp16", [16, 16, 4]))

    def test_only_opted_in_ops_use_op_rows(self):
        self.assertTrue(BatchedGemmKernelBuilder.USE_OP_WARP_TILE_ROWS)
        self.assertFalse(BatchedGemmKernelBuilder.__bases__[0].USE_OP_WARP_TILE_ROWS)

    def test_op_rows_match_dispatcher_arch_specs(self):
        for arch, rows in vu.OP_ARCH_WARP_TILES.items():
            spec = WARP_TILE_SUPPORTED_COMBINATIONS[arch]
            for dtype, tiles in rows.items():
                (key,) = [k for k in spec if k.startswith(f"{dtype}_{dtype}_")]
                self.assertEqual(
                    sorted(map(list, tiles)), sorted(spec[key]), (arch, dtype)
                )

    def test_gfx1201_keeps_only_16x16x16(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 2x4 waves: gfx1201 has no 2x2 warp layout.
            path = _write_tile_config(
                tmp,
                warp_n=[4],
                warp_tile_m=[16, 32],
                warp_tile_n=[16, 32],
                warp_tile_k=[8, 16, 32],
            )
            for dtype in ("fp16", "bf16", "fp8", "bf8"):
                with self.subTest(dtype=dtype):
                    tiles = _warp_tiles(_kernels(path, "gfx1201", dtype))
                    self.assertEqual(tiles, {(16, 16, 16)})

    def test_gfx1250_every_dtype_layout_keeps_its_wmma_tile(self):
        for dtype, wmma in (("fp16", 32), ("bf16", 32), ("fp32", 4), ("fp8", 64), ("bf8", 64)):
            for layout in ("rcr", "rrr", "crr", "ccr"):
                with self.subTest(dtype=dtype, layout=layout):
                    tiles = _warp_tiles(_kernels(_CI_CONFIG, "gfx1250", dtype, layout))
                    self.assertEqual(tiles, {(16, 16, wmma)})

    def test_gfx1250_fp32_tdm_rcr_only(self):
        """fp32 comp_tdm / comp_tdm_v2 give wrong results off rcr on gfx1250."""
        for layout in ("rcr", "rrr", "crr", "ccr"):
            with self.subTest(layout=layout):
                pipes = {
                    k["trait_combo"][0]
                    for k in _kernels(_CI_CONFIG, "gfx1250", "fp32", layout)
                }
                self.assertIn("compv3", pipes)
                self.assertEqual(
                    pipes & set(_TDM_PIPELINES),
                    set(_TDM_PIPELINES) if layout == "rcr" else set(),
                )
        reason = vu.gfx1250_tdm_fp32_layout_reject_reason
        self.assertEqual(reason("comp_tdm", "fp32", "fp32", "rcr"), "")
        self.assertEqual(
            reason("comp_tdm", "fp32", "fp32", "rrr"),
            vu.GFX1250_TDM_FP32_LAYOUT_REJECT_REASON,
        )
        self.assertEqual(reason("comp_tdm_v2", "fp16", "fp16", "ccr"), "")
        self.assertEqual(reason("compv3", "fp32", "fp32", "ccr"), "")

    def test_gfx1250_fp32_accumulators_per_lane(self):
        """fp32 tiles with >= 512 accumulators per lane spill on gfx1250."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_tile_config(
                tmp, tile_m=[128, 256], tile_n=[256], warp_m=[2, 4], warp_k=[1]
            )
            for arch, dtype in (("gfx1250", "fp32"), ("gfx1250", "fp16")):
                tiles = {
                    (
                        k["tile_config"]["tile_m"],
                        k["tile_config"]["tile_n"],
                        k["tile_config"]["warp_m"] * k["tile_config"]["warp_n"],
                    )
                    for k in _kernels(path, arch, dtype)
                }
                with self.subTest(dtype=dtype):
                    # 256 acc/lane (128x256 on 4 waves, 256x256 on 8) is kept.
                    self.assertIn((128, 256, 4), tiles)
                    self.assertIn((256, 256, 8), tiles)
                    # 512 acc/lane (256x256 on 4 waves) only for non-fp32.
                    self.assertEqual((256, 256, 4) in tiles, dtype != "fp32")
        self.assertEqual(
            vu.gfx1250_fp32_tile_reject_reason("gfx1250", "fp32", 256, 256, 4),
            vu.GFX1250_FP32_ACC_REJECT_REASON,
        )
        self.assertEqual(
            vu.gfx1250_fp32_tile_reject_reason("gfx942", "fp32", 256, 256, 4), ""
        )

    def test_gfx9_fp32_uses_fp32_mfma_tiles(self):
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                tiles = _warp_tiles(
                    _kernels(os.path.join(_CONFIG_DIR, "default_ci_config.json"), arch, "fp32")
                )
                self.assertTrue(tiles)
                self.assertLessEqual(
                    tiles, {tuple(t) for t in vu.GFX9_FP32_WARP_TILES}
                )

    def test_fp32_header_uses_float(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = _builder(tmp, _CI_CONFIG, dtype="fp32", layout="ccr")
            k = b._get_sampled_kernel_list()[0]
            _, code = b._generate_kernel_instance(k["tile_config"], k["trait_combo"])
        for line in (
            "using ADataType = float;",
            "using CDataType = float;",
            "using ALayout = ck_tile::tensor_layout::gemm::ColumnMajor;",
            "using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;",
        ):
            self.assertIn(line, code)

    def test_fp8_bf8_headers_accumulate_into_half(self):
        for dtype, a_type in (("fp8", "ck_tile::fp8_t"), ("bf8", "ck_tile::bf8_t")):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                b = _builder(tmp, _CI_CONFIG, dtype=dtype, layout="rcr")
                k = b._get_sampled_kernel_list()[0]
                _, code = b._generate_kernel_instance(k["tile_config"], k["trait_combo"])
                self.assertIn(f"using ADataType = {a_type};", code)
                self.assertIn("using CDataType = ck_tile::fp16_t;", code)


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
