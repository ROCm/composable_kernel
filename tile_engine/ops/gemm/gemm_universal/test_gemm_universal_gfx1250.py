# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the gfx1250 gemm_universal configs and the comp_async /
comp_tdm / comp_tdm_v2 kernels generated from them by the Tile Engine builder
and by the dispatcher codegen. No GPU is required.

Run: python3 -m pytest test_gemm_universal_gfx1250.py -v
"""

import collections
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
_GEMM_DIR = _HERE.parent
_CK_ROOT = _GEMM_DIR.parent.parent.parent
_CODEGEN_DIR = _CK_ROOT / "dispatcher" / "codegen"
_DISPATCHER_PY_DIR = _CK_ROOT / "dispatcher" / "python"
_BUILDER = _HERE / "gemm_universal_instance_builder.py"
_FULL_CONFIG = _HERE / "configs" / "default_config_gfx1250.json"
_CI_CONFIG = _HERE / "configs" / "default_ci_config_gfx1250.json"
_SHARED_CI_CONFIG = _GEMM_DIR / "configs" / "default_ci_config.json"

sys.path.insert(0, str(_GEMM_DIR))
sys.path.insert(0, str(_CODEGEN_DIR))
sys.path.insert(0, str(_DISPATCHER_PY_DIR))

import gemm_validation_utils as vu  # noqa: E402
from arch_specs_generated import WARP_TILE_SUPPORTED_COMBINATIONS  # noqa: E402
from gemm_instance_builder import GemmKernelBuilder  # noqa: E402


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("gemm_universal_builder", _BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gub = _load_builder_module()

DTYPES = ["fp16", "bf16", "fp8", "bf8"]
LAYOUTS = ["rcr", "rrr", "crr", "ccr"]
NEW_PIPELINES = ["comp_async", "comp_tdm", "comp_tdm_v2"]
TDM_PIPELINES = ["comp_tdm", "comp_tdm_v2"]
GFX1250_LDS = 320 * 1024

# Kernel counts per (dtype, layout) for the full gfx1250 config: total and per
# new pipeline. Pins the sweep so any config or validator drift is noticed.
# comp_async is rcr-only on gfx1250, so the rrr rows have no comp_async. The
# pads sweep [false, true]: comp_async keeps only all-True, TDM only all-False,
# and the legacy pipelines keep all 8 combos.
FULL_COUNTS = {
    ("fp16", "rcr"): (27525, 384, 174, 87),
    ("fp16", "rrr"): (27141, 0, 174, 87),
    ("bf16", "rcr"): (27525, 384, 174, 87),
    ("bf16", "rrr"): (27141, 0, 174, 87),
    # fp8/bf8 comp_async keeps only warp_tile_k=128 (the 16x16x64 set is gated).
    ("fp8", "rcr"): (46908, 288, 360, 180),
    ("fp8", "rrr"): (46620, 0, 360, 180),
    ("bf8", "rcr"): (46908, 288, 360, 180),
    ("bf8", "rrr"): (46620, 0, 360, 180),
}


def _c_dtype(dtype):
    return "fp16" if dtype in ("fp8", "bf8") else dtype


def _builder(gpu_target, dtype, layout, config, cls=None, workdir=None):
    cls = cls or gub.GemmUniversalKernelBuilder
    workdir = Path(workdir or tempfile.mkdtemp(prefix="gu_gfx1250_"))
    return cls("gemm_universal", workdir, gpu_target, dtype, layout, str(config))


def _kernels(gpu_target, dtype, layout, config, cls=None):
    tmp = tempfile.mkdtemp(prefix="gu_gfx1250_")
    try:
        return _builder(
            gpu_target, dtype, layout, config, cls, tmp
        )._get_sampled_kernel_list()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def setUpModule():
    # The shared warp-tile table has no gfx1250 row and warns once per tile.
    logging.disable(logging.WARNING)


def tearDownModule():
    logging.disable(logging.NOTSET)


class TestWarpTileRow(unittest.TestCase):
    def test_matches_dispatcher_arch_specs(self):
        row = WARP_TILE_SUPPORTED_COMBINATIONS["gfx1250"]
        for dtype, tiles in gub.GFX1250_WARP_TILES.items():
            key = f"{dtype}_{dtype}_fp32"
            self.assertEqual(sorted(map(list, tiles)), sorted(row[key]), dtype)
        self.assertEqual({k.split("_")[0] for k in row}, set(gub.GFX1250_WARP_TILES))

    def test_row_not_applied_to_other_arches(self):
        # 32x32x16 is not a gfx1250 WMMA tile but is valid on gfx942/gfx950.
        for arch in ["gfx942", "gfx950"]:
            b = _builder(arch, "fp16", "rcr", _SHARED_CI_CONFIG)
            base = _builder(arch, "fp16", "rcr", _SHARED_CI_CONFIG, GemmKernelBuilder)
            args = (256, 256, 64, 2, 2, 1, 32, 32, 16, "compv3")
            self.assertEqual(
                b._validate_tile_config(*args), base._validate_tile_config(*args)
            )
        b = _builder("gfx1250", "fp16", "rcr", _CI_CONFIG)
        self.assertFalse(
            b._validate_tile_config(256, 256, 64, 2, 2, 1, 32, 32, 16, "compv3")
        )
        self.assertFalse(
            b._validate_tile_config(128, 128, 64, 2, 2, 1, 16, 16, 64, "compv3")
        )
        self.assertTrue(
            b._validate_tile_config(128, 128, 64, 2, 2, 1, 16, 16, 32, "compv3")
        )
        b = _builder("gfx1250:xnack-", "fp8", "rcr", _CI_CONFIG)
        self.assertFalse(
            b._validate_tile_config(128, 128, 64, 2, 2, 1, 16, 16, 32, "compv3")
        )
        self.assertTrue(
            b._validate_tile_config(128, 128, 128, 2, 2, 1, 16, 16, 128, "compv3")
        )

    def test_other_arch_enumeration_unchanged(self):
        for arch in ["gfx942", "gfx950"]:
            for dtype in DTYPES:
                new = _kernels(arch, dtype, "rcr", _SHARED_CI_CONFIG)
                old = _kernels(arch, dtype, "rcr", _SHARED_CI_CONFIG, GemmKernelBuilder)
                self.assertEqual([k["name"] for k in new], [k["name"] for k in old])


class _ConfigLintMixin:
    CONFIG = None

    def _check_kernel(self, dtype, layout, k):
        t = k["tile_config"]
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = k[
            "trait_combo"
        ]
        name = k["name"]
        self.assertIn(
            [t["warp_tile_m"], t["warp_tile_n"], t["warp_tile_k"]],
            list(gub.GFX1250_WARP_TILES[dtype]),
            name,
        )
        self.assertTrue(
            vu.is_tile_config_valid(
                t["tile_m"],
                t["tile_n"],
                t["tile_k"],
                t["warp_m"],
                t["warp_n"],
                t["warp_k"],
                t["warp_tile_m"],
                t["warp_tile_n"],
                t["warp_tile_k"],
                dtype,
                dtype,
                _c_dtype(dtype),
                pipeline,
                layout,
                "gfx1250",
                "gemm_universal",
            ),
            name,
        )
        self.assertTrue(
            vu.is_trait_combination_valid(
                pipeline,
                epilogue,
                scheduler,
                persistent,
                "gemm_universal",
                layout,
                pad_m=pad_m,
                pad_n=pad_n,
                pad_k=pad_k,
            ),
            name,
        )
        if pipeline in TDM_PIPELINES:
            self.assertEqual(
                (epilogue, scheduler, persistent), ("tdm", "intrawave", False), name
            )
            self.assertEqual((pad_m, pad_n, pad_k), (False, False, False), name)
            c_bytes = t["tile_m"] * t["tile_n"] * vu.element_size(_c_dtype(dtype))
            self.assertLessEqual(c_bytes, GFX1250_LDS, name)
        else:
            self.assertNotEqual(epilogue, "tdm", name)
        if pipeline == "comp_async":
            self.assertEqual(layout[:2], "rc", name)
            self.assertEqual((pad_m, pad_n, pad_k), (True, True, True), name)
            if dtype in ("fp8", "bf8"):
                self.assertEqual(t["warp_tile_k"], 128, name)
        if pipeline == "comp_tdm_v2":
            self.assertEqual(t["warp_m"] * t["warp_n"] * t["warp_k"], 4, name)
        if pipeline in NEW_PIPELINES + ["compv4"]:
            # Double-buffered: two A+B stages must fit in the 320K LDS.
            esz = vu.element_size(dtype)
            a = t["tile_m"] * t["tile_k"] * esz
            b = t["tile_n"] * t["tile_k"] * esz
            if pipeline in TDM_PIPELINES:
                layer = max(1, 256 // (t["tile_k"] * esz))
                a += max(0, t["tile_m"] // layer - 1) * 16
                b += max(0, t["tile_n"] // layer - 1) * 16
            self.assertLessEqual(2 * (a + b), GFX1250_LDS, name)

    def _lint(self, dtype, layout):
        kernels = _kernels("gfx1250", dtype, layout, self.CONFIG)
        for k in kernels:
            self._check_kernel(dtype, layout, k)
        return kernels


class TestCiConfig(_ConfigLintMixin, unittest.TestCase):
    CONFIG = _CI_CONFIG

    def test_schema_matches_shared_ci_config(self):
        new = json.loads(_CI_CONFIG.read_text())
        shared = json.loads(_SHARED_CI_CONFIG.read_text())
        self.assertEqual(set(new), set(shared))
        for section in ("tile_config", "trait_config"):
            self.assertEqual(set(new[section]), set(shared[section]), section)

    def test_every_dtype_layout_has_each_new_pipeline(self):
        for dtype in DTYPES:
            for layout in LAYOUTS:
                kernels = self._lint(dtype, layout)
                counts = collections.Counter(k["trait_combo"][0] for k in kernels)
                # comp_async keeps only pads True; TDM keeps only pads False;
                # comp_async is rcr-only on gfx1250, and fp8/bf8 comp_async
                # needs warp_tile_k=128, which the CI config (tile_k=64) lacks.
                expected = {"comp_tdm": 1, "comp_tdm_v2": 1}
                if layout[:2] == "rc" and dtype not in ("fp8", "bf8"):
                    expected["comp_async"] = 1
                self.assertEqual(counts, expected, (dtype, layout))


class TestFullConfig(_ConfigLintMixin, unittest.TestCase):
    CONFIG = _FULL_CONFIG

    def test_schema_matches_shared_config(self):
        new = json.loads(_FULL_CONFIG.read_text())
        shared = json.loads((_GEMM_DIR / "configs" / "default_config.json").read_text())
        self.assertEqual(set(new), set(shared))
        for section in ("tile_config", "trait_config"):
            self.assertEqual(set(new[section]), set(shared[section]), section)

    def test_full_dtype_layout_set(self):
        for dtype in DTYPES:
            for layout in LAYOUTS:
                kernels = self._lint(dtype, layout)
                counts = collections.Counter(k["trait_combo"][0] for k in kernels)
                for p in TDM_PIPELINES:
                    self.assertGreater(counts[p], 0, (dtype, layout, p))
                if layout[:2] == "rc":
                    self.assertGreater(counts["comp_async"], 0, (dtype, layout))
                else:
                    self.assertEqual(counts["comp_async"], 0, (dtype, layout))
                expected = FULL_COUNTS.get((dtype, layout))
                if expected:
                    got = (
                        len(kernels),
                        counts["comp_async"],
                        counts["comp_tdm"],
                        counts["comp_tdm_v2"],
                    )
                    self.assertEqual(got, expected, (dtype, layout))

    def test_new_pipelines_absent_on_other_arches(self):
        for arch in ["gfx942", "gfx950"]:
            kernels = _kernels(arch, "fp16", "rcr", _FULL_CONFIG)
            pipelines = {k["trait_combo"][0] for k in kernels}
            self.assertFalse(pipelines & set(NEW_PIPELINES), arch)
            self.assertFalse(any(k["trait_combo"][1] == "tdm" for k in kernels), arch)


_GOLDEN_TILE = {
    "tile_m": 128,
    "tile_n": 128,
    "tile_k": 64,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 16,
    "warp_tile_n": 16,
    "warp_tile_k": 32,
}


class TestTileEngineGolden(unittest.TestCase):
    def _instance(self, pipeline, epilogue, pad=False, layout="rcr"):
        b = _builder("gfx1250", "fp16", layout, _CI_CONFIG)
        trait = (pipeline, epilogue, "intrawave", pad, pad, pad, False)
        return b._generate_kernel_instance(_GOLDEN_TILE, trait)

    def test_tdm_with_pad_raises(self):
        for pipeline in TDM_PIPELINES:
            with self.assertRaisesRegex(ValueError, "TDM bounds-clips"):
                self._instance(pipeline, "tdm", pad=True)

    def test_comp_async_non_rc_layout_raises(self):
        for layout in ["rrr", "crr", "ccr"]:
            with self.assertRaisesRegex(ValueError, "A row-major and B col-major"):
                self._instance("comp_async", "cshuffle", pad=True, layout=layout)
        # TDM is not layout-gated.
        name, _code = self._instance("comp_tdm", "tdm", layout="rrr")
        self.assertIn("_comp_tdm_tdm_intrawave_", name)

    def test_comp_async_without_pad_raises(self):
        with self.assertRaisesRegex(ValueError, "requires pad_m=pad_n=pad_k=True"):
            self._instance("comp_async", "cshuffle")

    def test_comp_async_with_pad_accepted(self):
        name, _code = self._instance("comp_async", "cshuffle", pad=True)
        self.assertIn("_comp_async_cshuffle_intrawave_True_True_True_", name)

    def test_comp_tdm(self):
        name, code = self._instance("comp_tdm", "tdm")
        self.assertIn("_comp_tdm_tdm_intrawave_", name)
        self.assertIn('#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"', code)
        self.assertIn("ck_tile::GemmPipelineAgBgCrCompTDMV1<", code)
        self.assertIn("ck_tile::TdmEpilogue<", code)
        self.assertNotIn("CShuffleEpilogue<", code)
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", code)
        self.assertIn("args.k_batch != 1", code)
        self.assertIn("TDM pipeline requires k_batch==1", code)

    def test_comp_tdm_v2(self):
        name, code = self._instance("comp_tdm_v2", "tdm")
        self.assertIn("_comp_tdm_v2_tdm_intrawave_", name)
        self.assertIn("ck_tile::GemmPipelineAgBgCrCompTDMV2<", code)
        self.assertIn("ck_tile::TdmEpilogue<", code)
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", code)
        self.assertIn("args.k_batch != 1", code)

    def test_comp_async(self):
        name, code = self._instance("comp_async", "cshuffle", pad=True)
        self.assertIn("_comp_async_cshuffle_intrawave_", name)
        self.assertIn("ck_tile::GemmPipelineAgBgCrCompAsync<", code)
        self.assertIn("CShuffleEpilogue<", code)
        self.assertNotIn("TdmEpilogue", code)
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", code)
        self.assertNotIn("TDM pipeline requires k_batch==1", code)

    def test_gen_single_cli_multi_token_pipeline(self):
        tmp = Path(tempfile.mkdtemp(prefix="gu_gfx1250_cli_"))
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(_BUILDER),
                    "--working_path",
                    str(tmp),
                    "--gpu_target",
                    "gfx1250",
                    "--datatype",
                    "fp16",
                    "--layout",
                    "rcr",
                    "--config_json",
                    str(_CI_CONFIG),
                    "--gen_single",
                    "--kernel_name",
                    "gemm_universal_fp16_rcr_comp_tdm_v2",
                    "--tile_config",
                    "128x128x64_2x2x1_16x16x32",
                    "--trait_combo",
                    "comp_tdm_v2_tdm_intrawave_False_False_False_False",
                ],
                check=True,
                capture_output=True,
                cwd=str(_HERE),
            )
            headers = list(tmp.glob("*.hpp"))
            self.assertEqual(len(headers), 1, [h.name for h in tmp.iterdir()])
            code = headers[0].read_text()
            self.assertIn("comp_tdm_v2_tdm_intrawave", headers[0].name)
            self.assertIn("ck_tile::GemmPipelineAgBgCrCompTDMV2<", code)
            self.assertIn("ck_tile::TdmEpilogue<", code)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestDispatcherGolden(unittest.TestCase):
    """The dispatcher universal (standard) path generates the same kernels."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gu_gfx1250_dispatcher_"))
        ci = json.loads(_CI_CONFIG.read_text())
        tile = {
            k: (v["values"] if isinstance(v, dict) else v)
            for k, v in ci["tile_config"].items()
        }
        tile["warp_tile_k"] = [32]
        cfg = {"tile_config": tile, "trait_config": dict(ci["trait_config"])}
        for key, value in cfg["trait_config"].items():
            if isinstance(value, dict):
                cfg["trait_config"][key] = value["values"]
        cls.cfg = cls.tmp / "cfg.json"
        cls.cfg.write_text(json.dumps(cfg))
        cls.out = cls.tmp / "out"
        subprocess.run(
            [
                sys.executable,
                str(_CODEGEN_DIR / "unified_gemm_codegen.py"),
                "--output-dir",
                str(cls.out),
                "--datatype",
                "fp16",
                "--layout",
                "rcr",
                "--gpu-target",
                "gfx1250",
                "--config",
                str(cls.cfg),
                "--variants",
                "standard",
                "--no-parallel",
            ],
            check=True,
            capture_output=True,
            cwd=str(_CODEGEN_DIR),
        )
        cls.names = sorted(p.name for p in cls.out.glob("*.hpp"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _header(self, marker, pads="False_False_False"):
        # comp_async keeps only pads True; TDM keeps only pads False.
        matches = [
            n for n in self.names if marker in n and f"_intrawave_{pads}_False_" in n
        ]
        self.assertEqual(len(matches), 1, (marker, self.names))
        name = matches[0]
        wrapper = self.out / "dispatcher_wrappers" / f"dispatcher_wrapper_{name}"
        return (self.out / name).read_text(), wrapper.read_text()

    def test_kernel_set(self):
        # comp_async: pads True only; comp_tdm / comp_tdm_v2: pads False only.
        self.assertEqual(len(self.names), 3, self.names)
        counts = collections.Counter(
            p
            for n in self.names
            for p in ("_comp_async_", "_comp_tdm_tdm_", "_comp_tdm_v2_tdm_")
            if p in n
        )
        self.assertEqual(
            counts, {"_comp_async_": 1, "_comp_tdm_tdm_": 1, "_comp_tdm_v2_tdm_": 1}
        )
        for n in self.names:
            if "_tdm_intrawave_" in n:
                self.assertIn("_intrawave_False_False_False_False_", n)

    def test_comp_tdm(self):
        code, wrapper = self._header("_comp_tdm_tdm_")
        self.assertIn("GemmPipelineAgBgCrCompTDMV1<", code)
        self.assertIn("TdmEpilogue<", code)
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", code)
        self.assertIn("args.k_batch != 1", code)
        self.assertIn("Pipeline::CompTDMV1", wrapper)
        self.assertIn("Epilogue::Tdm", wrapper)

    def test_comp_tdm_v2(self):
        code, wrapper = self._header("_comp_tdm_v2_tdm_")
        self.assertIn("GemmPipelineAgBgCrCompTDMV2<", code)
        self.assertIn("TdmEpilogue<", code)
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", code)
        self.assertIn("args.k_batch != 1", code)
        self.assertIn("Epilogue::Tdm", wrapper)

    def test_comp_async(self):
        code, _wrapper = self._header("_comp_async_", pads="True_True_True")
        self.assertIn("GemmPipelineAgBgCrCompAsync<", code)
        self.assertNotIn("TdmEpilogue", code)
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", code)

    def test_expand_sweep_matches_tile_engine(self):
        from gemm_utils import expand_sweep

        for dtype in DTYPES:
            for layout in LAYOUTS:
                bridge = {
                    (c.pipeline, c.epilogue, c.pad_m, c.pad_n, c.pad_k)
                    for c in expand_sweep(str(_CI_CONFIG), "gfx1250", dtype, layout)
                }
                te = {
                    k["trait_combo"][:2] + k["trait_combo"][3:6]
                    for k in _kernels("gfx1250", dtype, layout, _CI_CONFIG)
                }
                self.assertEqual(bridge, te, (dtype, layout))
        self.assertEqual(expand_sweep(str(_CI_CONFIG), "gfx942", "fp16", "rcr"), [])


if __name__ == "__main__":
    unittest.main()
