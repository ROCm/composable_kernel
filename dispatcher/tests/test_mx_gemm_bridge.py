#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the microscaling (mx) GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the codegen-JSON projection, the dtype/layout/warp-tile
validity gate, the e8m0 scale codec, the fp8/fp4 quantization round-trips, and the numpy
microscaled reference. Also covers build entry points, target normalization, and
Tile Engine configuration validation. No GPU or hipcc is required.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

_DISP = Path(__file__).resolve().parent.parent
_CK = _DISP.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

import mx_gemm_utils as mx  # noqa: E402
import unified_mx_gemm_codegen as codegen  # noqa: E402
from dispatcher_common import arch_feature_defines, unified_framework_flags  # noqa: E402

from mx_gemm_utils import (  # noqa: E402
    SCALE_BLOCK,
    E8M0_ONE,
    MxGemmKernelConfig,
    MxGemmProblem,
    default_fp8_config,
    default_fp4_config,
    e8m0_to_float,
    float_to_e8m0,
    quantize_fp8,
    dequantize_fp8,
    quantize_fp4_packed,
    dequantize_fp4_packed,
    mx_gemm_reference,
)


class TestConfigName(unittest.TestCase):
    def test_fallback_name_prefix(self):
        cfg = default_fp8_config()
        self.assertTrue(cfg._fallback_name().startswith("mx_gemm_fp8_rcr_"))

    def test_fallback_name_encodes_tiles(self):
        cfg = MxGemmKernelConfig(
            datatype="fp4",
            tile_m=64,
            tile_n=128,
            tile_k=256,
            warp_m=1,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=128,
        )
        name = cfg._fallback_name()
        self.assertIn("_fp4_rcr_", name)
        self.assertIn("64x128x256", name)
        self.assertIn("1x2x1", name)
        self.assertIn("16x16x128", name)
        self.assertNotIn(" ", name)

    def test_persistent_suffix_only_when_set(self):
        self.assertNotIn("True", default_fp8_config()._fallback_name())
        cfg = default_fp8_config()
        cfg.persistent = True
        self.assertIn("True", cfg._fallback_name())


class TestCodegenJson(unittest.TestCase):
    def test_projection_roundtrip(self):
        cfg = MxGemmKernelConfig(
            datatype="fp8",
            tile_m=128,
            tile_n=128,
            tile_k=128,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=128,
            k_block_per_cu=3,
            # Pin the arch so to_codegen_config() does not shell out to rocminfo;
            # keeps this a CPU-only test on non-ROCm runners.
            gpu_target="gfx950",
        )
        j = cfg.to_codegen_config()
        self.assertEqual(j["datatype"], "fp8")
        self.assertEqual(j["layout"], "rcr")
        self.assertEqual(j["tile_config"]["tile_k"], 128)
        self.assertEqual(j["tile_config"]["warp_tile_k"], 128)
        self.assertEqual(j["k_block_per_cu"], 3)


class TestValidity(unittest.TestCase):
    def test_default_configs_valid(self):
        self.assertTrue(default_fp8_config().is_valid())
        self.assertTrue(default_fp4_config().is_valid())

    def test_non_rcr_rejected(self):
        cfg = default_fp8_config()
        cfg.layout = "rrr"
        self.assertFalse(cfg.is_valid())

    def test_bad_dtype_rejected(self):
        cfg = default_fp8_config()
        cfg.datatype = "bf16"
        self.assertFalse(cfg.is_valid())

    def test_bf8_rejected(self):
        # Old-TE argparse (choices=["fp4","fp8"]) + validate_gemm_mx never
        # compile bf8/e5m2 for mx_gemm, so the bridge must reject it too. Assert
        # both the config-level gate (is_valid) and, when importable, the
        # codegen-level gate (_validate) refuse dtype="bf8".
        cfg = default_fp8_config()
        cfg.datatype = "bf8"
        # Pin the arch so to_codegen_config() below stays CPU-only (no rocminfo).
        cfg.gpu_target = "gfx950"
        self.assertFalse(cfg.is_valid())

        try:
            from unified_mx_gemm_codegen import _validate  # noqa: E402
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"codegen import unavailable: {exc}")
        with self.assertRaises(Exception):
            _validate(cfg.to_codegen_config())

    def test_wrong_warp_tile_rejected(self):
        cfg = default_fp8_config()
        cfg.warp_tile_k = 64
        self.assertFalse(cfg.is_valid())

    def test_indivisible_tile_rejected(self):
        cfg = default_fp8_config()
        cfg.tile_m = 100  # not a multiple of warp_m * warp_tile_m (2*16=32)
        self.assertFalse(cfg.is_valid())


class TestProblem(unittest.TestCase):
    def test_scale_k_and_flops(self):
        p = MxGemmProblem(M=64, N=128, K=256)
        self.assertEqual(p.scale_k, 256 // SCALE_BLOCK)
        self.assertEqual(p.flops, 2 * 64 * 128 * 256)

    def test_k_not_multiple_of_32_raises(self):
        with self.assertRaises(ValueError):
            MxGemmProblem(M=32, N=32, K=48)


class TestE8m0Codec(unittest.TestCase):
    def test_one_is_byte_127(self):
        self.assertEqual(int(float_to_e8m0(np.float32(1.0))), E8M0_ONE)
        self.assertEqual(float(e8m0_to_float(E8M0_ONE)), 1.0)

    def test_power_of_two_roundtrip(self):
        for s in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            b = float_to_e8m0(np.float32(s))
            self.assertAlmostEqual(float(e8m0_to_float(b)), s, places=6)

    def test_255_is_nan(self):
        with np.errstate(over="ignore"):
            self.assertTrue(np.isnan(float(e8m0_to_float(255))))

    def test_non_positive_raises(self):
        # Contract is a strictly-positive power-of-two; a 0.0/negative scale is a
        # caller bug and must fail loudly rather than silently encode 1.0.
        for bad in (0.0, -1.0, -0.0):
            with self.assertRaises(ValueError):
                float_to_e8m0(np.float32(bad))
        with self.assertRaises(ValueError):
            float_to_e8m0(np.array([1.0, 0.0, 2.0], np.float32))

    def test_non_finite_raises(self):
        with np.errstate(invalid="ignore"):
            for bad in (np.inf, np.nan):
                with self.assertRaises(ValueError):
                    float_to_e8m0(np.float32(bad))


class TestFp8Codec(unittest.TestCase):
    def test_known_bytes(self):
        self.assertEqual(int(quantize_fp8(np.array([1.0], np.float32))[0]), 0x38)
        self.assertEqual(int(quantize_fp8(np.array([-2.0], np.float32))[0]), 0xC0)

    def test_roundtrip_grid(self):
        grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], np.float32)
        vals = np.tile(grid, (4, 1))
        self.assertTrue(np.array_equal(dequantize_fp8(quantize_fp8(vals)), vals))

    def test_off_grid_raises(self):
        # The vectorized codec must reject values not on the exact e4m3 grid
        # instead of snapping them to a neighbour byte.
        with self.assertRaises(KeyError):
            quantize_fp8(np.array([[0.3]], np.float32))

    def test_neg_zero_collapses_to_zero_byte(self):
        self.assertEqual(int(quantize_fp8(np.array([-0.0], np.float32))[0]), 0x00)

    def test_shape_preserved_2d(self):
        vals = np.full((3, 5), 1.0, np.float32)
        self.assertEqual(quantize_fp8(vals).shape, (3, 5))


class TestFp4Codec(unittest.TestCase):
    def test_pack_two_per_byte(self):
        # one row, K=2 -> a single packed byte; low nibble even-K, high nibble odd-K.
        vals = np.array([[1.0, 2.0]], np.float32)  # codes: 1.0->2, 2.0->4
        packed = quantize_fp4_packed(vals)
        self.assertEqual(packed.shape, (1, 1))
        self.assertEqual(int(packed[0, 0]), (4 << 4) | 2)

    def test_roundtrip_grid(self):
        grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], np.float32)
        rng = np.random.default_rng(0)
        vals = rng.choice(grid, size=(4, 8)).astype(np.float32)
        packed = quantize_fp4_packed(vals)
        self.assertEqual(packed.shape, (4, 4))
        self.assertTrue(np.array_equal(dequantize_fp4_packed(packed, 8), vals))

    def test_off_grid_raises(self):
        # 2.5 exists in fp8 e4m3 but NOT in the fp4 e2m1 grid -> must reject.
        with self.assertRaises(KeyError):
            quantize_fp4_packed(np.array([[2.5, 1.0]], np.float32))


class TestReference(unittest.TestCase):
    def _inputs(self, M, N, K, seed=0):
        rng = np.random.default_rng(seed)
        grid = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], np.float32)
        A = rng.choice(grid, size=(M, K)).astype(np.float32)
        B = rng.choice(grid, size=(K, N)).astype(np.float32)
        return A, B

    def test_unit_scales_equals_plain_matmul(self):
        M, N, K = 4, 3, 32
        A, B = self._inputs(M, N, K)
        prob = MxGemmProblem(M=M, N=N, K=K)
        one = int(float_to_e8m0(np.float32(1.0)))
        sa = np.full((M, prob.scale_k), one, np.uint8)
        sb = np.full((N, prob.scale_k), one, np.uint8)
        ref = mx_gemm_reference(A, B, sa, sb, prob).astype(np.float32)
        plain = (A @ B).astype(np.float16).astype(np.float32)
        self.assertTrue(np.allclose(ref, plain, atol=1e-2))

    def test_power_of_two_scale_multiplies(self):
        M, N, K = 4, 3, 32
        A, B = self._inputs(M, N, K, seed=1)
        prob = MxGemmProblem(M=M, N=N, K=K)
        two = int(float_to_e8m0(np.float32(2.0)))
        sa = np.full((M, prob.scale_k), two, np.uint8)
        sb = np.full((N, prob.scale_k), two, np.uint8)
        ref = mx_gemm_reference(A, B, sa, sb, prob).astype(np.float32)
        # scale_a=2 and scale_b=2 -> product scaled by 4.
        expected = (4.0 * (A @ B)).astype(np.float16).astype(np.float32)
        self.assertTrue(np.allclose(ref, expected, atol=1e-1))


class TestCodegenNameContract(unittest.TestCase):
    """Optional: byte-exact name parity when the Old-TE builder is importable."""

    def test_codegen_name_matches_config(self):
        try:
            from unified_mx_gemm_codegen import kernel_name
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"codegen import unavailable: {exc}")
        cfg = default_fp8_config()
        # Pin the arch so to_codegen_config() stays CPU-only (no rocminfo).
        cfg.gpu_target = "gfx950"
        try:
            name = kernel_name(cfg.to_codegen_config())
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Old-TE builder unavailable: {exc}")
        self.assertTrue(name.startswith("mx_gemm_fp8_rcr_"))


from mx_gemm_utils import fp8_ocp_is_default_for_arch  # noqa: E402


class TestGfx1250MxEnablement(unittest.TestCase):
    def test_fp8_ocp_default_for_gfx1250(self):
        # gfx1250 defaults to OCP e4m3, so the fp8 numpy reference codec is valid.
        self.assertTrue(fp8_ocp_is_default_for_arch("gfx1250"))

    def test_default_fp8_config_accepts_gfx1250(self):
        cfg = default_fp8_config("gfx1250")
        self.assertEqual(cfg.gpu_target, "gfx1250")
        self.assertEqual(cfg.datatype, "fp8")

    def test_default_fp4_config_accepts_gfx1250(self):
        cfg = default_fp4_config("gfx1250")
        self.assertEqual(cfg.gpu_target, "gfx1250")
        self.assertEqual(cfg.datatype, "fp4")

    def test_warp_tile_is_wmma_mx_shape(self):
        # gfx1250 WMMA MX and gfx950 XDL MX share the 16x16x128 warp tile, so the
        # single mx warp-tile config is arch-portable.
        cfg = default_fp8_config("gfx1250")
        self.assertEqual(
            (cfg.warp_tile_m, cfg.warp_tile_n, cfg.warp_tile_k), (16, 16, 128)
        )

    def test_gfx1250_codegen_name_fp8(self):
        try:
            from unified_mx_gemm_codegen import kernel_name
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"codegen import unavailable: {exc}")
        cfg = default_fp8_config("gfx1250")
        try:
            name = kernel_name(cfg.to_codegen_config())
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Old-TE builder unavailable: {exc}")
        self.assertTrue(name.startswith("mx_gemm_fp8_rcr_"))
        self.assertIn("16x16x128", name)

    def test_gfx1250_codegen_name_fp4(self):
        try:
            from unified_mx_gemm_codegen import kernel_name
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"codegen import unavailable: {exc}")
        cfg = default_fp4_config("gfx1250")
        try:
            name = kernel_name(cfg.to_codegen_config())
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Old-TE builder unavailable: {exc}")
        self.assertTrue(name.startswith("mx_gemm_fp4_rcr_"))


class TestMxArchitectureKernels(unittest.TestCase):
    def test_gfx1250_filters_persistent_and_k_padding(self):
        config = _CK / "tile_engine/ops/gemm/mx_gemm/configs/default_config.json"
        with tempfile.TemporaryDirectory() as tmp:
            for arch in ("gfx950", "gfx1250", "gfx1250:xnack-"):
                builder = codegen._load_mx_builder()(
                    "mx_gemm", tmp, arch, "fp8", "rcr", str(config)
                )
                builder.config["trait_config"]["pad_k"]["values"] = [False, True]
                traits = builder._generate_trait_combinations()
                with self.subTest(arch=arch):
                    self.assertEqual(
                        {t[0] for t in traits},
                        {"comp_async", "comp_async_eight_waves", "weight_preshuffle"},
                    )
                    flags = {(t[5], t[6]) for t in traits}
                    self.assertEqual(
                        flags,
                        {(False, False), (False, True), (True, False), (True, True)}
                        if arch == "gfx950"
                        else {(False, False)},
                    )

    def test_build_arch_matches_config(self):
        from unittest.mock import patch
        from mx_gemm_utils import setup_multiple_mx_gemm_dispatchers
        import tempfile

        cfg = default_fp8_config("gfx1250")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "mx_gemm_utils._get_arch",
                side_effect=AssertionError("unexpected detection"),
            ),
            patch(
                "mx_gemm_utils._generate_kernel", return_value=Path(tmp) / "kernel.hpp"
            ),
            patch("mx_gemm_utils._compile_kernel", return_value=True) as compile_kernel,
        ):
            result = setup_multiple_mx_gemm_dispatchers(
                [cfg], output_dir=Path(tmp), parallel=False
            )
            self.assertIsNotNone(result[0])
            self.assertEqual(compile_kernel.call_args.args[2], "gfx1250")
            with self.assertRaisesRegex(ValueError, "one architecture"):
                setup_multiple_mx_gemm_dispatchers([cfg, default_fp8_config("gfx950")])

    def test_shared_builder_parity(self):
        from unified_mx_gemm_codegen import (
            _generate,
            _make_builder,
            _trait_combo_from_cfg,
        )
        import contextlib
        import io

        for arch in ("gfx950", "gfx1250"):
            for dtype in ("fp8", "fp4"):
                with (
                    self.subTest(arch=arch, dtype=dtype),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    cfg = MxGemmKernelConfig(
                        datatype=dtype, gpu_target=arch
                    ).to_codegen_config()
                    name, code = _generate(cfg)
                    with _make_builder(cfg) as builder:
                        te_name, te_code = builder._generate_kernel_instance(
                            cfg["tile_config"], _trait_combo_from_cfg(cfg)
                        )
                    self.assertEqual((name, code), (te_name, te_code))
                    if arch == "gfx1250":
                        self.assertIn("GemmPipelineAgBgCrCompTDMV1", code)
                        self.assertIn("MxGemmPipelineProblem", code)
                        self.assertIn("ck_tile::TdmEpilogue<", code)
                    else:
                        self.assertIn("GemmPipelineAgBgCrCompAsync", code)
                        self.assertIn("ck_tile::CShuffleEpilogue<", code)

    def test_architecture_and_trait_rejections(self):
        from unified_mx_gemm_codegen import _validate
        from dataclasses import replace

        cfg = default_fp8_config("gfx1250")
        for changes in (
            {"gpu_target": "gfx1200"},
            {"pipeline": "mx_flatmm"},
            {"epilogue": "cshuffle"},
            {"persistent": True},
            {"pad_k": True},
            {"warp_m": 0},
            {"warp_tile_m": 32},
            {"scheduler": "interwave"},
        ):
            with self.subTest(changes=changes):
                invalid = replace(cfg, **changes)
                self.assertFalse(invalid.is_valid())
                with self.assertRaises(ValueError):
                    _validate(invalid.to_codegen_config())

    def test_gfx1250_ci_config_enumerates_real_kernels(self):
        from unified_mx_gemm_codegen import _load_mx_builder
        import tempfile

        config = (
            _DISP.parent
            / "tile_engine/ops/gemm/mx_gemm/configs/default_ci_config_gfx1250.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            for dtype in ("fp8", "fp4"):
                builder = _load_mx_builder()(
                    "mx_gemm",
                    Path(tmp),
                    "gfx1250",
                    dtype,
                    "rcr",
                    config_json=str(config),
                )
                tiles = builder._get_tile_configs()
                traits = builder._generate_trait_combinations()
                self.assertEqual(len(tiles), 4)
                self.assertEqual(
                    traits,
                    [
                        (pipeline, "tdm", "intrawave", False, False, False, False)
                        for pipeline in ("comp_tdm", "comp_tdm_v2")
                    ],
                )

    def test_all_native_mx_pipeline_defaults_and_codegen(self):
        from unified_mx_gemm_codegen import (
            _generate,
            _make_builder,
            _trait_combo_from_cfg,
        )
        import contextlib
        import io

        pipelines = {
            "gfx950": {
                "comp_async": "GemmPipelineAgBgCrCompAsync",
                "comp_async_eight_waves": "GemmPipelineAgBgCrCompAsyncEightWaves",
                "weight_preshuffle": "MXGemmPreshufflePipelineAGmemBGmemCRegV1",
            },
            "gfx1250": {
                "comp_tdm": "GemmPipelineAgBgCrCompTDMV1",
                "comp_tdm_v2": "GemmPipelineAgBgCrCompTDMV2",
                "comp_async": "GemmPipelineAgBgCrCompAsync",
                "comp_async_eight_waves": "GemmPipelineAgBgCrCompAsyncEightWaves",
                "weight_preshuffle": "MXGemmPreshufflePipelineAGmemBGmemCRegV1",
            },
        }
        for arch, implementations in pipelines.items():
            for pipeline, implementation in implementations.items():
                for make_config in (default_fp4_config, default_fp8_config):
                    cfg = make_config(arch, pipeline)
                    with self.subTest(arch=arch, pipeline=pipeline, dtype=cfg.datatype):
                        self.assertTrue(cfg.is_valid())
                        config = cfg.to_codegen_config()
                        with contextlib.redirect_stdout(io.StringIO()):
                            name, code = _generate(config)
                            with _make_builder(config) as builder:
                                te_name, te_code = builder._generate_kernel_instance(
                                    config["tile_config"], _trait_combo_from_cfg(config)
                                )
                        self.assertEqual((name, code), (te_name, te_code))
                        self.assertIn(f"ck_tile::{implementation}<", code)
                        self.assertIn("MxGemmPipelineProblem", code)
                        self.assertIn(
                            "Preshuffle = true;"
                            if pipeline == "weight_preshuffle"
                            else "Preshuffle = false;",
                            code,
                        )

    def test_gfx1250_tdm_optional_warp_tiles(self):
        import contextlib
        import io
        from dataclasses import replace
        from unified_mx_gemm_codegen import (
            _generate,
            _make_builder,
            _trait_combo_from_cfg,
        )

        for pipeline in ("comp_tdm", "comp_tdm_v2"):
            for make_config, warp_n in (
                (default_fp4_config, 16),
                (default_fp4_config, 32),
                (default_fp8_config, 32),
            ):
                cfg = replace(
                    make_config("gfx1250", pipeline), warp_tile_m=32, warp_tile_n=warp_n
                )
                with self.subTest(pipeline=pipeline, dtype=cfg.datatype, warp_n=warp_n):
                    self.assertTrue(cfg.is_valid())
                    config = cfg.to_codegen_config()
                    with contextlib.redirect_stdout(io.StringIO()):
                        name, code = _generate(config)
                        with _make_builder(config) as builder:
                            builder.config["tile_config"] = {
                                key: {"values": [value]}
                                for key, value in config["tile_config"].items()
                            }
                            builder.config["trait_config"] = {
                                key: {"values": [config[key]]}
                                for key in (
                                    "pipeline",
                                    "epilogue",
                                    "scheduler",
                                    "pad_m",
                                    "pad_n",
                                    "pad_k",
                                    "persistent",
                                )
                            }
                            sampled = builder._get_sampled_kernel_list()
                            self.assertEqual(len(sampled), 1)
                            self.assertEqual(
                                sampled[0]["tile_config"], config["tile_config"]
                            )
                            native = builder._generate_kernel_instance(
                                config["tile_config"], _trait_combo_from_cfg(config)
                            )
                    self.assertEqual((name, code), native)
                    self.assertIn(f"_32x{warp_n}x128", name)
                    self.assertIn("WarpTileM = 32;", code)
                    self.assertIn(f"WarpTileN = {warp_n};", code)
                    self.assertIn("MxGemmPipelineProblem", code)

    def test_32x32_rejected_outside_gfx1250_tdm(self):
        from dataclasses import replace
        from unified_mx_gemm_codegen import _validate

        for arch in ("gfx950", "gfx1250"):
            for pipeline in (
                "comp_async",
                "comp_async_eight_waves",
                "weight_preshuffle",
            ):
                for make_config in (default_fp4_config, default_fp8_config):
                    cfg = replace(
                        make_config(arch, pipeline), warp_tile_m=32, warp_tile_n=32
                    )
                    with self.subTest(arch=arch, pipeline=pipeline, dtype=cfg.datatype):
                        self.assertFalse(cfg.is_valid())
                        diagnostic = (
                            "gfx950:.*16, 16, 128" if arch == "gfx950" else arch
                        )
                        with self.assertRaisesRegex(ValueError, diagnostic):
                            _validate(cfg.to_codegen_config())

    def test_fp4_32x16_rejected_on_gfx950(self):
        from dataclasses import replace
        from unified_mx_gemm_codegen import _validate

        cfg = replace(default_fp4_config("gfx950"), warp_tile_m=32)
        self.assertFalse(cfg.is_valid())
        with self.assertRaisesRegex(ValueError, "gfx950:.*16, 16, 128"):
            _validate(cfg.to_codegen_config())

    def test_pipeline_architecture_and_tile_rejections(self):
        from dataclasses import replace
        from unified_mx_gemm_codegen import _validate

        invalid = [
            replace(default_fp8_config("gfx1250", "comp_async"), warp_m=1),
            default_fp8_config("gfx950", "comp_tdm_v2"),
            replace(default_fp8_config("gfx1250", "weight_preshuffle"), tile_k=128),
            replace(default_fp4_config("gfx950", "comp_async"), tile_m=192, tile_n=512),
            replace(
                default_fp8_config("gfx950", "comp_async"),
                tile_m=64,
                tile_n=256,
                tile_k=256,
            ),
            replace(default_fp8_config("gfx950", "comp_async_eight_waves"), warp_m=2),
            replace(default_fp4_config("gfx950", "comp_async_eight_waves"), tile_n=384),
            replace(
                default_fp4_config("gfx950", "comp_async_eight_waves"),
                tile_n=512,
                tile_k=256,
            ),  # Native LDS allocation is 168960 bytes, above gfx950's 163840.
            replace(default_fp4_config("gfx950", "comp_async_eight_waves"), tile_k=384),
            replace(default_fp8_config("gfx950", "weight_preshuffle"), warp_m=2),
            replace(default_fp4_config("gfx950", "weight_preshuffle"), tile_n=256),
            replace(default_fp8_config("gfx950", "weight_preshuffle"), tile_k=128),
            replace(default_fp8_config("gfx1250", "comp_tdm_v2"), persistent=True),
            replace(default_fp8_config("gfx1250", "comp_tdm_v2"), pad_k=True),
            replace(default_fp8_config("gfx1250", "comp_tdm_v2"), epilogue="cshuffle"),
        ]
        for cfg in invalid:
            with self.subTest(cfg=cfg):
                self.assertFalse(cfg.is_valid())
                with self.assertRaises(ValueError):
                    _validate(cfg.to_codegen_config())

    def test_gfx1250_default_covers_both_tdm_pipelines(self):
        from collections import Counter
        from unified_mx_gemm_codegen import _load_mx_builder
        import tempfile

        config = (
            _DISP.parent
            / "tile_engine/ops/gemm/mx_gemm/configs/default_config_gfx1250.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            for dtype in ("fp4", "fp8"):
                builder = _load_mx_builder()(
                    "mx_gemm", Path(tmp), "gfx1250", dtype, "rcr", str(config)
                )
                kernels = builder._get_sampled_kernel_list()
                self.assertEqual(
                    Counter(k["trait_combo"][0] for k in kernels),
                    {"comp_tdm": 32, "comp_tdm_v2": 32},
                )

    def test_benchmark_metadata_preserves_pipeline_names(self):
        from mx_gemm_benchmark import MxGemmBenchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            benchmark = MxGemmBenchmark(tmp)
            for arch, pipelines in (
                (
                    "gfx950",
                    ("comp_async", "comp_async_eight_waves", "weight_preshuffle"),
                ),
                (
                    "gfx1250",
                    (
                        "comp_tdm",
                        "comp_tdm_v2",
                        "comp_async",
                        "comp_async_eight_waves",
                        "weight_preshuffle",
                    ),
                ),
            ):
                for pipeline in pipelines:
                    cfg = default_fp8_config(arch, pipeline)
                    info = benchmark.extract_kernel_info(
                        Path(tmp) / ("benchmark_" + cfg.name)
                    )
                    with self.subTest(pipeline=pipeline):
                        self.assertEqual(info["pipeline"], pipeline)
                        self.assertEqual(info["scheduler"], "intrawave")
                        self.assertEqual(
                            info["epilogue"],
                            "tdm"
                            if pipeline in ("comp_tdm", "comp_tdm_v2")
                            else "cshuffle",
                        )


class TestMxLdsCapacity(unittest.TestCase):
    def test_tdm_output_lds_boundary(self):
        for pipeline in ("comp_tdm", "comp_tdm_v2"):
            for factory in (mx.default_fp4_config, mx.default_fp8_config):
                for n, expected in ((320, True), (352, False), (512, False)):
                    cfg = replace(
                        factory("gfx1250", pipeline), tile_m=512, tile_n=n, tile_k=128
                    )
                    with self.subTest(pipeline=pipeline, dtype=cfg.datatype, n=n):
                        self.assertEqual(cfg.is_valid(), expected)
                        if expected:
                            codegen._validate(cfg.to_codegen_config())
                        else:
                            with self.assertRaises(ValueError):
                                codegen._validate(cfg.to_codegen_config())

    def test_tile_engine_capacity_matches_dispatcher(self):
        from unified_mx_gemm_codegen import _load_mx_builder
        from arch_specs_generated import LDS_TOTAL_CAPACITY_BY_ARCH

        _load_mx_builder()
        from gemm_validation_utils import LDS_SIZE_MAP

        self.assertEqual(LDS_SIZE_MAP["gfx1250"], 320 * 1024)
        for arch, capacity in LDS_SIZE_MAP.items():
            with self.subTest(arch=arch):
                self.assertEqual(capacity, LDS_TOTAL_CAPACITY_BY_ARCH[arch])

    def test_all_mx_pipelines_use_gfx1250_capacity(self):
        from dataclasses import replace
        from unified_mx_gemm_codegen import _validate
        from gemm_validation_utils import is_tile_config_valid, validate_lds_capacity

        # Each tile needs more than 64 KiB but fits in gfx1250's 320 KiB,
        # including both buffers and async descriptor padding.
        for pipeline in (
            "comp_tdm",
            "comp_tdm_v2",
            "comp_async",
            "comp_async_eight_waves",
            "weight_preshuffle",
        ):
            for make_config in (default_fp4_config, default_fp8_config):
                cfg = replace(
                    make_config("gfx1250", pipeline),
                    tile_m=256,
                    tile_n=512 if pipeline == "weight_preshuffle" else 256,
                    tile_k=512 if make_config == default_fp4_config else 256,
                )
                for factor, expected in ((1, True), (4, False)):
                    candidate = replace(cfg, tile_k=cfg.tile_k * factor)
                    with self.subTest(
                        pipeline=pipeline, dtype=cfg.datatype, factor=factor
                    ):
                        self.assertEqual(candidate.is_valid(), expected)
                        if expected:
                            _validate(candidate.to_codegen_config())
                        else:
                            with self.assertRaises(ValueError):
                                _validate(candidate.to_codegen_config())
                        for arch in ("gfx1250", "gfx1250:xnack-"):
                            valid, error = validate_lds_capacity(
                                candidate.tile_m,
                                candidate.tile_n,
                                candidate.tile_k,
                                cfg.datatype,
                                cfg.datatype,
                                pipeline,
                                arch,
                            )
                            self.assertEqual(valid, expected, error)
                            self.assertEqual(
                                is_tile_config_valid(
                                    candidate.tile_m,
                                    candidate.tile_n,
                                    candidate.tile_k,
                                    cfg.warp_m,
                                    cfg.warp_n,
                                    cfg.warp_k,
                                    16,
                                    16,
                                    128,
                                    cfg.datatype,
                                    cfg.datatype,
                                    "fp16",
                                    pipeline,
                                    "rcr",
                                    arch,
                                    "mx_gemm",
                                ),
                                expected,
                            )

    def test_tdm_padding_is_included_at_capacity_boundary(self):
        from dataclasses import replace
        from unified_mx_gemm_codegen import _validate
        from gemm_validation_utils import validate_lds_capacity

        for pipeline in ("comp_tdm", "comp_tdm_v2"):
            for make_config in (default_fp4_config, default_fp8_config):
                for n, expected in ((320, True), (352, False)):
                    cfg = replace(
                        make_config("gfx1250", pipeline),
                        tile_m=256,
                        tile_n=n,
                        tile_k=512 if make_config == default_fp4_config else 256,
                    )
                    with self.subTest(pipeline=pipeline, dtype=cfg.datatype, n=n):
                        # Both raw tiles fit. With padding, the two buffers
                        # need 313280 bytes for N=320 and 330688 for N=352.
                        self.assertEqual(cfg.is_valid(), expected)
                        if expected:
                            _validate(cfg.to_codegen_config())
                        else:
                            with self.assertRaises(ValueError):
                                _validate(cfg.to_codegen_config())
                        for arch in ("gfx1250", "gfx1250:xnack-"):
                            valid, error = validate_lds_capacity(
                                cfg.tile_m,
                                cfg.tile_n,
                                cfg.tile_k,
                                cfg.datatype,
                                cfg.datatype,
                                pipeline,
                                arch,
                            )
                            self.assertEqual(valid, expected, error)

    def test_other_arch_double_buffer_boundary_and_unknown_arch(self):
        from unified_mx_gemm_codegen import _load_mx_builder

        _load_mx_builder()
        from gemm_validation_utils import validate_lds_capacity

        for pipeline in ("comp_tdm", "comp_tdm_v2"):
            for arch, single_buffer_bytes in (
                ("gfx950", 80 * 1024),
                ("gfx942", 32 * 1024),
                ("unknown", 32 * 1024),
            ):
                for extra, expected in ((0, True), (1, False)):
                    with self.subTest(pipeline=pipeline, arch=arch, extra=extra):
                        # FP8: (M + N) * K bytes per buffer, two buffers.
                        valid, error = validate_lds_capacity(
                            64,
                            single_buffer_bytes // 256 - 64 + extra,
                            256,
                            "fp8",
                            "fp8",
                            pipeline,
                            arch,
                        )
                        self.assertEqual(valid, expected, error)


class TestMxBuildEntryPoints(unittest.TestCase):
    def test_direct_tile_engine_listing_without_pythonpath(self):
        mx_dir = _CK / "tile_engine/ops/gemm/mx_gemm"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as tmp:
            for cwd in (mx_dir, Path(tmp)):
                with self.subTest(cwd=cwd):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(mx_dir / "mx_gemm_instance_builder.py"),
                            "--working_path",
                            tmp,
                            "--gpu_target",
                            "gfx1250:xnack-",
                            "--datatype",
                            "fp8",
                            "--layout",
                            "rcr",
                            "--list_kernels",
                            "--config_json",
                            str(mx_dir / "configs/default_ci_config_gfx1250.json"),
                        ],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual(
                        (Path(tmp) / "mx_gemm_kernel_count.txt").read_text(), "8"
                    )

    def test_suffixed_targets_match_bare_codegen_and_defaults(self):
        for arch, target in (
            ("gfx950", "gfx950:sramecc+:xnack-"),
            ("gfx1250", "gfx1250:xnack-"),
        ):
            for pipeline in (
                None,
                "comp_async",
                "comp_async_eight_waves",
                "weight_preshuffle",
            ):
                with self.subTest(arch=arch, pipeline=pipeline):
                    bare = mx.default_fp8_config(arch, pipeline)
                    suffixed = mx.default_fp8_config(target, pipeline)
                    self.assertEqual(
                        bare.to_codegen_config(), suffixed.to_codegen_config()
                    )
                    # Bypass factory normalization to exercise direct config/codegen callers.
                    raw_config = replace(bare, gpu_target=target)
                    self.assertEqual(raw_config._fallback_name(), bare._fallback_name())
                    raw = bare.to_codegen_config()
                    raw["gpu_target"] = target
                    self.assertEqual(
                        codegen._generate(raw),
                        codegen._generate(bare.to_codegen_config()),
                    )
                    self.assertEqual(codegen.kernel_name(raw), bare.name)
                    self.assertEqual(raw["gpu_target"], target)

    def test_detection_and_setup_normalize_target_suffixes(self):
        for arch, target in (
            ("gfx950", "gfx950:sramecc+:xnack-"),
            ("gfx1250", "gfx1250:xnack-"),
        ):
            with self.subTest(arch=arch):
                with patch(
                    "mx_gemm_utils.subprocess.check_output",
                    return_value=f"Name: {target}\n",
                ):
                    self.assertEqual(mx._get_arch(), arch)
                for explicit in (None, target):
                    configs = [
                        replace(mx.default_fp8_config(arch), gpu_target=target),
                        mx.default_fp8_config(arch),
                    ]
                    with (
                        tempfile.TemporaryDirectory() as tmp,
                        patch(
                            "mx_gemm_utils._compile_kernel", return_value=True
                        ) as compile_kernel,
                    ):
                        results = mx.setup_multiple_mx_gemm_dispatchers(
                            configs, Path(tmp), gfx_arch=explicit, parallel=False
                        )
                        self.assertEqual(results[0], results[1])
                        self.assertIsNotNone(results[0])
                        self.assertEqual(compile_kernel.call_count, 1)
                        self.assertEqual(compile_kernel.call_args.args[2], arch)
                        self.assertTrue(all(cfg.gpu_target == arch for cfg in configs))
        with self.assertRaisesRegex(ValueError, "one architecture"):
            mx.setup_multiple_mx_gemm_dispatchers(
                [
                    replace(
                        mx.default_fp8_config("gfx950"), gpu_target="gfx950:xnack-"
                    ),
                    replace(
                        mx.default_fp8_config("gfx1250"), gpu_target="gfx1250:xnack-"
                    ),
                ]
            )

    def test_standalone_compiler_uses_arch_feature_definitions(self):
        for arch in ("gfx950", "gfx1250"):
            with (
                self.subTest(arch=arch),
                patch("mx_gemm_utils._mx_codegen_flags", return_value=()),
                patch("mx_gemm_utils.subprocess.run") as run,
            ):
                run.return_value.returncode = 0
                self.assertTrue(
                    mx._compile_kernel(Path("kernel.hpp"), Path("kernel.so"), arch)
                )
                command = run.call_args.args[0]
                for flag in arch_feature_defines(arch) + unified_framework_flags(arch):
                    self.assertIn(flag, command)
                self.assertIn(f"--offload-arch={arch}", command)
                self.assertIn(f'-DGFX_ARCH="{arch}"', command)


if __name__ == "__main__":
    unittest.main()
