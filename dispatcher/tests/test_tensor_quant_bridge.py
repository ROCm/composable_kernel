#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the tensor_quant GEMM TileEngine -> Dispatcher bridge.

Locks the byte-exact fp8/bf8 x rcr scope, the arch-derived warp_tile_k trap, and
the config problem defaults for the Old-TE gemm_quant_tensor.cpp instance builder.
No GPU, no hipcc, no Old-TE builder import required.

The config-name prefix / tiles-in-name contract, the byte-exact codegen<->utils
kernel-name contract, and the codegen-JSON projection roundtrip are exercised for
every quant bridge (including this one) by the shared parametrized tests in
test_quant_bridge_shared.py, driven by _quant_bridge_descriptors.py.
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_tensor_quant_utils import (  # noqa: E402
    TensorQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    fp8_warp_tile_k_for_arch,
    expand_tensor_quant_sweep,
    setup_multiple_tensor_quant_dispatchers,
)
from unified_gemm_tensor_quant_codegen import _build_specs  # noqa: E402


class TestScope(unittest.TestCase):
    """gemm_quant_tensor.cpp registers exactly fp8/tensor and bf8/tensor, rcr only."""

    def test_default_variants(self):
        self.assertEqual(default_fp8_config().variant_key, "fp8")
        self.assertEqual(default_bf8_config().variant_key, "bf8")

    def test_layout_is_rcr(self):
        self.assertEqual(default_fp8_config().layout, "rcr")
        self.assertEqual(default_bf8_config().layout, "rcr")


class TestArchWarpTileK(unittest.TestCase):
    """WarpTileK must be arch-derived (get_k_warp_tile<fp8/bf8, 16>()).

    Hardcoding warp_tile_k=128 on gfx942 compiles but silently outputs
    all-zeros (confirmed on GPU, MI300X): there is no valid 16x16x128 fp8/bf8
    warp-gemm on gfx942. The correct value there is 32; gfx950 uses 128.
    """

    def test_helper_gfx942_is_32(self):
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx942"), 32)

    def test_helper_gfx950_is_128(self):
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx950"), 128)

    def test_fp8_default_gfx942_warp_tile_k_32(self):
        self.assertEqual(default_fp8_config("gfx942").warp_tile_k, 32)

    def test_fp8_default_gfx950_warp_tile_k_128(self):
        self.assertEqual(default_fp8_config("gfx950").warp_tile_k, 128)

    def test_bf8_default_gfx942_warp_tile_k_32(self):
        self.assertEqual(default_bf8_config("gfx942").warp_tile_k, 32)

    def test_bf8_default_gfx950_warp_tile_k_128(self):
        self.assertEqual(default_bf8_config("gfx950").warp_tile_k, 128)

    def test_name_reflects_arch_warp_tile_k(self):
        self.assertIn("16x16x32", default_fp8_config("gfx942").name)
        self.assertIn("16x16x128", default_fp8_config("gfx950").name)


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = TensorQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)


class TestGfx1250LogicalK32(unittest.TestCase):
    """GPU regressions: K32 returned rc=0 with 50% zeros and no MFMA/WMMA.

    The logical-K32 WMMA adapter must retain the explicit tile and name.
    These bridge tests verify request preservation; arithmetic needs GPU tests.
    """

    def test_construction_preserves_k32(self):
        for factory in (default_fp8_config, default_bf8_config):
            for arch in ("gfx1250", "gfx1250:xnack-"):
                with self.subTest(factory=factory.__name__, arch=arch):
                    cfg = replace(factory(arch), warp_tile_k=32)
                    self.assertEqual(cfg.warp_tile_k, 32)
                    self.assertIn("16x16x32", cfg.name)

    def test_supported_tiles_and_other_targets_are_preserved(self):
        for factory in (default_fp8_config, default_bf8_config):
            for arch, ks in (("gfx942", (32,)), ("gfx950", (32, 128)),
                             ("gfx1250", (32, 64, 128)), ("gfx1250:xnack-", (32, 64, 128))):
                for k in ks:
                    with self.subTest(factory=factory.__name__, arch=arch, k=k):
                        cfg = replace(factory(arch), warp_tile_k=k)
                        spec, = _build_specs(cfg.to_codegen_config())
                        self.assertEqual(cfg.warp_tile_k, k)
                        self.assertEqual(spec.tile.warp_tile_k, k)
                        self.assertEqual(spec.name, cfg.name)

    def test_mutation_preserved_through_codegen_and_build(self):
        cfg = default_bf8_config("gfx1250")
        cfg.warp_tile_k = 32
        original = copy.deepcopy(vars(cfg))
        spec, = _build_specs(cfg.to_codegen_config())
        self.assertEqual(spec.tile.warp_tile_k, 32)
        with patch("gemm_tensor_quant_utils.build_dispatchers") as build:
            setup_multiple_tensor_quant_dispatchers([cfg], gfx_arch="gfx1250")
            build.assert_called_once()
            self.assertEqual(build.call_args.args[0][0].warp_tile_k, 32)
        self.assertEqual(vars(cfg), original)

    def test_build_checks_actual_target_without_recorded_arch(self):
        cfg = default_fp8_config("gfx942")
        cfg.gfx_arch = ""
        with patch("gemm_tensor_quant_utils.build_dispatchers") as build:
            setup_multiple_tensor_quant_dispatchers([cfg], gfx_arch="gfx1250")
            build.assert_called_once()
            self.assertEqual(build.call_args.kwargs["arch"], "gfx1250")
        self.assertEqual(cfg.warp_tile_k, 32)

    def test_retargeting_is_rejected_before_build(self):
        cfg = default_fp8_config("gfx942")
        with patch("gemm_tensor_quant_utils.build_dispatchers") as build:
            with self.assertRaisesRegex(ValueError, "different architecture"):
                setup_multiple_tensor_quant_dispatchers([cfg], gfx_arch="gfx1250")
            build.assert_not_called()

    def test_json_sweep_and_direct_specs_preserve_embedded_target(self):
        data = default_bf8_config("gfx942").to_codegen_config()
        data["gfx_arch"] = "gfx1250"
        original = copy.deepcopy(data)
        spec, = _build_specs(data)
        self.assertEqual(spec.tile.warp_tile_k, 32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data))
            cfg, = expand_tensor_quant_sweep(str(path), gfx_arch="gfx1250")
            self.assertEqual(cfg.warp_tile_k, 32)
            self.assertEqual(cfg.name, spec.name)
        self.assertEqual(data, original)

    def test_replacement_has_distinct_name(self):
        requested = default_bf8_config("gfx942")
        replacement = replace(requested, gfx_arch="gfx1250", warp_tile_k=64)
        self.assertNotEqual(requested.name, replacement.name)
        self.assertEqual(requested.warp_tile_k, 32)


class TestTensorQuantTargetCli(unittest.TestCase):
    SCRIPT = _DISP / "codegen" / "unified_gemm_tensor_quant_codegen.py"

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(self.SCRIPT), *args],
                              capture_output=True, text=True, timeout=30)

    def test_explicit_and_embedded_targets_generate_k32_without_retuning(self):
        for variant in (default_fp8_config, default_bf8_config):
            for embedded in (False, True):
                data = variant("gfx942").to_codegen_config()
                data.pop("gfx_arch")
                extra = ["--gfx-arch", "gfx1250:xnack-"]
                if embedded:
                    data["gfx_arch"] = "gfx1250:xnack-"
                    extra = []
                with self.subTest(variant=variant.__name__, embedded=embedded):
                    with tempfile.TemporaryDirectory() as tmp:
                        result = self.run_cli("--config-json", json.dumps(data),
                                              "--output-dir", tmp, *extra)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        header, = Path(tmp).glob("*.hpp")
                        self.assertIn("16x16x32", header.stem)
                        self.assertEqual(header.stem, variant("gfx942").name)

    def test_config_file_list_names_preserves_k32(self):
        data = default_bf8_config("gfx942").to_codegen_config()
        data.pop("gfx_arch")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data))
            result = self.run_cli("--config", str(path), "--list-names",
                                  "--gfx-arch", "gfx1250")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), default_bf8_config("gfx942").name)

    def test_embedded_and_cli_arch_must_agree(self):
        data = default_fp8_config("gfx942").to_codegen_config()
        result = self.run_cli("--config-json", json.dumps(data), "--list-names",
                              "--gfx-arch", "gfx1250")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config records gfx_arch=", result.stderr)

    def test_explicit_supported_tiles_are_unchanged(self):
        for arch, k in (("gfx942", 32), ("gfx950", 32),
                        ("gfx1250", 32), ("gfx1250", 64), ("gfx1250:xnack-", 128)):
            cfg = replace(default_bf8_config(arch), warp_tile_k=k)
            with self.subTest(arch=arch, k=k):
                result = self.run_cli("--config-json", json.dumps(cfg.to_codegen_config()),
                                      "--list-names")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), cfg.name)

    def test_default_generation_and_listing_use_same_target(self):
        for arch, k in (("gfx942", 32), ("gfx950", 128), ("gfx1250", 128)):
            with self.subTest(arch=arch):
                listed = self.run_cli("--list-names", "--gfx-arch", arch)
                self.assertEqual(listed.returncode, 0, listed.stderr)
                with tempfile.TemporaryDirectory() as tmp:
                    generated = self.run_cli("--output-dir", tmp, "--no-parallel",
                                             "--gfx-arch", arch)
                    self.assertEqual(generated.returncode, 0, generated.stderr)
                    names = {p.stem for p in Path(tmp).glob("*.hpp")}
                self.assertEqual(names, set(listed.stdout.splitlines()))
                self.assertTrue(all(f"16x16x{k}" in name for name in names))


if __name__ == "__main__":
    unittest.main()
