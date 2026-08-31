#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the batched-contraction TileEngine -> Dispatcher bridge.

Lock the byte-exact name contract between codegen and utils, the codegen-JSON
projection, problem flop counting, and sweep expansion. No GPU required.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from batched_contraction_utils import (  # noqa: E402
    BatchedContractionKernelConfig,
    BatchedContractionProblem,
    _SUPPORTED_ARCHS,
    _get_arch,
    _validate_arch,
    expand_sweep,
)

_CONFIG_DIR = _DISP.parent / "tile_engine" / "ops" / "gemm" / "batched_contraction" / "configs"
from unified_batched_contraction_codegen import (  # noqa: E402
    make_batched_contraction_kernel_name,
    _spec_from_config,
)


class TestNameContract(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(dtype="fp16", layout="rcr", pipeline="compv3", epilogue="cshuffle",
                    scheduler="intrawave", tile_m=64, tile_n=64, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1, warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)
        base.update(kw)
        return BatchedContractionKernelConfig(**base)

    def test_name_prefix(self):
        self.assertTrue(self._cfg().name.startswith("batched_contraction_fp16_rcr_"))

    def test_config_name_equals_codegen_name(self):
        # utils .name and codegen _spec_from_config(...).name must be byte-identical
        cfg = self._cfg(num_dim_g=2, num_dim_m=1, num_dim_n=1, num_dim_k=1)
        spec = _spec_from_config(cfg.to_codegen_config())
        self.assertEqual(cfg.name, spec.name)

    def test_num_dim_changes_name(self):
        a = self._cfg(num_dim_g=1).name
        b = self._cfg(num_dim_g=2).name
        self.assertNotEqual(a, b)
        self.assertIn("g1m1n1k1", a)
        self.assertIn("g2m1n1k1", b)

    def test_dtype_layout_in_name(self):
        self.assertIn("_bf16_rrr_", self._cfg(dtype="bf16", layout="rrr").name)

    def test_no_spaces(self):
        self.assertNotIn(" ", self._cfg().name)

    def test_elementwise_suffix(self):
        self.assertTrue(self._cfg(elementwise="MultiDAdd", num_d_tensors=1).name.endswith("_MultiDAdd"))
        self.assertNotIn("PassThrough", self._cfg().name)


class TestCodegenJson(unittest.TestCase):
    def test_roundtrip_tile(self):
        cfg = BatchedContractionKernelConfig(tile_m=128, tile_n=256, tile_k=64,
                                             warp_m=2, warp_n=2, warp_k=1,
                                             warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)
        j = cfg.to_codegen_config()
        self.assertEqual(j["tile_config"]["tile_m"], 128)
        self.assertEqual(j["tile_config"]["tile_n"], 256)
        self.assertEqual(j["tile_config"]["warp_tile_k"], 16)
        self.assertEqual(j["datatype"], "fp16")

    def test_num_dim_projection(self):
        cfg = BatchedContractionKernelConfig(num_dim_g=2, num_dim_k=3)
        j = cfg.to_codegen_config()
        self.assertEqual(j["num_dim_g"], 2)
        self.assertEqual(j["num_dim_k"], 3)


class TestProblem(unittest.TestCase):
    def test_products(self):
        p = BatchedContractionProblem(g_dims=[2, 3], m_dims=[4, 16], n_dims=[128], k_dims=[4, 16])
        self.assertEqual(p.G, 6)
        self.assertEqual(p.M, 64)
        self.assertEqual(p.N, 128)
        self.assertEqual(p.K, 64)

    def test_flops(self):
        p = BatchedContractionProblem(g_dims=[3], m_dims=[128], n_dims=[128], k_dims=[128])
        self.assertEqual(p.flops, 2 * 3 * 128 * 128 * 128)

    def test_roundtrip(self):
        p = BatchedContractionProblem(g_dims=[2], m_dims=[64], n_dims=[64], k_dims=[64], k_batch=2)
        self.assertEqual(BatchedContractionProblem.from_dict(p.to_dict()).to_dict(), p.to_dict())


class TestValidity(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(BatchedContractionKernelConfig(
            tile_m=64, tile_n=64, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16).is_valid())

    def test_invalid_divisibility(self):
        self.assertFalse(BatchedContractionKernelConfig(
            tile_m=48, tile_n=64, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16).is_valid())

    def _base(self, **kw):
        base = dict(dtype="fp16", layout="rcr", tile_m=128, tile_n=128, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1,
                    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)
        base.update(kw)
        return BatchedContractionKernelConfig(**base)

    def test_reject_non_rcr_layout(self):
        # Only rcr compiles; col-major A/B trip kernel static_asserts in Old-TE.
        for layout in ("rrr", "ccr", "crr", "rcc"):
            self.assertFalse(self._base(layout=layout).is_valid(), layout)

    def test_reject_bad_dtype(self):
        # fp8/bf8/int8 are out of this bridge's dtype allow-list.
        for dt in ("fp8", "bf8", "int8", "fp64"):
            self.assertFalse(self._base(dtype=dt).is_valid(), dt)

    def test_reject_bad_warp_tile(self):
        # A warp tile not in the per-dtype MFMA allow-list must be rejected.
        self.assertFalse(self._base(warp_tile_m=8, warp_tile_n=8, warp_tile_k=8).is_valid())
        # fp32 does not admit the fp16 32x32x16 warp tile.
        self.assertFalse(self._base(dtype="fp32", tile_k=64,
                                    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16).is_valid())

    def test_num_d_range(self):
        # num_d==0 => PassThrough valid; negative rejected; large rejected.
        self.assertTrue(self._base(num_d_tensors=0, elementwise="PassThrough").is_valid())
        self.assertFalse(self._base(num_d_tensors=-1).is_valid())
        self.assertFalse(self._base(num_d_tensors=9, elementwise="MultiDAdd").is_valid())

    def test_num_d_elementwise_consistency(self):
        # num_d>0 requires a MultiD* op; PassThrough with D is rejected.
        self.assertFalse(self._base(num_d_tensors=1, elementwise="PassThrough").is_valid())
        # num_d==0 with a MultiD* op is rejected (nothing for the op to consume).
        self.assertFalse(self._base(num_d_tensors=0, elementwise="MultiDAdd").is_valid())
        # Valid D configs.
        self.assertTrue(self._base(num_d_tensors=1, elementwise="MultiDAdd").is_valid())
        self.assertTrue(self._base(num_d_tensors=2, elementwise="MultiDMultiply").is_valid())


class TestNumDContract(unittest.TestCase):
    """num_d>0 name + codegen-projection round-trip (CPU only)."""

    def _cfg(self, num_d, ew):
        return BatchedContractionKernelConfig(
            dtype="fp16", layout="rcr", pipeline="compv3", epilogue="cshuffle",
            scheduler="intrawave", tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1, warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            num_dim_g=1, num_dim_m=1, num_dim_n=1, num_dim_k=1,
            num_d_tensors=num_d, elementwise=ew)

    def test_name_has_d_and_op_suffix(self):
        n = self._cfg(1, "MultiDAdd").name
        self.assertIn("_d1_", n + "_")
        self.assertTrue(n.endswith("_MultiDAdd"))
        n2 = self._cfg(2, "MultiDMultiply").name
        self.assertIn("_d2_", n2 + "_")
        self.assertTrue(n2.endswith("_MultiDMultiply"))

    def test_name_equals_codegen_name(self):
        for num_d, ew in [(1, "MultiDAdd"), (2, "MultiDAdd"), (1, "MultiDMultiply")]:
            cfg = self._cfg(num_d, ew)
            spec = _spec_from_config(cfg.to_codegen_config())
            self.assertEqual(cfg.name, spec.name, f"{num_d}/{ew}")

    def test_codegen_json_projects_num_d_and_op(self):
        j = self._cfg(2, "MultiDMultiply").to_codegen_config()
        self.assertEqual(j["num_d_tensors"], 2)
        self.assertEqual(j["elementwise"], "MultiDMultiply")

    def test_num_d_changes_name(self):
        self.assertNotEqual(self._cfg(1, "MultiDAdd").name, self._cfg(2, "MultiDAdd").name)
        # Same D count but different op must still differ.
        self.assertNotEqual(self._cfg(1, "MultiDAdd").name, self._cfg(1, "MultiDMultiply").name)


class TestSweep(unittest.TestCase):
    def test_expand_dedup_and_valid(self):
        config = {
            "tile_config": {
                "tile_m": {"values": [64, 128]}, "tile_n": {"values": [64]},
                "tile_k": {"values": [64]}, "warp_m": {"values": [2]},
                "warp_n": {"values": [2]}, "warp_k": {"values": [1]},
                "warp_tile_m": {"values": [32]}, "warp_tile_n": {"values": [32]},
                "warp_tile_k": {"values": [16]},
            },
            "trait_config": {"pipeline": {"values": ["compv3", "mem"]},
                             "scheduler": {"values": ["intrawave"]},
                             "epilogue": {"values": ["cshuffle"]}},
            "num_dim_g": 1, "num_dim_m": 1, "num_dim_n": 1, "num_dim_k": 1,
        }
        cfgs = expand_sweep(config)
        names = [c.name for c in cfgs]
        self.assertEqual(len(names), len(set(names)))  # deduped
        self.assertTrue(all(c.is_valid() for c in cfgs))
        self.assertEqual(len(cfgs), 2 * 2)  # (tile_m 2) x (pipeline 2)


# --- gfx1250 (MI400 / RDNA4-WMMA) enablement -------------------------------
# The batched-contraction bridge historically allow-listed only CDNA
# (gfx90a/942/950, MFMA). gfx1250 uses WMMA, so it needs an arch-tuple entry and
# WMMA CI configs (warp_tile 16x16x32 -- the CDNA MFMA 32x32x64/32x32x16 tiles do
# not run on gfx1250). These CPU-only tests lock that surface in.
class TestGfx1250Enablement(unittest.TestCase):
    def test_gfx1250_in_supported_archs(self):
        self.assertIn("gfx1250", _SUPPORTED_ARCHS)

    def test_validate_arch_accepts_gfx1250(self):
        self.assertEqual(_validate_arch("gfx1250"), "gfx1250")

    def test_get_arch_detects_gfx1250(self):
        with mock.patch("subprocess.check_output", return_value="  Name:  gfx1250\n"):
            self.assertEqual(_get_arch(), "gfx1250")

    def _assert_wmma_tile(self, name):
        cfg_path = _CONFIG_DIR / name
        self.assertTrue(cfg_path.is_file(), cfg_path)
        tc = json.loads(cfg_path.read_text())["tile_config"]
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32])

    def test_gfx1250_ci_config_present_and_wmma(self):
        self._assert_wmma_tile("default_ci_config_gfx1250.json")

    def test_gfx1250_bridge_ci_config_present_and_wmma(self):
        self._assert_wmma_tile("bridge_default_ci_config_gfx1250.json")

    def test_gfx1250_config_expands_to_valid_wmma_kernels(self):
        cfg = json.loads((_CONFIG_DIR / "default_ci_config_gfx1250.json").read_text())
        cfgs = expand_sweep(cfg)
        self.assertTrue(cfgs)
        self.assertTrue(all(c.is_valid() for c in cfgs))
        self.assertTrue(all(c.warp_tile_m == 16 and c.warp_tile_k == 32 for c in cfgs))


if __name__ == "__main__":
    unittest.main()
