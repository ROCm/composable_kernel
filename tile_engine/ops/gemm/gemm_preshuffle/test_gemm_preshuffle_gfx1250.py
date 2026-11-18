# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the gfx1250 pipelines of the gemm_preshuffle Tile Engine op.

preshuffle_tdm, comp_tdm, comp_tdm_v2 and comp_async are emitted on gfx1250
for rcr only (the TDM ones unpadded, comp_async fully padded). gfx942 /
gfx950 keep only preshufflev2, which cannot pad N or K on gfx1250.
"""

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from gemm_preshuffle_instance_builder import GemmPreshuffleKernelBuilder  # noqa: E402
from gemm_validation_utils import (  # noqa: E402
    is_tile_config_valid,
    is_trait_combination_valid,
)
from trait_parse import join_trait, parse_trait  # noqa: E402

GFX1250_CONFIG = os.path.join(_HERE, "configs", "default_config_gfx1250.json")
COMP = ("comp_tdm", "comp_tdm_v2", "comp_async")


def _trait_ok(pipeline, gpu_target="gfx1250", layout="rcr", pad=False, pad_m=None):
    scheduler = "intrawave" if pipeline.startswith("comp_") else "default"
    epilogue = "tdm" if pipeline.startswith("comp_tdm") else "cshuffle"
    return is_trait_combination_valid(
        pipeline, epilogue, scheduler, False, "gemm_preshuffle", layout,
        pad_m=pad if pad_m is None else pad_m, pad_n=pad, pad_k=pad, gpu_target=gpu_target,
    )


def _tile_ok(pipeline, gpu_target="gfx1250", warp_n=4, dtype="fp16", tile_k=64, wtk=None):
    is_8bit = dtype in ("fp8", "bf8")
    wtk = wtk or (64 if is_8bit else 32)
    # Like the instance builder, 8-bit inputs write fp16 C.
    return is_tile_config_valid(
        128, 128, tile_k, 1, warp_n, 1, 16, 16, wtk, dtype, dtype, "fp16" if is_8bit else dtype,
        pipeline, "rcr", gpu_target, kernel_name_prefix="gemm_preshuffle",
    )


class TestTraitRules(unittest.TestCase):
    def test_preshuffle_tdm_gfx1250_rcr_unpadded_only(self):
        self.assertTrue(_trait_ok("preshuffle_tdm"))
        self.assertTrue(_trait_ok("preshuffle_tdm", gpu_target="gfx1250:sramecc+"))
        self.assertFalse(_trait_ok("preshuffle_tdm", gpu_target="gfx942"))
        self.assertFalse(_trait_ok("preshuffle_tdm", gpu_target="gfx950"))
        self.assertFalse(_trait_ok("preshuffle_tdm", layout="rrr"))
        self.assertFalse(_trait_ok("preshuffle_tdm", pad=True))

    def test_comp_pipelines_gfx1250_rcr_only(self):
        for pipeline in COMP:
            # The TDM pipelines must be unpadded, comp_async padded in M, N and K.
            pad = pipeline == "comp_async"
            self.assertTrue(_trait_ok(pipeline, pad=pad), pipeline)
            self.assertFalse(_trait_ok(pipeline, gpu_target="gfx950", pad=pad), pipeline)
            self.assertFalse(_trait_ok(pipeline, layout="rrr", pad=pad), pipeline)
            self.assertFalse(_trait_ok(pipeline, pad=not pad), pipeline)
            self.assertFalse(_trait_ok(pipeline, pad=pad, pad_m=not pad), pipeline)

    def test_unwired_tdm_pipeline_rejected(self):
        # comp_tdm_pc is a gemm_universal TDM pipeline with no preshuffle kernel.
        self.assertFalse(_trait_ok("comp_tdm_pc"))

    def test_preshufflev2_unchanged(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            self.assertTrue(_trait_ok("preshufflev2", gpu_target=arch))
            self.assertTrue(_trait_ok("preshufflev2", gpu_target=arch, pad_m=True))
            # Padded N/K cannot be preshuffled on gfx1250.
            self.assertEqual(_trait_ok("preshufflev2", gpu_target=arch, pad=True),
                             arch != "gfx1250", arch)


class TestTileRules(unittest.TestCase):
    def test_preshuffle_tdm_tile_accepted_on_gfx1250(self):
        for dtype in ("fp16", "bf16", "fp8", "bf8"):
            self.assertTrue(_tile_ok("preshuffle_tdm", dtype=dtype), dtype)

    def test_preshuffle_tdm_tile_rejected_off_gfx1250(self):
        self.assertFalse(_tile_ok("preshuffle_tdm", gpu_target="gfx942"))

    def test_comp_pipelines_tile_accepted_on_gfx1250(self):
        for pipeline in COMP:
            self.assertTrue(_tile_ok(pipeline), pipeline)
            self.assertFalse(_tile_ok(pipeline, gpu_target="gfx950"), pipeline)

    def test_comp_async_rejects_8bit(self):
        for dtype in ("fp8", "bf8"):
            self.assertTrue(_tile_ok("comp_tdm", dtype=dtype), dtype)
            self.assertFalse(_tile_ok("comp_async", dtype=dtype), dtype)

    def test_comp_pipelines_use_ordinary_b_warp_tiles(self):
        # comp_* read ordinary B, so 8-bit 16x16x128 is valid for them (comp_async
        # requires it); the packed-B whitelist stays K64-only for preshuffle_tdm.
        for dtype in ("fp8", "bf8"):
            for pipeline in COMP:
                self.assertTrue(_tile_ok(pipeline, dtype=dtype, tile_k=128, wtk=128),
                                (pipeline, dtype))
            self.assertFalse(_tile_ok("preshuffle_tdm", dtype=dtype, tile_k=128, wtk=128), dtype)


class TestTraitParse(unittest.TestCase):
    def test_preshuffle_tdm_round_trip(self):
        name = "preshuffle_tdm_cshuffle_default_False_False_False_False"
        combo = parse_trait(name)
        self.assertEqual(combo.pipeline, "preshuffle_tdm")
        self.assertEqual(combo.epilogue, "cshuffle")
        self.assertEqual(join_trait(combo), name)


class TestBuilder(unittest.TestCase):
    def _builder(self, tmpdir, gpu_target="gfx1250", layout="rcr", datatype="fp16"):
        return GemmPreshuffleKernelBuilder(
            "gemm_preshuffle", tmpdir, gpu_target, datatype, layout, GFX1250_CONFIG
        )

    def test_gfx1250_config_keeps_gfx1250_pipelines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            combos = self._builder(tmpdir)._generate_trait_combinations()
        self.assertEqual({c[0] for c in combos}, {"preshufflev2", "preshuffle_tdm", *COMP})

    def test_gfx1250_config_keeps_only_preshufflev2_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for arch, layout in (("gfx942", "rcr"), ("gfx950", "rcr"), ("gfx1250", "rrr")):
                builder = self._builder(tmpdir, gpu_target=arch, layout=layout)
                combos = builder._generate_trait_combinations()
                self.assertEqual({c[0] for c in combos}, {"preshufflev2"}, (arch, layout))

    def test_generated_instance_uses_tdm_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = self._builder(tmpdir)
            tiles = builder._get_tile_configs()
            combo = next(c for c in builder._generate_trait_combinations()
                         if c[0] == "preshuffle_tdm")
            self.assertTrue(tiles)
            _, code = builder._generate_kernel_instance(tiles[0], combo)
        self.assertIn("WeightPreshufflePipelineAGmemBGmemCRegTDM", code)
        self.assertNotIn("CRegV2", code)

    def test_generated_comp_instances_use_intrawave(self):
        impl = {"comp_tdm": "CompTDMV1", "comp_tdm_v2": "CompTDMV2", "comp_async": "CompAsync"}
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = self._builder(tmpdir)
            tile = builder._get_tile_configs()[0]
            for combo in builder._generate_trait_combinations():
                if combo[0] not in impl:
                    continue
                _, code = builder._generate_kernel_instance(tile, combo)
                self.assertIn(f"GemmPipelineAgBgCr{impl[combo[0]]}", code)
                self.assertIn("GemmPipelineScheduler::Intrawave", code)


if __name__ == "__main__":
    unittest.main()
