# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU unit tests for the gfx1250 comp_async / comp_tdm / comp_tdm_v2 GEMM
pipelines in the Tile Engine: trait parsing, validation rules and generated
kernel instances. No GPU is required."""

import json
import os
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gemm_validation_utils as vu  # noqa: E402
from gemm_instance_builder import GemmKernelBuilder, lookup_pipeline  # noqa: E402
from trait_parse import (  # noqa: E402
    MULTI_TOKEN_PIPELINES,
    TraitCombo,
    join_trait,
    parse_trait,
    split_trait,
)

_TILE = {
    "tile_m": 128,
    "tile_n": 128,
    "tile_k": 64,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 16,
    "warp_tile_n": 16,
    "warp_tile_k": 16,
}


def _config(pipelines, epilogues, persistent=(False,), pads=(False,)):
    return {
        "tile_config": {k: {"values": [v]} for k, v in _TILE.items()},
        "trait_config": {
            "pipeline": {"values": list(pipelines)},
            "scheduler": {"values": ["intrawave"]},
            "epilogue": {"values": list(epilogues)},
            "pad_m": {"values": list(pads)},
            "pad_n": {"values": list(pads)},
            "pad_k": {"values": list(pads)},
            "persistent": {"values": list(persistent)},
        },
        "k_block_per_cu": 1,
    }


class TestTraitParse(unittest.TestCase):
    def test_legacy_split_unchanged(self):
        for trait in (
            "compv3_cshuffle_intrawave_False_False_False_False",
            "mem_default_interwave_True_True_True",
            "compv4_cshuffle_intrawave_True_False_True_True",
            "preshufflev2_default_intrawave_False_False_False_False",
        ):
            self.assertEqual(split_trait(trait), trait.split("_"))

    def test_multi_token_pipelines(self):
        for pipeline in MULTI_TOKEN_PIPELINES:
            for epilogue in ("cshuffle", "default", "tdm"):
                trait = f"{pipeline}_{epilogue}_intrawave_True_False_True_False"
                parts = split_trait(trait)
                self.assertEqual(parts[0], pipeline)
                self.assertEqual(parts[1], epilogue)
                self.assertEqual(parts[2], "intrawave")
                self.assertEqual(len(parts), 7)

    def test_v2_not_matched_as_v1(self):
        self.assertEqual(
            split_trait("comp_tdm_v2_tdm_intrawave_False_False_False")[0],
            "comp_tdm_v2",
        )
        self.assertEqual(
            split_trait("comp_tdm_tdm_intrawave_False_False_False")[0], "comp_tdm"
        )

    def test_round_trip(self):
        for pipeline in ("mem", "compv3", "compv4") + MULTI_TOKEN_PIPELINES:
            for persistent in (False, True):
                combo = TraitCombo(
                    pipeline, "cshuffle", "intrawave", True, False, True, persistent
                )
                self.assertEqual(parse_trait(join_trait(combo)), combo)
            combo6 = TraitCombo(pipeline, "tdm", "intrawave", False, True, False, False)
            self.assertEqual(
                parse_trait(join_trait(combo6, with_persistent=False)), combo6
            )

    def test_bad_field_count(self):
        with self.assertRaises(ValueError):
            parse_trait("comp_tdm_tdm_intrawave")

    @unittest.skipUnless(shutil.which("cmake"), "cmake not available")
    def test_cmake_macro_matches_python(self):
        traits = [
            "compv3_cshuffle_intrawave_False_False_False_False",
            "mem_default_interwave_True_True_True",
            "comp_async_cshuffle_intrawave_False_False_False_True",
            "comp_tdm_tdm_intrawave_False_False_False",
            "comp_tdm_v2_tdm_intrawave_True_False_False_False",
            "comp_async_eight_waves_cshuffle_intrawave_False_False_False",
            "weight_preshuffle_default_intrawave_False_False_False",
        ]
        macro = os.path.join(_HERE, "gemm_trait_parse.cmake")
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "probe.cmake")
            with open(script, "w") as f:
                f.write(f'include("{macro}")\n')
                for t in traits:
                    f.write(f"ck_tile_gemm_split_trait({t} p e s)\n")
                    f.write('message("${p}|${e}|${s}")\n')
            out = subprocess.run(
                ["cmake", "-P", script], capture_output=True, text=True, check=True
            )
        got = [line for line in out.stderr.splitlines() if line]
        want = ["|".join(split_trait(t)[:3]) for t in traits]
        self.assertEqual(got, want)


class TestValidationRules(unittest.TestCase):
    def test_pipelines_by_arch(self):
        for arch in ("gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"):
            self.assertEqual(vu.get_pipelines_for_arch(arch), vu.GEMM_PIPELINES)
        gfx1250 = vu.get_pipelines_for_arch("gfx1250")
        for p in ("comp_async", "comp_tdm", "comp_tdm_v2"):
            self.assertIn(p, gfx1250)
        self.assertEqual(vu.get_pipelines_for_arch("gfx1250:xnack-"), gfx1250)
        self.assertEqual(vu.GEMM_PIPELINES, ["mem", "compv3", "compv4"])

    def test_tdm_trait_rules(self):
        ok = vu.is_trait_combination_valid
        self.assertTrue(ok("comp_tdm", "tdm", "intrawave", False, "gemm_universal"))
        self.assertTrue(ok("comp_tdm_v2", "tdm", "intrawave", False, "batched_gemm"))
        self.assertFalse(
            ok("comp_tdm", "cshuffle", "intrawave", False, "gemm_universal")
        )
        self.assertFalse(ok("comp_tdm", "tdm", "interwave", False, "gemm_universal"))
        self.assertFalse(ok("comp_tdm", "tdm", "intrawave", True, "gemm_universal"))
        self.assertFalse(ok("compv3", "tdm", "intrawave", False, "gemm_universal"))
        self.assertFalse(ok("comp_async", "tdm", "intrawave", False, "gemm_universal"))

    def test_tdm_pad_rejected(self):
        ok = vu.is_trait_combination_valid
        for prefix in ("gemm_universal", "batched_gemm"):
            for p in ("comp_tdm", "comp_tdm_v2"):
                self.assertTrue(
                    ok(p, "tdm", "intrawave", False, prefix, "rcr", False, False, False)
                )
                for pads in (
                    (True, False, False),
                    (False, True, False),
                    (False, False, True),
                    (True, True, True),
                    ("true", False, False),
                ):
                    self.assertFalse(
                        ok(p, "tdm", "intrawave", False, prefix, "rcr", *pads),
                        (prefix, p, pads),
                    )
        # Full padding is allowed for comp_async and the legacy pipelines.
        for p in ("comp_async", "compv3", "compv4", "mem"):
            self.assertTrue(
                ok(
                    p,
                    "cshuffle",
                    "intrawave",
                    False,
                    "gemm_universal",
                    "rcr",
                    True,
                    True,
                    True,
                ),
                p,
            )

    def test_tdm_pad_reject_reason(self):
        r = vu.tdm_pad_reject_reason
        self.assertEqual(r("comp_tdm", "tdm"), "")
        self.assertEqual(r("comp_tdm", "tdm", False, False, False), "")
        self.assertEqual(r("comp_tdm_v2", "tdm", pad_k=True), vu.TDM_PAD_REJECT_REASON)
        self.assertEqual(r("compv3", "tdm", pad_m=True), vu.TDM_PAD_REJECT_REASON)
        self.assertEqual(r("comp_async", "cshuffle", True, True, True), "")
        self.assertEqual(r("compv3", "cshuffle", True, True, True), "")
        self.assertIn("pad_m=pad_n=pad_k=False", vu.TDM_PAD_REJECT_REASON)

    def test_comp_async_pad_reject_reason(self):
        r = vu.gfx1250_comp_async_pad_reject_reason
        self.assertEqual(r("comp_async", True, True, True), "")
        self.assertEqual(r("comp_async", "true", "True", True), "")
        for pads in (
            (False, False, False),
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            self.assertEqual(
                r("comp_async", *pads), vu.GFX1250_COMP_ASYNC_PAD_REJECT_REASON, pads
            )
        for p in ("compv3", "compv4", "mem", "comp_tdm", "comp_tdm_v2"):
            self.assertEqual(r(p, False, False, False), "", p)
        self.assertIn("pad_m=pad_n=pad_k=True", vu.GFX1250_COMP_ASYNC_PAD_REJECT_REASON)

    def test_comp_async_8bit_warp_tile_k_reject_reason(self):
        r = vu.gfx1250_comp_async_8bit_warp_tile_k_reject_reason
        reason = vu.GFX1250_COMP_ASYNC_8BIT_WARP_TILE_K_REJECT_REASON
        for dt in ("fp8", "bf8"):
            self.assertEqual(r("comp_async", dt, dt, 64), reason, dt)
            self.assertEqual(r("comp_async", dt, dt, 32), reason, dt)
            self.assertEqual(r("comp_async", dt, dt, 128), "", dt)
            for p in ("compv3", "compv4", "mem", "comp_tdm", "comp_tdm_v2"):
                self.assertEqual(r(p, dt, dt, 64), "", (p, dt))
        self.assertEqual(r("comp_async", "fp16", "fp16", 32), "")
        self.assertIn("warp_tile_k", reason)

    def test_comp_async_8bit_warp_tile_k_tile_config(self):
        tile = dict(_TILE, tile_k=128)

        def valid(pipe, dt, wtk, arch="gfx1250"):
            t = dict(tile, warp_tile_k=wtk)
            args = [t[k] for k in _TILE] + [dt, dt, "fp16"]
            return vu.is_tile_config_valid(*args, pipe, "rcr", arch, "gemm_universal")

        for dt in ("fp8", "bf8"):
            self.assertFalse(valid("comp_async", dt, 64), dt)
            self.assertTrue(valid("comp_async", dt, 128), dt)
            self.assertTrue(valid("compv3", dt, 64), dt)

    def test_comp_async_pad_trait_gate(self):
        ok = vu.is_trait_combination_valid
        for prefix in ("gemm_universal", "batched_gemm"):
            self.assertTrue(
                ok(
                    "comp_async",
                    "cshuffle",
                    "intrawave",
                    False,
                    prefix,
                    "rcr",
                    True,
                    True,
                    True,
                )
            )
            for pads in (
                (False, False, False),
                (True, True, False),
                (False, True, True),
            ):
                self.assertFalse(
                    ok(
                        "comp_async",
                        "cshuffle",
                        "intrawave",
                        False,
                        prefix,
                        "rcr",
                        *pads,
                    ),
                    (prefix, pads),
                )
        # No padding given: the pad rule is not checked (legacy callers).
        for prefix in ("", "gemm_universal", "batched_gemm"):
            self.assertTrue(
                ok("comp_async", "cshuffle", "intrawave", False, prefix), prefix
            )
        # Legacy pipelines keep accepting any padding.
        for p in ("compv3", "compv4", "mem"):
            self.assertTrue(
                ok(
                    p,
                    "cshuffle",
                    "intrawave",
                    False,
                    "gemm_universal",
                    "rcr",
                    False,
                    False,
                    False,
                ),
                p,
            )

    def test_comp_async_layout_gate(self):
        v = vu.validate_gemm_gfx1250_pipeline
        self.assertEqual(
            v(128, 128, 2, 2, 1, "fp16", "comp_async", "gfx1250", "rcr"), (True, "")
        )
        for layout in ("rrr", "crr", "ccr"):
            self.assertEqual(
                v(128, 128, 2, 2, 1, "fp16", "comp_async", "gfx1250", layout),
                (False, vu.GFX1250_COMP_ASYNC_LAYOUT_REJECT_REASON),
                layout,
            )
            # The TDM pipelines are not layout-gated.
            for p in ("comp_tdm", "comp_tdm_v2"):
                self.assertTrue(
                    v(128, 128, 2, 2, 1, "fp16", p, "gfx1250", layout)[0], (p, layout)
                )
        args = [_TILE[k] for k in _TILE] + ["fp16", "fp16", "fp16"]
        for prefix in ("gemm_universal", "batched_gemm"):
            self.assertTrue(
                vu.is_tile_config_valid(*args, "comp_async", "rcr", "gfx1250", prefix)
            )
            for layout in ("rrr", "crr", "ccr"):
                self.assertFalse(
                    vu.is_tile_config_valid(
                        *args, "comp_async", layout, "gfx1250", prefix
                    ),
                    (prefix, layout),
                )
                self.assertTrue(
                    vu.is_tile_config_valid(
                        *args, "comp_tdm", layout, "gfx1250", prefix
                    ),
                    (prefix, layout),
                )

    def test_mx_comp_async_gfx950_unaffected(self):
        # MX comp_async (gfx950) keeps its own rules: neither the gfx1250
        # layout gate nor the TDM pad gate is consulted on that path.
        tile = (128, 128, 256, 2, 2, 1, 16, 16, 128)
        want = {"rcr": True, "rrr": False, "crr": False, "ccr": False}

        def boom(*_a, **_k):
            raise AssertionError("gfx1250 gate consulted on the MX path")

        with mock.patch.object(
            vu, "gfx1250_comp_async_layout_reject_reason", boom
        ), mock.patch.object(vu, "tdm_pad_reject_reason", boom):
            for dtype in ("fp8", "fp4"):
                for layout, expected in want.items():
                    got = vu.is_tile_config_valid(
                        *tile,
                        dtype,
                        dtype,
                        "fp16",
                        "comp_async",
                        layout,
                        "gfx950",
                        "mx_gemm",
                    )
                    self.assertEqual(got, expected, (dtype, layout))
                    self.assertTrue(
                        vu.is_trait_combination_valid(
                            "comp_async",
                            "cshuffle",
                            "intrawave",
                            False,
                            "mx_gemm",
                            layout,
                            True,
                            True,
                            True,
                        ),
                        (dtype, layout),
                    )

    def test_quant_ops_reject_gfx1250_pipelines(self):
        ok = vu.is_trait_combination_valid
        for prefix in (
            "gemm_aquant",
            "gemm_bquant",
            "gemm_abquant",
            "gemm_rowcolquant",
        ):
            for p, e in (("comp_tdm", "tdm"), ("comp_tdm_v2", "tdm")):
                self.assertFalse(ok(p, e, "intrawave", False, prefix), (prefix, p))
        # aquant's legacy trait result for comp_async is unchanged; the async
        # rejection is arch-scoped in is_tile_config_valid instead.
        args = [_TILE[k] for k in _TILE] + ["fp16", "fp16", "fp16"]
        self.assertFalse(
            vu.is_tile_config_valid(
                *args, "comp_async", "rcr", "gfx1250", "gemm_aquant"
            )
        )

    def test_existing_trait_rules_unchanged(self):
        ok = vu.is_trait_combination_valid
        for p in ("mem", "compv3", "compv4"):
            for e in ("cshuffle", "default"):
                for s in ("intrawave", "interwave"):
                    want = (p, e, s) not in vu.TRAIT_UNSUPPORTED_COMBINATIONS
                    self.assertEqual(ok(p, e, s, False, "gemm_universal"), want)

    def test_uses_gfx1250_pipeline_routing(self):
        self.assertTrue(vu._uses_gfx1250_gemm_pipeline("comp_tdm", "gemm_universal"))
        self.assertTrue(vu._uses_gfx1250_gemm_pipeline("comp_async", "batched_gemm"))
        self.assertFalse(vu._uses_gfx1250_gemm_pipeline("comp_async", "mx_gemm"))
        # Empty prefix keeps the legacy MX routing of comp_async.
        self.assertFalse(vu._uses_gfx1250_gemm_pipeline("comp_async", ""))
        self.assertFalse(vu._uses_gfx1250_gemm_pipeline("compv3", "gemm_universal"))

    def test_gfx1250_pipeline_arch_and_waves(self):
        v = vu.validate_gemm_gfx1250_pipeline
        self.assertTrue(v(128, 128, 2, 2, 1, "fp16", "comp_tdm", "gfx1250")[0])
        self.assertTrue(v(128, 128, 2, 2, 1, "fp16", "comp_tdm_v2", "gfx1250")[0])
        self.assertTrue(v(128, 128, 4, 2, 1, "fp16", "comp_tdm", "gfx1250")[0])
        self.assertFalse(v(128, 128, 4, 2, 1, "fp16", "comp_tdm_v2", "gfx1250")[0])
        for arch in ("gfx942", "gfx950", "gfx1201"):
            for p in ("comp_tdm", "comp_tdm_v2", "comp_async"):
                self.assertFalse(v(128, 128, 2, 2, 1, "fp16", p, arch)[0])

    def test_tile_config_rejects_off_arch(self):
        args = [_TILE[k] for k in _TILE] + ["fp16", "fp16", "fp16"]
        for p in ("comp_tdm", "comp_tdm_v2", "comp_async"):
            self.assertFalse(
                vu.is_tile_config_valid(*args, p, "rcr", "gfx942", "gemm_universal")
            )
            self.assertTrue(
                vu.is_tile_config_valid(*args, p, "rcr", "gfx1250", "gemm_universal")
            )


class _BuilderCase(unittest.TestCase):
    def _builder(self, prefix, cfg, arch="gfx1250", layout="rcr"):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "cfg.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        return GemmKernelBuilder(prefix, tmp, arch, "fp16", layout, path)

    def _gen(
        self,
        prefix,
        pipeline,
        epilogue,
        persistent=False,
        arch="gfx1250",
        pads=(False, False, False),
        layout="rcr",
    ):
        b = self._builder(prefix, _config([pipeline], [epilogue]), arch, layout)
        combo = (pipeline, epilogue, "intrawave", *pads, persistent)
        return b._generate_kernel_instance(dict(_TILE), combo)[1]


class TestBuilderGolden(_BuilderCase):
    def test_universal_tdm_instance(self):
        for pipeline, impl in (
            ("comp_tdm", "GemmPipelineAgBgCrCompTDMV1"),
            ("comp_tdm_v2", "GemmPipelineAgBgCrCompTDMV2"),
        ):
            code = self._gen("gemm_universal", pipeline, "tdm")
            self.assertIn('#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"', code)
            self.assertIn(impl, code)
            self.assertIn("TdmEpilogue", code)
            self.assertIn(
                'throw std::runtime_error("TDM pipeline requires k_batch==1")', code
            )
            self.assertIn("DoubleSmemBuffer = true", code)
            self.assertNotIn("CShuffleEpilogue<", code)

    def test_batched_tdm_instance(self):
        code = self._gen("batched_gemm", "comp_tdm", "tdm")
        self.assertIn("TdmEpilogue", code)
        self.assertIn("k_batch==1", code)

    def test_comp_async_instance(self):
        for prefix in ("gemm_universal", "batched_gemm"):
            code = self._gen(prefix, "comp_async", "cshuffle", pads=(True,) * 3)
            self.assertIn("GemmPipelineAgBgCrCompAsync", code, prefix)
            self.assertIn("DoubleSmemBuffer = true", code, prefix)
            self.assertNotIn("TdmEpilogue", code, prefix)
            self.assertNotIn("tdm_epilogue.hpp", code, prefix)
            self.assertNotIn("k_batch==1", code, prefix)

    def test_legacy_instance_has_no_tdm(self):
        code = self._gen("gemm_universal", "compv3", "cshuffle", arch="gfx942")
        for needle in ("tdm", "Tdm", "TDM", "CompAsync"):
            self.assertNotIn(needle, code)
        self.assertIn("DoubleSmemBuffer = false", code)

    def test_ops_reject_unsupported_pipelines(self):
        for prefix in (
            "gemm_multi_d",
            "gemm_multi_abd",
            "grouped_gemm",
            "batched_contraction",
        ):
            with self.assertRaises(ValueError, msg=prefix):
                self._gen(prefix, "comp_tdm", "tdm")
            with self.assertRaises(ValueError, msg=prefix):
                self._gen(prefix, "comp_async", "cshuffle", pads=(True,) * 3)
        with self.assertRaises(ValueError):
            self._gen("gemm_universal", "compv3", "tdm")
        with self.assertRaises(ValueError):
            self._gen("gemm_universal", "comp_tdm", "cshuffle")

    def test_builder_rejects_padded_tdm(self):
        for prefix in ("gemm_universal", "batched_gemm"):
            for pipeline in ("comp_tdm", "comp_tdm_v2"):
                for pads in (
                    (True, False, False),
                    (False, True, False),
                    (False, False, True),
                ):
                    with self.assertRaisesRegex(ValueError, "TDM bounds-clips"):
                        self._gen(prefix, pipeline, "tdm", pads=pads)
            # comp_async requires full padding.
            code = self._gen(prefix, "comp_async", "cshuffle", pads=(True,) * 3)
            self.assertIn("GemmPipelineAgBgCrCompAsync", code)

    def test_builder_rejects_unpadded_comp_async(self):
        for prefix in ("gemm_universal", "batched_gemm"):
            for pads in (
                (False, False, False),
                (False, True, True),
                (True, False, True),
                (True, True, False),
            ):
                with self.assertRaisesRegex(
                    ValueError, "requires pad_m=pad_n=pad_k=True"
                ):
                    self._gen(prefix, "comp_async", "cshuffle", pads=pads)
        # Off gfx1250 the builder does not consult the gate.
        code = self._gen("gemm_universal", "compv3", "cshuffle", arch="gfx942")
        self.assertIn("DoubleSmemBuffer = false", code)

    def test_builder_comp_async_rc_layout_only(self):
        for prefix in ("gemm_universal", "batched_gemm"):
            for layout in ("rrr", "crr", "ccr"):
                with self.assertRaisesRegex(ValueError, "A row-major and B col-major"):
                    self._gen(
                        prefix,
                        "comp_async",
                        "cshuffle",
                        pads=(True,) * 3,
                        layout=layout,
                    )
                code = self._gen(prefix, "comp_tdm", "tdm", layout=layout)
                self.assertIn("TdmEpilogue", code, (prefix, layout))

    def test_trait_enumeration_tdm_unpadded(self):
        b = self._builder(
            "gemm_universal",
            _config(
                ["comp_async", "comp_tdm", "comp_tdm_v2"],
                ["cshuffle", "tdm"],
                pads=(False, True),
            ),
        )
        combos = b._generate_trait_combinations()
        tdm_pads = {
            tuple(c[3:6]) for c in combos if c[0] in ("comp_tdm", "comp_tdm_v2")
        }
        self.assertEqual(tdm_pads, {(False, False, False)})
        async_pads = {tuple(c[3:6]) for c in combos if c[0] == "comp_async"}
        self.assertEqual(async_pads, {(True, True, True)})

    def test_lookup_pipeline_strict(self):
        self.assertEqual(lookup_pipeline({"a": "x"}, "a"), "x")
        with self.assertRaises(ValueError):
            lookup_pipeline({"a": "x"}, "b")

    def test_trait_enumeration(self):
        b = self._builder(
            "gemm_universal",
            _config(
                ["compv3", "comp_async", "comp_tdm", "comp_tdm_v2"],
                ["cshuffle", "tdm"],
                persistent=(False, True),
                pads=(False, True),
            ),
        )
        combos = {(c[0], c[1], c[6]) for c in b._generate_trait_combinations()}
        self.assertIn(("comp_tdm", "tdm", False), combos)
        self.assertIn(("comp_tdm_v2", "tdm", False), combos)
        self.assertNotIn(("comp_tdm", "tdm", True), combos)
        self.assertNotIn(("comp_tdm", "cshuffle", False), combos)
        self.assertNotIn(("compv3", "tdm", False), combos)
        self.assertIn(("comp_async", "cshuffle", True), combos)


if __name__ == "__main__":
    unittest.main()
