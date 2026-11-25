#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests of comp_tdm_scheduled in tile_engine and the dispatcher.

The tuning, the validation rules, the dispatcher codegen and the compile flags, and that the
tile_engine CMake builder rejects the pipeline (it runs through the dispatcher). No GPU or hipcc
is required.
"""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_DISP = Path(__file__).resolve().parent.parent
_CK = _DISP.parent
_GEMM = _CK / "tile_engine" / "ops" / "gemm"
_MX = _GEMM / "mx_gemm"
_HOST_TEST = _CK / "test" / "ck_tile" / "gemm_mx" / "tdm_scheduled"
_POLICY_HPP = (
    _CK
    / "include"
    / "ck_tile"
    / "ops"
    / "gemm"
    / "pipeline"
    / "mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_policy.hpp"
)
for _p in (_DISP / "python", _DISP / "codegen", _GEMM, _MX, _HOST_TEST):
    sys.path.insert(0, str(_p))

import gemm_validation_utils as vu  # noqa: E402
import mx_gemm_utils as mx  # noqa: E402
import parity_cases  # noqa: E402
import unified_mx_gemm_codegen as codegen  # noqa: E402

sched = vu.load_mx_tdm_scheduled()
BUILDER = _MX / "mx_gemm_instance_builder.py"
TILE = {
    "tile_m": 256,
    "tile_n": 256,
    "tile_k": 512,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 32,
    "warp_tile_n": 16,
    "warp_tile_k": 128,
}


class TestTuning(unittest.TestCase):
    def test_unknown_key(self):
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"gl2": 4})
        # The tile-count parity is the pipeline's TailNumber, not an instance trait.
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"odd_tiles": 1})
        # TdmEpilogue stores E with no vector width.
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"store_width": 1})

    def test_instance_rules(self):
        def reason(t):
            return sched.validate_instance(
                sched.spec_from_tile(TILE, t, out_dtype="fp16"), t
            )

        self.assertEqual(reason({}), "ok")
        self.assertEqual(reason({"prefetch_lds_read": False}), "split_b_read")
        self.assertEqual(reason({"tail_steps": 4}), "tail_steps")
        self.assertEqual(reason({"cluster_dim": [8, 4]}), "cluster_size")
        self.assertEqual(reason({"gl2_prefetch_distance": 1}), "gl2_distance")
        # 8-bit types on the 32x16 FP4 atom, with a tile_k the 8-bit types allow.
        fp8_tile = {**TILE, "tile_m": 128, "tile_n": 128, "tile_k": 256}
        self.assertEqual(
            sched.validate_instance(
                sched.spec_from_tile(fp8_tile, {}, a_dtype="fp8", out_dtype="fp16"), {}
            ),
            "mma",
        )

    def test_k_limit(self):
        # K plus (4 + 1) tiles of 256 must fit in i32; M = N = 32 keeps the
        # scales under 2^31.
        tile = {**TILE, "tile_m": 32, "tile_n": 32, "tile_k": 256, "warp_tile_m": 16}
        t = {"cluster_dim": [1, 1], "split_b_lds_read": False}

        def reason(K):
            return sched.spec_from_tile(
                tile, t, a_dtype="fp8", out_dtype="fp16", M=32, N=32, K=K
            ).validate()

        self.assertNotEqual(reason(2147482336), "k_long")
        self.assertEqual(reason(2147482368), "k_long")

    def test_k_tail_full(self):
        # A K tail has fewer K steps than a tile: K mod tile_k at most tile_k - 128.
        tile = {**TILE, "tile_m": 128, "tile_n": 128, "tile_k": 256, "warp_tile_m": 16}
        t = {"split_b_lds_read": False}

        def reason(K, tile_k=256):
            return sched.spec_from_tile(
                {**tile, "tile_k": tile_k},
                t,
                a_dtype="fp8",
                out_dtype="fp16",
                M=1024,
                N=1024,
                K=K,
            ).validate()

        self.assertEqual(reason(2048 + 160), "k_tail_full")
        self.assertEqual(reason(2048 + 224), "k_tail_full")
        self.assertEqual(reason(2048 + 96), "ok")
        self.assertEqual(reason(2048), "ok")
        self.assertEqual(reason(1024 + 32, tile_k=128), "k_tail_full")

    def test_cpp_spelling(self):
        t = {
            "cluster_dim": [2, 2],
            "interleave_lds_regions": False,
        }
        spec = sched.spec_from_tile(TILE, t, out_dtype="fp16")
        self.assertEqual(sched.cpp_cluster_sequence(t), "ck_tile::sequence<2, 2, 1>")
        self.assertEqual(
            sched.cpp_cluster_sequence({"cluster_dim": [1, 4]}),
            "ck_tile::sequence<4, 1, 1>",
        )
        self.assertEqual(
            sched.cpp_policy_type(t),
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, false>",
        )
        self.assertEqual(
            sched.cpp_kernel_traits(spec, {"edge": False, "tail_steps": 2}),
            "ck_tile::MxTdmScheduledGemmTraits<false, false, false, false, false, 2>",
        )
        self.assertEqual(
            sched.instance_name_suffix(spec, {**t, "edge": False, "tail_steps": 2}),
            "_c2x2_lsi0_noedge_tail2",
        )


class TestParity(unittest.TestCase):
    """The Python half of parity_cases.py; test_mx_gemm_tdm_scheduled_host.cpp checks the C++ half."""

    def test_rows(self):
        for row in parity_cases.PARITY_CASES:
            with self.subTest(row=row["name"]):
                self.assertEqual(parity_cases.python_reason(row), row["reason"])

    def test_render_inc(self):
        # CMake generates the C++ half at build time: one check per row.
        inc = parity_cases.render_inc()
        for row in parity_cases.PARITY_CASES:
            with self.subTest(row=row["name"]):
                self.assertEqual(inc.count(f'("{row["name"]}", '), 1)
                self.assertIn(f'"{row["reason"]}");', inc)

    def test_every_reason_covered(self):
        rows = {row["reason"] for row in parity_cases.PARITY_CASES}
        unreachable = set(parity_cases.HOST_UNREACHABLE)
        self.assertFalse(rows & unreachable)
        self.assertEqual(rows | unreachable, set(sched.REASONS))
        self.assertFalse(parity_cases.PYTHON_ONLY & set(sched.REASONS))

    def test_enum_matches_reasons(self):
        text = _POLICY_HPP.read_text()
        enum = re.search(
            r"enum class MxTdmScheduledReason : int\s*\{(.*?)\};", text, re.S
        )
        names = re.findall(r"\b(\w+)\b", enum.group(1))
        start = text.index("mx_tdm_scheduled_reason_code(MxTdmScheduledReason r)")
        end = text.index("mx_tdm_scheduled_reason_message(MxTdmScheduledReason r)")
        cases = re.findall(
            r'case MxTdmScheduledReason::(\w+): return "(\w+)";', text[start:end]
        )
        self.assertEqual([name for name, _ in cases], names)
        self.assertEqual(tuple(code for _, code in cases), sched.REASONS)


class TestValidationUtils(unittest.TestCase):
    def test_registered_for_gfx1250_only(self):
        self.assertIn("comp_tdm_scheduled", vu.GEMM_MX_PIPELINES_BY_ARCH["gfx1250"])
        self.assertNotIn("comp_tdm_scheduled", vu.GEMM_MX_PIPELINES_BY_ARCH["gfx950"])
        self.assertIn(
            "gfx1250",
            vu.mx_tdm_scheduled_reject_reason(
                *TILE.values(), "fp4", "fp4", "fp16", "gfx950"
            ),
        )

    def test_traits(self):
        ok = vu.is_trait_combination_valid
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "direct",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertTrue(
            ok(
                "comp_tdm_scheduled",
                "tdm",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "cshuffle",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "tdm",
                "intrawave",
                True,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "tdm",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                True,
                False,
                False,
            )
        )
        self.assertFalse(ok("comp_tdm", "direct", "intrawave", False, "mx_gemm", "rcr"))

    def test_tiles(self):
        def valid(**tile):
            dims = [{**TILE, **tile}[k] for k in TILE]
            return vu.is_tile_config_valid(
                *dims,
                "fp4",
                "fp4",
                "fp16",
                "comp_tdm_scheduled",
                "rcr",
                "gfx1250",
                "mx_gemm",
            )

        self.assertTrue(valid())
        # Valid without SplitBLdsRead only: kept for the tuning_config product to decide.
        self.assertTrue(valid(tile_n=64, warp_m=1, warp_n=4))
        self.assertTrue(
            valid(tile_m=128, tile_n=16, warp_m=4, warp_n=1, warp_tile_m=16)
        )
        dims = [{**TILE, "tile_n": 64, "warp_m": 1, "warp_n": 4}[k] for k in TILE]
        self.assertIn(
            "split_b_read",
            vu.mx_tdm_scheduled_reject_reason(
                *dims, "fp4", "fp4", "fp16", "gfx1250", {}
            ),
        )
        self.assertEqual(
            vu.mx_tdm_scheduled_reject_reason(
                *dims, "fp4", "fp4", "fp16", "gfx1250", {"split_b_lds_read": False}
            ),
            "",
        )
        self.assertFalse(valid(tile_k=1024))  # LDS
        self.assertFalse(valid(warp_tile_n=32))
        self.assertFalse(valid(warp_k=2))


class TestTileEngine(unittest.TestCase):
    def test_builder_rejects_comp_tdm_scheduled(self):
        config = _MX / "configs" / "comp_tdm_scheduled_maf_gfx1250.json"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--working_path",
                    tmp,
                    "--datatype",
                    "fp4",
                    "--layout",
                    "rcr",
                    "--config_json",
                    str(config),
                    "--gpu_target",
                    "gfx1250",
                    "--list_kernels",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mx_gemm_full_benchmark.py", result.stderr)
            self.assertFalse(list(Path(tmp).iterdir()))


def _cfg(**extra):
    return {
        "datatype": "fp4",
        "layout": "rcr",
        "gpu_target": "gfx1250",
        "pipeline": "comp_tdm_scheduled",
        "tile_config": dict(TILE),
        **extra,
    }


def _kernel_config(**extra):
    return mx.MxGemmKernelConfig(
        datatype="fp4",
        gpu_target="gfx1250",
        pipeline="comp_tdm_scheduled",
        **TILE,
        **extra,
    )


class TestDispatcher(unittest.TestCase):
    def test_names(self):
        base = "mx_gemm_fp4_rcr_comp_tdm_scheduled_tdm_intrawave_False_False_False_False_256x256x512_2x2x1_32x16x128"
        self.assertEqual(codegen.kernel_name(_cfg()), base + "_c4x4")
        self.assertEqual(
            codegen.kernel_name(
                _cfg(cluster_dim=[2, 2], gl2_prefetch_distance=2, tail_steps=1)
            ),
            base + "_c2x2_pf2_tail1",
        )

    def test_rejects(self):
        with self.assertRaisesRegex(ValueError, "split_b_read"):
            codegen.kernel_name(_cfg(prefetch_lds_read=False))
        with self.assertRaisesRegex(ValueError, "only for comp_tdm_scheduled"):
            codegen.kernel_name(_cfg(pipeline="comp_tdm", cluster_dim=[2, 2]))
        for epilogue in ("cshuffle", "direct"):
            with self.assertRaisesRegex(ValueError, "epilogue"):
                codegen.kernel_name(_cfg(epilogue=epilogue))
        with self.assertRaisesRegex(ValueError, "gfx950"):
            codegen.kernel_name(_cfg(gpu_target="gfx950"))

    def test_header(self):
        with tempfile.TemporaryDirectory() as out:
            path = codegen.generate_kernel(Path(out), _cfg(cluster_dim=[2, 2]))
            header = path.read_text()
        self.assertTrue(path.name.endswith("_c2x2.hpp"))
        self.assertIn("using Cluster = ck_tile::sequence<2, 2, 1>;", header)
        self.assertIn(
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, true>",
            header,
        )
        self.assertIn("KMajorScales = true", header)
        self.assertIn("_comp_tdm_scheduled_tdm_", path.name)
        self.assertIn("ck_tile::MxTdmStoreEpilogue<P>", header)

    def test_kernel_config(self):
        cfg = _kernel_config(
            tuning={"cluster_dim": (2, 2), "gl2_prefetch_distance": 2}
        ).to_codegen_config()
        self.assertEqual(cfg["epilogue"], "tdm")
        self.assertEqual(cfg["cluster_dim"], [2, 2, 1])
        self.assertTrue(codegen.kernel_name(cfg).endswith("_c2x2_pf2"))

    def test_fallback_name_carries_the_tuning(self):
        for tuning, suffix in (
            (None, "_c4x4"),
            ({"cluster_dim": (2, 2), "gl2_prefetch_distance": 2}, "_c2x2_pf2"),
        ):
            with self.subTest(tuning=tuning):
                cfg = _kernel_config(tuning=tuning)
                name = codegen.kernel_name(cfg.to_codegen_config())
                self.assertTrue(name.endswith(suffix), name)
                self.assertTrue(cfg._fallback_name().endswith(suffix))
        # Without the spec module: a hash of the tuning, still distinct per tuning.
        with mock.patch.object(mx, "_mx_codegen", side_effect=ImportError):
            hashed = {
                _kernel_config(tuning={"cluster_dim": c})._fallback_name()
                for c in ((1, 1), (2, 2))
            }
        self.assertEqual(len(hashed), 2)

    def test_name_falls_back_per_tuning(self):
        failed = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(mx.subprocess, "run", return_value=failed):
            names = [
                _kernel_config(tuning={"cluster_dim": c}).name for c in ((1, 1), (2, 2))
            ]
        self.assertNotEqual(names[0], names[1])
        self.assertTrue(names[0].endswith("_c1x1"), names[0])
        self.assertTrue(names[1].endswith("_c2x2"), names[1])

    def test_kernel_config_tuning_keys(self):
        typo = _kernel_config(tuning={"cluster": (2, 2)})
        self.assertFalse(typo.is_valid())
        with self.assertRaisesRegex(ValueError, "unknown"):
            typo.to_codegen_config()
        for tuning, suffix in (({"cluster_dim": (2, 2)}, "_c2x2"), (None, "_c4x4")):
            with self.subTest(tuning=tuning):
                cfg = _kernel_config(tuning=tuning)
                self.assertTrue(cfg.is_valid())
                name = codegen.kernel_name(cfg.to_codegen_config())
                self.assertTrue(name.endswith(suffix), name)

    def test_tuning_config(self):
        path = _MX / "configs" / "comp_tdm_scheduled_tuning_gfx1250.json"
        tuning_config = json.loads(path.read_text())["tuning_config"]
        self.assertEqual(
            set(tuning_config), {*sched.TUNING_DEFAULTS, *sched.SHAPE_DEFAULTS}
        )
        cfgs = mx.expand_mx_sweep(path, "gfx1250", "fp4")
        self.assertTrue(cfgs)
        self.assertEqual({c.pipeline for c in cfgs}, {"comp_tdm_scheduled"})
        self.assertEqual({c.epilogue for c in cfgs}, {"tdm"})
        names = [c.name for c in cfgs]
        self.assertEqual(len(set(names)), len(names))
        problems = json.loads(
            (_MX / "configs" / "comp_tdm_scheduled_problems.json").read_text()
        )["problems"]
        self.assertTrue(problems)
        for p in problems:
            self.assertEqual(set(p), {"M", "N", "K"})

    def test_tuned_instance_configs(self):
        # Example 56's maf, medium and mab256.
        expect = {
            "maf": "256x256x512_2x2x1_32x16x128_c4x4_hplrnone_noedge",
            "medium": "128x128x1024_2x2x1_16x16x128_c4x4_hplrnone_noedge",
            "mab": "256x16x512_4x1x1_16x16x128_c1x4_pf0_lp0_hplrnone_noedge",
        }
        for case, suffix in expect.items():
            with self.subTest(case=case):
                path = _MX / "configs" / f"comp_tdm_scheduled_{case}_gfx1250.json"
                cfgs = mx.expand_mx_sweep(path, "gfx1250", "fp4")
                self.assertEqual(len(cfgs), 1)
                self.assertTrue(cfgs[0].name.endswith(suffix), cfgs[0].name)


class TestCompileFlags(unittest.TestCase):
    def test_sched_flags_only_for_comp_tdm_scheduled(self):
        flags = mx._MX_TDM_SCHEDULED_FLAGS
        for pipeline, expect in (("comp_tdm_scheduled", True), ("comp_tdm", False)):
            run_result = mock.Mock(returncode=0, stderr="")
            with mock.patch.object(mx, "_mx_codegen_flags", return_value=()):
                with mock.patch.object(
                    mx.subprocess, "run", return_value=run_result
                ) as run:
                    self.assertTrue(
                        mx._compile_kernel(
                            Path("k.hpp"), Path("k.so"), "gfx1250", pipeline
                        )
                    )
            cmd = run.call_args.args[0]
            for flag in flags[1::2]:
                self.assertEqual(flag in cmd, expect, (pipeline, flag))


if __name__ == "__main__":
    unittest.main()
