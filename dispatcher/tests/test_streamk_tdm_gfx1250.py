#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU tests for Stream-K GEMM on the gfx1250 TDM pipelines comp_tdm and
comp_tdm_v2: the shared trait gate (codegen_common.gfx1250_pipeline_reject_reason
with streamk=True plus streamk_trait_reject_reason), its agreement across
unified_gemm_codegen, arch_filter and gemm_utils.expand_sweep, and the generated
Stream-K kernel header.

Run: python3 -m pytest tests/test_streamk_tdm_gfx1250.py -v
"""

import itertools
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CODEGEN_DIR = DISPATCHER_DIR / "codegen"
sys.path.insert(0, str(CODEGEN_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"))

from arch_filter import ArchFilter, OperatorType  # noqa: E402
from codegen_common import (  # noqa: E402
    GFX1250_ONLY_PIPELINES,
    STREAMK_GFX1250_PIPELINES,
    STREAMK_REDUCTION_STRATEGIES,
    STREAMK_TDM_C_LAYOUT_REJECT_REASON,
    STREAMK_TDM_WARP_K_REJECT_REASON,
    TDM_PAD_REJECT_REASON,
    gfx1250_pipeline_reject_reason,
    streamk_trait_reject_reason,
)
from gemm_utils import expand_sweep  # noqa: E402
from gemm_vector_fallback import VectorFallback, _tile_fits  # noqa: E402

ARCH = "gfx1250"
# The pipelines this file validates. STREAMK_GFX1250_PIPELINES may hold more
# (e.g. comp_tdm_pc), whose rules are tested elsewhere.
SK_TDM_PIPELINES = ("comp_tdm", "comp_tdm_v2")
# The gfx1250-only pipelines Stream-K must reject.
SK_REJECTED_PIPELINES = tuple(
    p for p in GFX1250_ONLY_PIPELINES if p not in STREAMK_GFX1250_PIPELINES
)
EPILOGUES = ("cshuffle", "tdm", "default")
PADS = tuple(itertools.product((False, True), repeat=3))
TILE = dict(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)


def _gate_reason(
    pipeline,
    epilogue,
    reduction,
    pads=(False, False, False),
    persistent=False,
    num_waves=4,
    arch=ARCH,
    scheduler="intrawave",
    layout="crr",
    warp_k=1,
):
    """The Stream-K gate as unified_gemm_codegen applies it ("" when accepted)."""
    return gfx1250_pipeline_reject_reason(
        arch,
        pipeline,
        epilogue,
        scheduler,
        num_waves=num_waves,
        warp_tile_k=32,
        dtype_a="bf16",
        dtype_b="bf16",
        layout=layout,
        variant_supported=True,
        variant_name="stream_k",
        persistent=persistent,
        pads=pads,
        warp_k=warp_k,
        streamk=True,
    ) or streamk_trait_reject_reason(pipeline, epilogue, reduction, pads=pads)


def _expected(pipeline, epilogue, reduction, pads, num_waves, c_row=True, warp_k=1):
    """Independent statement of the rules: unpadded TDM, row-major C, warp_k=1,
    comp_tdm_v2 on 4 waves, atomic on cshuffle only (TdmEpilogue overwrites C),
    linear/tree on cshuffle or tdm, persistent or not. Legacy pipelines keep the
    cshuffle-only Stream-K rule."""
    if pipeline not in SK_TDM_PIPELINES:
        return epilogue == "cshuffle"
    if not c_row or warp_k != 1 or any(pads):
        return False
    if pipeline == "comp_tdm_v2" and num_waves != 4:
        return False
    if reduction == "atomic":
        return epilogue == "cshuffle"
    return epilogue in ("cshuffle", "tdm")


class TestStreamKTdmGate(unittest.TestCase):
    def test_enabled_pipelines(self):
        self.assertLessEqual(set(SK_TDM_PIPELINES), set(STREAMK_GFX1250_PIPELINES))
        self.assertIn("comp_async", SK_REJECTED_PIPELINES)
        self.assertEqual(STREAMK_REDUCTION_STRATEGIES, ("atomic", "linear", "tree"))

    def test_trait_matrix(self):
        for combo in itertools.product(
            SK_TDM_PIPELINES,
            EPILOGUES,
            STREAMK_REDUCTION_STRATEGIES,
            PADS,
            (False, True),
            (4, 8),
            ("crr", "rcr", "rrc", "ccc"),
            (1, 2),
        ):
            (
                pipeline,
                epilogue,
                reduction,
                pads,
                persistent,
                num_waves,
                layout,
                warp_k,
            ) = combo
            with self.subTest(combo=combo):
                reason = _gate_reason(
                    pipeline,
                    epilogue,
                    reduction,
                    pads,
                    persistent,
                    num_waves,
                    layout=layout,
                    warp_k=warp_k,
                )
                self.assertEqual(
                    not reason,
                    _expected(
                        pipeline,
                        epilogue,
                        reduction,
                        pads,
                        num_waves,
                        layout[2] == "r",
                        warp_k,
                    ),
                    reason,
                )

    def test_reject_reasons(self):
        for pipeline in SK_TDM_PIPELINES:
            self.assertIn("overwrites C", _gate_reason(pipeline, "tdm", "atomic"))
            self.assertIn(
                "requires epilogue=cshuffle or tdm",
                _gate_reason(pipeline, "default", "linear"),
            )
            self.assertEqual(
                _gate_reason(pipeline, "cshuffle", "linear", pads=(True, True, False)),
                TDM_PAD_REJECT_REASON,
            )
            self.assertIn(
                "requires gfx1250", _gate_reason(pipeline, "tdm", "tree", arch="gfx950")
            )
            self.assertIn(
                "intrawave",
                _gate_reason(pipeline, "tdm", "tree", scheduler="interwave"),
            )
            # StreamKKernel static_asserts a row-major C, and neither epilogue
            # reduces across K waves.
            for epilogue, reduction in (("cshuffle", "atomic"), ("tdm", "tree")):
                self.assertEqual(
                    _gate_reason(pipeline, epilogue, reduction, layout="crc"),
                    STREAMK_TDM_C_LAYOUT_REJECT_REASON,
                )
                self.assertEqual(
                    _gate_reason(pipeline, epilogue, reduction, warp_k=2),
                    STREAMK_TDM_WARP_K_REJECT_REASON,
                )
        self.assertEqual(
            _gate_reason("comp_tdm_v2", "tdm", "tree", num_waves=8),
            "comp_tdm_v2 requires exactly 4 waves",
        )

    def test_unlisted_gfx1250_pipelines_rejected(self):
        # Same reason text the codegen gave on develop (variant_supported=False).
        for pipeline, epilogue, reduction in itertools.product(
            SK_REJECTED_PIPELINES, EPILOGUES, STREAMK_REDUCTION_STRATEGIES
        ):
            if pipeline == "comp_async" and epilogue == "tdm":
                continue  # rejected earlier: the tdm epilogue needs a TDM pipeline
            with self.subTest(
                pipeline=pipeline, epilogue=epilogue, reduction=reduction
            ):
                self.assertEqual(
                    _gate_reason(pipeline, epilogue, reduction),
                    f"pipeline={pipeline} is not supported for stream_k",
                )
                self.assertIn(
                    "not supported for stream_k",
                    streamk_trait_reject_reason(pipeline, epilogue, reduction),
                )

    def test_other_variants_unchanged(self):
        def reason(pipeline, epilogue, persistent=False, variant="standard"):
            return gfx1250_pipeline_reject_reason(
                ARCH,
                pipeline,
                epilogue,
                "intrawave",
                num_waves=4,
                warp_tile_k=32,
                dtype_a="bf16",
                dtype_b="bf16",
                layout="crr",
                variant_supported=variant in ("standard", "batched", "preshuffle"),
                variant_name=variant,
                persistent=persistent,
                pads=(False, False, False),
            )

        for pipeline in SK_TDM_PIPELINES + ("comp_tdm_pc",):
            self.assertEqual(reason(pipeline, "tdm"), "")
            self.assertIn("persistent", reason(pipeline, "tdm", persistent=True))
            for variant in ("multi_d", "grouped", "multi_abd"):
                self.assertIn("not supported", reason(pipeline, "tdm", variant=variant))
        for pipeline in SK_TDM_PIPELINES:
            self.assertIn("requires epilogue=tdm", reason(pipeline, "cshuffle"))
        # Plain comp_tdm_pc GEMM also takes cshuffle (GEMM_TDM_CSHUFFLE_PIPELINES),
        # still without the persistent kernel.
        self.assertEqual(reason("comp_tdm_pc", "cshuffle"), "")
        self.assertIn("persistent", reason("comp_tdm_pc", "cshuffle", persistent=True))

    def test_legacy_pipelines_unchanged(self):
        for arch, pipeline, reduction, pads in itertools.product(
            ("gfx942", "gfx950", ARCH),
            ("compv3", "compv4", "mem"),
            STREAMK_REDUCTION_STRATEGIES,
            PADS,
        ):
            with self.subTest(arch=arch, pipeline=pipeline, reduction=reduction):
                self.assertEqual(
                    _gate_reason(pipeline, "cshuffle", reduction, pads, arch=arch), ""
                )
                self.assertTrue(_gate_reason(pipeline, "default", reduction, arch=arch))
                self.assertTrue(_gate_reason(pipeline, "tdm", reduction, arch=arch))


class TestStreamKTdmGateEquivalence(unittest.TestCase):
    """unified_gemm_codegen, gemm_utils.expand_sweep and arch_filter agree."""

    PIPELINES = SK_TDM_PIPELINES + ("comp_tdm_pc", "comp_async", "compv3")
    # (warp_m, warp_k) with warp_n = 2 that arch_filter allows on gfx1250; 1x2x2 is
    # the 4-wave map with K waves.
    WAVE_MAPS = ((2, 1), (4, 1), (1, 2))
    CONFIG = {
        "tile_config": {
            "tile_m": [128],
            "tile_n": [128],
            "tile_k": [64],
            "warp_m": [1, 2, 4],
            "warp_n": [2],
            "warp_k": [1, 2],
            "warp_tile_m": [16],
            "warp_tile_n": [16],
            "warp_tile_k": [32],
        },
        "trait_config": {
            "pipeline": list(PIPELINES),
            "epilogue": list(EPILOGUES),
            "scheduler": ["intrawave"],
            "pad_m": [False, True],
            "pad_n": [False, True],
            "pad_k": [False],
            "persistent": [False, True],
        },
        "streamk_config": {"reduction_strategy": list(STREAMK_REDUCTION_STRATEGIES)},
    }
    NAME_RE = re.compile(
        r"gemm_bf16_crr_(?P<pipeline>\w+?)_(?P<epilogue>cshuffle|tdm|default)"
        r"_intrawave_(?P<pad_m>True|False)_(?P<pad_n>True|False)_(?:True|False)"
        r"_(?P<persistent>True|False)_128x128x64_(?P<warp_m>\d)x2x(?P<warp_k>\d)"
        r"_16x16x32"
        r"_streamk(?:_(?P<reduction>linear|tree))?$"
    )

    @classmethod
    def _generate(cls, layout):
        """(codegen output dir, kernel names, expand_sweep configs) for ``layout``."""
        out = cls.tmp / f"out_{layout}"
        cmd = [
            sys.executable,
            str(CODEGEN_DIR / "unified_gemm_codegen.py"),
            "--output-dir",
            str(out),
            "--datatype",
            "bf16",
            "--layout",
            layout,
            "--gpu-target",
            ARCH,
            "--config",
            str(cls.cfg),
            "--variants",
            "stream_k",
            "--no-parallel",
        ]
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(CODEGEN_DIR))
        names = sorted(p.stem for p in out.glob("*.hpp"))
        sweep = expand_sweep(
            str(cls.te_cfg), ARCH, dtype="bf16", layout=layout, variant="stream_k"
        )
        return out, names, sweep

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="streamk_tdm_"))
        cls.cfg = cls.tmp / "cfg.json"
        cls.cfg.write_text(json.dumps(cls.CONFIG))
        # expand_sweep reads the Tile Engine value-list format.
        cls.te_cfg = cls.tmp / "te_cfg.json"
        cls.te_cfg.write_text(
            json.dumps(
                {
                    section: {key: {"values": v} for key, v in entries.items()}
                    for section, entries in cls.CONFIG.items()
                }
            )
        )
        cls.out, cls.names, cls.sweep = cls._generate("crr")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _parsed(self):
        combos = set()
        for name in self.names:
            m = self.NAME_RE.fullmatch(name)
            self.assertIsNotNone(m, name)
            combos.add(
                (
                    m["pipeline"],
                    m["epilogue"],
                    m["reduction"] or "atomic",
                    (m["pad_m"] == "True", m["pad_n"] == "True", False),
                    m["persistent"] == "True",
                    int(m["warp_m"]) * 2 * int(m["warp_k"]),
                    int(m["warp_k"]),
                )
            )
        return combos

    def test_codegen_matches_rules(self):
        # comp_async is never built; a listed comp_tdm_pc follows its own rules.
        checked = SK_TDM_PIPELINES + ("compv3",)
        self.assertFalse(
            any(c[0] in SK_REJECTED_PIPELINES for c in self._parsed()),
            SK_REJECTED_PIPELINES,
        )
        expected = {
            (
                pipeline,
                epilogue,
                reduction,
                (pm, pn, False),
                persistent,
                warp_m * 2 * warp_k,
                warp_k,
            )
            for pipeline, epilogue, reduction, pm, pn, persistent, (warp_m, warp_k) in (
                itertools.product(
                    checked,
                    EPILOGUES,
                    STREAMK_REDUCTION_STRATEGIES,
                    (False, True),
                    (False, True),
                    (False, True),
                    self.WAVE_MAPS,
                )
            )
            if _expected(
                pipeline,
                epilogue,
                reduction,
                (pm, pn, False),
                warp_m * 2 * warp_k,
                warp_k=warp_k,
            )
        }
        self.assertEqual({c for c in self._parsed() if c[0] in checked}, expected)
        self.assertTrue(any(c[0] == "compv3" and c[6] == 2 for c in self._parsed()))
        # Every TDM kernel is unpadded; tdm-epilogue kernels are linear/tree only.
        tdm_epi = [n for n in self.names if "_tdm_intrawave_" in n]
        self.assertTrue(tdm_epi)
        self.assertFalse(any(n.endswith("_streamk") for n in tdm_epi))

    def test_codegen_matches_expand_sweep(self):
        self.assertEqual(sorted(c.name for c in self.sweep), self.names)

    def test_column_major_c_has_no_stream_k_kernels(self):
        # StreamKKernel static_asserts a row-major C for the TDM pipelines, and the
        # CShuffle epilogue stores only a row-major C, so a column-major C has no
        # Stream-K kernel at all: the codegen emits none and expand_sweep says why.
        with self.assertRaisesRegex(ValueError, "row-major C"):
            self._generate("crc")
        self.assertFalse(list((self.tmp / "out_crc").glob("*.hpp")))

    def test_tdm_cshuffle_pairing_needs_c_tile_multiple(self):
        # The TDM loads clip A and B, but a CShuffle store does not clip C's
        # contiguous dimension, so the Tile Engine pairing must keep that extent a
        # tile multiple; the TDM epilogue clips C too (e.g. N = 448 on a 128-wide tile).
        for pipeline in SK_TDM_PIPELINES:
            cfgs = {
                c.epilogue: c
                for c in self.sweep
                if c.pipeline == pipeline and c.wave_str == "2x2x1"
            }
            cshuffle, tdm = cfgs["cshuffle"], cfgs["tdm"]
            self.assertTrue(_tile_fits(cshuffle, {"m": 192, "n": 384, "k": 393216}))
            self.assertFalse(_tile_fits(cshuffle, {"m": 192, "n": 448, "k": 393216}))
            self.assertTrue(_tile_fits(tdm, {"m": 192, "n": 448, "k": 393216}))

    def test_tdm_native_vector_sizes(self):
        # TDM loads A and B at width 1 whatever the epilogue; CShuffle still stores C
        # at its own width (8 bf16 for a 16-wide warp tile on 2 N waves).
        for cfg in self.sweep:
            if cfg.pipeline in SK_TDM_PIPELINES:
                with self.subTest(name=cfg.name):
                    want_c = 1 if cfg.epilogue == "tdm" else 8
                    self.assertEqual(cfg.effective_vector_sizes, (1, 1, want_c))

    def test_tdm_cshuffle_pairs_misaligned_a(self):
        # crr: A is column-major, so M is its contiguous extent. M = 100 is not a
        # multiple of 8, but TDM loads A at width 1, so the kernel still serves it.
        cfgs = [
            c
            for c in self.sweep
            if c.pipeline in SK_TDM_PIPELINES and c.wave_str == "2x2x1"
        ]
        self.assertTrue(cfgs)
        problems = [dict(M=100, N=384, K=512), dict(M=100, N=448, K=512)]
        fallback = VectorFallback(problems, "crr", "bf16", "stream_k")
        pairs = fallback.pairs(problems, [(c, None) for c in cfgs])
        cshuffle = {i for i, c in enumerate(cfgs) if c.epilogue == "cshuffle"}
        self.assertTrue(cshuffle)
        self.assertEqual(set(pairs[0]), set(range(len(cfgs))))
        self.assertEqual(set(pairs[1]), set(range(len(cfgs))) - cshuffle)

    def test_arch_filter_matches_shared_gate(self):
        # arch_filter has no reduction strategy: it applies the arch/pipeline half
        # of the gate (gfx1250_pipeline_reject_reason with streamk=True).
        filt = ArchFilter(ARCH)
        for pipeline, epilogue, pads, (warp_m, warp_k), layout in itertools.product(
            SK_TDM_PIPELINES + ("comp_tdm_pc", "comp_async"),
            EPILOGUES,
            PADS,
            self.WAVE_MAPS,
            ("rcr", "rcc"),
        ):
            num_waves = warp_m * 2 * warp_k
            reason = gfx1250_pipeline_reject_reason(
                ARCH,
                pipeline,
                epilogue,
                "intrawave",
                num_waves=num_waves,
                warp_tile_k=32,
                dtype_a="fp16",
                dtype_b="fp16",
                layout=layout,
                pads=pads,
                tile_mn=(128, 128),
                warp_k=warp_k,
                streamk=True,
            )
            valid = filt.is_kernel_valid(
                warp_m=warp_m,
                warp_n=2,
                pipeline=pipeline,
                epilogue=epilogue,
                layout=layout,
                operator=OperatorType.GEMM_STREAMK,
                pad_m=pads[0],
                pad_n=pads[1],
                pad_k=pads[2],
                **{**TILE, "warp_k": warp_k},
            )
            with self.subTest(
                pipeline=pipeline,
                epilogue=epilogue,
                pads=pads,
                waves=(warp_m, 2, warp_k),
                layout=layout,
            ):
                self.assertEqual(valid, not reason, reason)

    def test_generated_headers(self):
        def text(name):
            return (self.out / f"{name}.hpp").read_text()

        base = (
            "gemm_bf16_crr_{}_intrawave_False_False_False_{}_128x128x64_2x2x1_16x16x32"
        )
        v1_tree = text(base.format("comp_tdm_tdm", "True") + "_streamk_tree")
        self.assertIn('#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"', v1_tree)
        self.assertIn("GemmPipelineAgBgCrCompTDMV1<SkUniversalGemmProblem>", v1_tree)
        self.assertIn("using GemmEpilogue = TdmEpilogue<EpilogueProblem>", v1_tree)
        self.assertIn("StreamKReductionStrategy::Tree", v1_tree)
        self.assertIn("UsePersistentKernel = true", v1_tree)
        self.assertIn("DoubleSmemBuffer = true", v1_tree)
        self.assertIn(
            "StreamKKernel<SkStreamKTilePartitioner, SkGemmPipeline, GemmEpilogue>",
            v1_tree,
        )
        v2_atomic = text(base.format("comp_tdm_v2_cshuffle", "False") + "_streamk")
        self.assertIn("GemmPipelineAgBgCrCompTDMV2<SkUniversalGemmProblem>", v2_atomic)
        self.assertIn(
            "using GemmEpilogue = CShuffleEpilogue<EpilogueProblem>", v2_atomic
        )
        self.assertNotIn("TdmEpilogue<", v2_atomic)
        self.assertIn("StreamKReductionStrategy::Atomic", v2_atomic)
        self.assertIn("UsePersistentKernel = false", v2_atomic)


if __name__ == "__main__":
    unittest.main()
