#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only equivalence of the gfx1250 pipeline gates (non-MX comp_async,
comp_tdm, comp_tdm_v2 and the tdm epilogue).

The Tile Engine keeps its own copy of these rules
(tile_engine/ops/gemm/gemm_validation_utils.py); the dispatcher uses
codegen_common.gfx1250_pipeline_reject_reason through unified_gemm_codegen,
python/gemm_utils and arch_filter. This test enumerates the gfx1250 trait x
tile space and asserts that both sides accept exactly the same configs.

Only the gfx1250-gated pipelines/epilogue are compared. Tile rules that are not
specific to them (warp-tile tables, generic alignment) are factored out: a tile
takes part only when compv3 with the same dtype/layout is legal on both sides,
so any disagreement is a genuine gap in the gfx1250 gate.

Run: python3 -m pytest tests/test_gfx1250_gate_equivalence.py -v
"""

import itertools
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
TE_GEMM_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(TE_GEMM_DIR))

try:
    import gemm_validation_utils as te  # noqa: E402
except ImportError:  # pragma: no cover - TE tree not present
    te = None

from arch_filter import ArchFilter  # noqa: E402
from codegen_common import CommonTypeMappings  # noqa: E402

try:
    import gemm_utils  # noqa: E402
except ImportError:  # pragma: no cover - numpy missing
    gemm_utils = None

ARCH = "gfx1250"
TE_PREFIX = "gemm_universal"
PIPELINES = ("comp_async", "comp_tdm", "comp_tdm_v2", "compv3", "compv4", "mem")
GATED_PIPELINES = ("comp_async", "comp_tdm", "comp_tdm_v2")
EPILOGUES = ("cshuffle", "default", "tdm")
SCHEDULERS = ("intrawave", "interwave")
LAYOUTS = ("rcr", "rrr", "crr", "ccr")
PADS = tuple(itertools.product((False, True), repeat=3))
DTYPES = ("fp16", "bf16", "fp8", "bf8")


def _gated(pipeline, epilogue):
    return pipeline in GATED_PIPELINES or epilogue == "tdm"


def _te_accepts(pipeline, epilogue, scheduler, persistent, layout, pads, dtype, t):
    pad_m, pad_n, pad_k = pads
    if not te.is_trait_combination_valid(
        pipeline,
        epilogue,
        scheduler,
        persistent,
        kernel_name_prefix=TE_PREFIX,
        layout=layout,
        pad_m=pad_m,
        pad_n=pad_n,
        pad_k=pad_k,
    ):
        return False
    return te.is_tile_config_valid(
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
        CommonTypeMappings.get_output_dtype(dtype),
        pipeline,
        layout,
        ARCH,
        kernel_name_prefix=TE_PREFIX,
    )


def _arch_filter_accepts(filt, pipeline, epilogue, scheduler, layout, pads, dtype, t):
    pad_m, pad_n, pad_k = pads
    return filt.is_kernel_valid(
        datatype_a=dtype,
        datatype_b=dtype,
        datatype_c=CommonTypeMappings.get_output_dtype(dtype),
        pipeline=pipeline,
        epilogue=epilogue,
        scheduler=scheduler,
        layout=layout,
        pad_m=pad_m,
        pad_n=pad_n,
        pad_k=pad_k,
        **t,
    )


def _gemm_utils_accepts(
    pipeline, epilogue, scheduler, persistent, layout, pads, dtype, t
):
    pad_m, pad_n, pad_k = pads
    return gemm_utils._gfx1250_pipeline_supported(
        pipeline,
        scheduler,
        epilogue,
        persistent,
        t["warp_m"],
        t["warp_n"],
        t["warp_k"],
        ARCH,
        "standard",
        pad_m=pad_m,
        pad_n=pad_n,
        pad_k=pad_k,
        layout=layout,
        dtype=dtype,
        warp_tile_k=t["warp_tile_k"],
    )


def _tile(tm, tn, tk, wm, wn, wk, wtm, wtn, wtk):
    return dict(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=wk,
        warp_tile_m=wtm,
        warp_tile_n=wtn,
        warp_tile_k=wtk,
    )


# Trait sweep tiles: 4 waves and 8 waves, warp_tile_k below and at the 8-bit
# comp_async minimum.
TRAIT_TILES = (
    _tile(128, 128, 128, 2, 2, 1, 16, 16, 32),
    _tile(128, 128, 128, 2, 2, 1, 16, 16, 128),
    _tile(256, 128, 128, 4, 2, 1, 16, 16, 32),
    _tile(256, 128, 128, 4, 2, 1, 16, 16, 128),
)


def _tile_space():
    """The gfx1250 tile x wave x warp-tile space swept for the LDS / C-tile /
    wave / warp_tile_k rules."""
    waves = te.WARP_SUPPORTED_COMBINATIONS[ARCH]
    for tm, tn, tk in itertools.product(
        (64, 128, 256, 512, 1024), (64, 128, 256, 512), (32, 64, 128, 256)
    ):
        for wm, wn, wk in waves:
            for wtk in (32, 64, 128):
                t = _tile(tm, tn, tk, wm, wn, wk, 16, 16, wtk)
                if tm % (wm * 16) or tn % (wn * 16) or tk % (wk * wtk):
                    continue
                yield t


# Trait combinations that are legal for the gated pipelines, one per pipeline.
LEGAL_TRAITS = {
    "comp_async": ("cshuffle", "intrawave", (True, True, True)),
    "comp_tdm": ("tdm", "intrawave", (False, False, False)),
    "comp_tdm_v2": ("tdm", "intrawave", (False, False, False)),
}


@unittest.skipIf(te is None, "tile_engine/ops/gemm not importable")
class TestGfx1250GateEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filt = ArchFilter(ARCH)

    def _baseline_ok(self, layout, dtype, t, pads):
        """compv3 legal on both sides: the tile passes every rule that is not
        specific to the gfx1250 pipelines."""
        return _te_accepts(
            "compv3", "cshuffle", "intrawave", False, layout, pads, dtype, t
        ) and _arch_filter_accepts(
            self.filt, "compv3", "cshuffle", "intrawave", layout, pads, dtype, t
        )

    def test_trait_space(self):
        """Every pipeline x epilogue x scheduler x persistent x layout x pads x
        dtype on a few tiles."""
        mismatches = []
        compared = accepted = 0
        for t, dtype, layout, pads in itertools.product(
            TRAIT_TILES, DTYPES, LAYOUTS, PADS
        ):
            if not self._baseline_ok(layout, dtype, t, pads):
                continue
            for pipe, epi, sched, persistent in itertools.product(
                PIPELINES, EPILOGUES, SCHEDULERS, (False, True)
            ):
                if not _gated(pipe, epi):
                    continue
                want = _te_accepts(pipe, epi, sched, persistent, layout, pads, dtype, t)
                got_gate = gemm_utils is None or _gemm_utils_accepts(
                    pipe, epi, sched, persistent, layout, pads, dtype, t
                )
                # KernelConfig cannot express the persistent trait; the codegen
                # and gemm_utils enforce the TDM no-persistent rule.
                got_filter = _arch_filter_accepts(
                    self.filt, pipe, epi, sched, layout, pads, dtype, t
                )
                got = got_gate and got_filter
                compared += 1
                accepted += want
                if persistent:
                    if got != want and gemm_utils is not None:
                        mismatches.append(
                            (
                                pipe,
                                epi,
                                sched,
                                persistent,
                                layout,
                                pads,
                                dtype,
                                t,
                                want,
                                got,
                            )
                        )
                    continue
                if not (
                    want == got_filter and (gemm_utils is None or want == got_gate)
                ):
                    mismatches.append(
                        (
                            pipe,
                            epi,
                            sched,
                            persistent,
                            layout,
                            pads,
                            dtype,
                            t,
                            want,
                            got_gate,
                            got_filter,
                        )
                    )
        self.assertGreater(compared, 1000)
        self.assertGreater(accepted, 0)
        self.assertEqual(
            mismatches[:10], [], f"{len(mismatches)} mismatches of {compared}"
        )

    def test_tile_space(self):
        """Every tile x wave x warp_tile_k x dtype for the legal trait of each
        gated pipeline (LDS incl. TDM row padding, TDM C tile, 4-wave v2 and
        8-bit warp_tile_k rules)."""
        mismatches = []
        compared = accepted = rejected = 0
        for t in _tile_space():
            for dtype in DTYPES:
                if not self._baseline_ok("rcr", dtype, t, (True, True, True)):
                    continue
                for pipe, (epi, sched, pads) in LEGAL_TRAITS.items():
                    want = _te_accepts(pipe, epi, sched, False, "rcr", pads, dtype, t)
                    got = _arch_filter_accepts(
                        self.filt, pipe, epi, sched, "rcr", pads, dtype, t
                    )
                    if gemm_utils is not None:
                        got = got and _gemm_utils_accepts(
                            pipe, epi, sched, False, "rcr", pads, dtype, t
                        )
                    compared += 1
                    accepted += want
                    rejected += not want
                    if want != got:
                        mismatches.append((pipe, dtype, t, want, got))
        self.assertGreater(compared, 1000)
        self.assertGreater(accepted, 0)
        self.assertGreater(rejected, 0)
        self.assertEqual(
            mismatches[:10], [], f"{len(mismatches)} mismatches of {compared}"
        )


if __name__ == "__main__":
    unittest.main()
