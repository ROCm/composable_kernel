#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only equivalence of the gfx1250 pipeline gates (non-MX comp_async,
comp_tdm, comp_tdm_v2, comp_tdm_pc, comp_tdm_v3, comp_tdm_ra and the tdm / tdmstaged
epilogues).

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

from arch_filter import ArchFilter, OperatorType  # noqa: E402
from codegen_common import (  # noqa: E402
    TDM_EPILOGUES,
    CommonTypeMappings,
    comp_tdm_v3_reject_reason,
    preshuffle_pipeline_reject_reason,
    split_pipeline_ring,
    valid_warp_configs,
)

try:
    import gemm_utils  # noqa: E402
except ImportError:  # pragma: no cover - numpy missing
    gemm_utils = None

try:
    import ctypes_utils  # noqa: E402
except ImportError:  # pragma: no cover - numpy missing
    ctypes_utils = None

ARCH = "gfx1250"
TE_PREFIX = "gemm_universal"
PIPELINES = (
    "comp_async",
    "comp_tdm",
    "comp_tdm_v2",
    "comp_tdm_pc",
    "comp_tdm_v3",
    "comp_tdm_ra",
    "compv3",
    "compv4",
    "mem",
)
GATED_PIPELINES = (
    "comp_async",
    "comp_tdm",
    "comp_tdm_v2",
    "comp_tdm_pc",
    "comp_tdm_v3",
    "comp_tdm_ra",
)
EPILOGUES = ("cshuffle", "default") + TDM_EPILOGUES
SCHEDULERS = ("intrawave", "interwave")
# The *c layouts have a column-major C, which tdmstaged and comp_tdm_v3 reject.
LAYOUTS = ("rcr", "rrr", "crr", "ccr", "rcc", "rrc", "crc", "ccc")
PADS = tuple(itertools.product((False, True), repeat=3))
DTYPES = ("fp16", "bf16", "fp8", "bf8")


def _gated(pipeline, epilogue):
    return pipeline in GATED_PIPELINES or epilogue in TDM_EPILOGUES


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
        warp_tile_m=t["warp_tile_m"],
        warp_tile_n=t["warp_tile_n"],
        tile_mn=(t["tile_m"], t["tile_n"]),
        tile_k=t["tile_k"],
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
# comp_async minimum, plus the fp8/bf8-only 32x32x128 warp tile.
TRAIT_TILES = (
    _tile(128, 128, 128, 2, 2, 1, 16, 16, 32),
    _tile(128, 128, 128, 2, 2, 1, 16, 16, 128),
    _tile(256, 128, 128, 4, 2, 1, 16, 16, 32),
    _tile(256, 128, 128, 4, 2, 1, 16, 16, 128),
    _tile(128, 128, 256, 2, 2, 1, 32, 32, 128),
    _tile(256, 256, 256, 4, 2, 1, 32, 32, 128),
)


def _tile_space():
    """The gfx1250 tile x wave x warp-tile space swept for the LDS / C-tile /
    wave / warp_tile_k rules."""
    waves = te.WARP_SUPPORTED_COMBINATIONS[ARCH]
    for tm, tn, tk in itertools.product(
        (64, 128, 256, 512, 1024), (64, 128, 256, 512), (32, 64, 128, 256)
    ):
        for wm, wn, wk in waves:
            for wtmn, wtk in ((16, 32), (16, 64), (16, 128), (32, 128)):
                t = _tile(tm, tn, tk, wm, wn, wk, wtmn, wtmn, wtk)
                if tm % (wm * wtmn) or tn % (wn * wtmn) or tk % (wk * wtk):
                    continue
                yield t


# Trait combinations that are legal for the gated pipelines, one per pipeline.
LEGAL_TRAITS = {
    "comp_async": ("cshuffle", "intrawave", (True, True, True)),
    "comp_tdm": ("tdm", "intrawave", (False, False, False)),
    "comp_tdm_v2": ("tdm", "intrawave", (False, False, False)),
    "comp_tdm_pc": ("tdm", "intrawave", (False, False, False)),
    "comp_tdm_v3": ("cshuffle", "intrawave", (False, False, False)),
    "comp_tdm_ra": ("tdm", "intrawave", (False, False, False)),
}


@unittest.skipIf(te is None, "tile_engine/ops/gemm not importable")
class TestGfx1250GateEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filt = ArchFilter(ARCH)

    def test_v3_rule_parity(self):
        for dtype_a, dtype_b, waves, layout, tile_k in itertools.product(
            DTYPES + ("fp32",),
            DTYPES + ("fp32",),
            ((2, 2, 1), (4, 4, 1), (4, 2, 1)),
            LAYOUTS + ("",),
            (None, 64, 96, 192),
        ):
            with self.subTest(
                dtypes=(dtype_a, dtype_b), waves=waves, layout=layout, tile_k=tile_k
            ):
                self.assertEqual(
                    te.gfx1250_tdm_v3_reject_reason(
                        "comp_tdm_v3",
                        dtype_a,
                        dtype_b,
                        *waves,
                        layout=layout,
                        tile_k=tile_k,
                        warp_tile_k=32,
                    ),
                    comp_tdm_v3_reject_reason(
                        dtype_a, dtype_b, waves, layout, tile_k, 32
                    ),
                )

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
                # gemm_utils gates traits and waves only; the LDS budget (which
                # rejects some of these tiles for the four-slot comp_tdm_pc) is
                # the arch filter's, so gemm_utils is checked through the
                # combined result.
                if not (want == got_filter and (gemm_utils is None or want == got)):
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

    def test_8bit_32x32x128_compared(self):
        """The fp8/bf8 32x32x128 warp tile passes the shared rules on both
        sides, so the sweeps above compare it; other dtypes reject it. Both
        sides reject comp_async with it (does not compile) and accept the TDM
        pipelines."""
        t = TRAIT_TILES[-2]
        pads = (True, True, True)
        for dtype in ("fp8", "bf8"):
            self.assertTrue(self._baseline_ok("rcr", dtype, t, pads), dtype)
            for pipe, (epi, sched, legal_pads) in LEGAL_TRAITS.items():
                want = pipe != "comp_async"
                self.assertEqual(
                    _te_accepts(pipe, epi, sched, False, "rcr", legal_pads, dtype, t),
                    want,
                    (dtype, pipe),
                )
                self.assertEqual(
                    _arch_filter_accepts(
                        self.filt, pipe, epi, sched, "rcr", legal_pads, dtype, t
                    ),
                    want,
                    (dtype, pipe),
                )
                if gemm_utils is not None:
                    self.assertEqual(
                        _gemm_utils_accepts(
                            pipe, epi, sched, False, "rcr", legal_pads, dtype, t
                        ),
                        want,
                        (dtype, pipe),
                    )
        for dtype in ("fp16", "bf16"):
            self.assertFalse(
                _arch_filter_accepts(
                    self.filt, "compv3", "cshuffle", "intrawave", "rcr", pads, dtype, t
                ),
                dtype,
            )

    def test_tile_space(self):
        """Every tile x wave x warp_tile_k x dtype for the legal traits of each
        gated pipeline, both TDM epilogues for the TDM pipelines (LDS incl. TDM
        row padding, TDM C tile, 4-wave v2, producer/consumer accumulator,
        8-bit warp_tile_k and warp_tile_m/n rules)."""
        mismatches = []
        compared = accepted = rejected = 0
        for t in _tile_space():
            for dtype in DTYPES:
                if not self._baseline_ok("rcr", dtype, t, (True, True, True)):
                    continue
                for pipe, (legal_epi, sched, pads) in LEGAL_TRAITS.items():
                    for epi in TDM_EPILOGUES if legal_epi == "tdm" else (legal_epi,):
                        want = _te_accepts(
                            pipe, epi, sched, False, "rcr", pads, dtype, t
                        )
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
                            mismatches.append((pipe, epi, dtype, t, want, got))
        self.assertGreater(compared, 1000)
        self.assertGreater(accepted, 0)
        self.assertGreater(rejected, 0)
        self.assertEqual(
            mismatches[:10], [], f"{len(mismatches)} mismatches of {compared}"
        )

    def test_comp_tdm_v3_wave_maps(self):
        """comp_tdm_v3 on its own wave maps (incl. [4, 4, 1], which no other
        pipeline has, so the compv3 baseline cannot vet those tiles): every
        layout x dtype x tile x warp tile, both sides directly."""
        epi, sched, pads = LEGAL_TRAITS["comp_tdm_v3"]
        mismatches = []
        compared = accepted = 0
        for tm, tn, tk, (wm, wn), (wtmn, wtk), dtype, layout in itertools.product(
            (64, 128, 256),
            (64, 128, 256),
            (64, 128, 192, 256),
            ((2, 2), (4, 4), (4, 2)),
            ((16, 32), (16, 128), (32, 128)),
            DTYPES,
            LAYOUTS,
        ):
            if tm % (wm * wtmn) or tn % (wn * wtmn) or tk % wtk:
                continue
            # The Tile Engine has no gfx1250 warp-tile table (the compv3 baseline
            # factors that out elsewhere), so only the listed warp tiles compare.
            if [wtmn, wtmn, wtk] not in valid_warp_configs(ARCH, dtype):
                continue
            t = _tile(tm, tn, tk, wm, wn, 1, wtmn, wtmn, wtk)
            want = _te_accepts("comp_tdm_v3", epi, sched, False, layout, pads, dtype, t)
            got = _arch_filter_accepts(
                self.filt, "comp_tdm_v3", epi, sched, layout, pads, dtype, t
            )
            if gemm_utils is not None:
                got = got and _gemm_utils_accepts(
                    "comp_tdm_v3", epi, sched, False, layout, pads, dtype, t
                )
            compared += 1
            accepted += want
            if want != got:
                mismatches.append((dtype, layout, t, want, got))
        self.assertGreater(compared, 500)
        self.assertGreater(accepted, 0)
        self.assertEqual(
            mismatches[:10], [], f"{len(mismatches)} mismatches of {compared}"
        )

    def test_ring_tile_space(self):
        """Ring variants: the TE takes the variant name, the dispatcher the
        split ring (num_slots / ring_param), over the whole tile space."""
        variants = (
            "comp_tdm_ra_s3r2",
            "comp_tdm_ra_s2r1",
            "comp_tdm_ra_s4r1",
            "comp_tdm_pc_s2l1",
        )
        mismatches = []
        accepted = dict.fromkeys(variants, 0)
        rejected = dict.fromkeys(variants, 0)
        pads = (False, False, False)
        for t in _tile_space():
            for dtype in DTYPES:
                if not self._baseline_ok("rcr", dtype, t, (True, True, True)):
                    continue
                for name in variants:
                    base, slots, param = split_pipeline_ring(name)
                    want = _te_accepts(
                        name, "tdm", "intrawave", False, "rcr", pads, dtype, t
                    )
                    got = self.filt.is_kernel_valid(
                        datatype_a=dtype,
                        datatype_b=dtype,
                        datatype_c=CommonTypeMappings.get_output_dtype(dtype),
                        pipeline=base,
                        epilogue="tdm",
                        scheduler="intrawave",
                        layout="rcr",
                        pad_m=False,
                        pad_n=False,
                        pad_k=False,
                        num_slots=slots,
                        ring_param=param,
                        **t,
                    )
                    if gemm_utils is not None:
                        got = got and gemm_utils._gfx1250_pipeline_supported(
                            base,
                            "intrawave",
                            "tdm",
                            False,
                            t["warp_m"],
                            t["warp_n"],
                            t["warp_k"],
                            ARCH,
                            "standard",
                            layout="rcr",
                            dtype=dtype,
                            warp_tile_k=t["warp_tile_k"],
                            warp_tile_m=t["warp_tile_m"],
                            warp_tile_n=t["warp_tile_n"],
                            tile_mn=(t["tile_m"], t["tile_n"]),
                            ring=(slots, param),
                        )
                    accepted[name] += want
                    rejected[name] += not want
                    if want != got:
                        mismatches.append((name, dtype, t, want, got))
        for name in variants:
            self.assertGreater(accepted[name], 0, name)
            self.assertGreater(rejected[name], 0, name)
        self.assertEqual(mismatches[:10], [], f"{len(mismatches)} mismatches")


# comp_tdm_ra is validated with the plain gfx1250 GEMM kernel and warp_k=1 only.
# Dispatcher GEMM variant -> (Tile Engine kernel_name_prefix, ArchFilter operator).
# The Tile Engine Stream-K builder has its own validation module; "gemm_streamk"
# exercises the shared per-op rule that keeps comp_tdm_ra on gemm_universal.
# Batched GEMM has no ArchFilter operator.
RA_VARIANTS = {
    "standard": ("gemm_universal", OperatorType.GEMM),
    "batched": ("batched_gemm", None),
    "grouped": ("grouped_gemm", OperatorType.GEMM_GROUPED),
    "multi_d": ("gemm_multi_d", OperatorType.GEMM_MULTI_D),
    "preshuffle": ("gemm_preshuffle", OperatorType.GEMM_PRESHUFFLE),
    "stream_k": ("gemm_streamk", OperatorType.GEMM_STREAMK),
}
RA_PIPELINES = ("comp_tdm_ra", "comp_tdm_ra_s3r2", "comp_tdm_ra_s2r1")
# is_tile_config_valid's positional tile arguments.
TILE_KEYS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
)
RA_TILE = _tile(256, 256, 64, 2, 2, 1, 16, 16, 32)
RA_TILE_WARP_K2 = _tile(256, 256, 64, 1, 2, 2, 16, 16, 32)


@unittest.skipIf(te is None, "tile_engine/ops/gemm not importable")
class TestCompTdmRaRejectedOutsidePlainGfx1250(unittest.TestCase):
    """Every layer that can produce or select a comp_tdm_ra kernel (Tile Engine
    trait + tile rules, ArchFilter, the gemm_utils sweep gate and ctypes_utils
    validate_kernel_config) accepts it for plain gfx1250 GEMM with warp_k=1 and
    rejects it on other archs, other GEMM variants and warp_k=2."""

    def _layers(self, arch, variant, name, t, dtype="fp16"):
        """{layer: accepts} for one config; layers that do not model it are absent."""
        prefix, operator = RA_VARIANTS[variant]
        base, slots, param = split_pipeline_ring(name)
        pads = (False, False, False)
        out_dtype = CommonTypeMappings.get_output_dtype(dtype)
        got = {
            "tile_engine": te.is_trait_combination_valid(
                name,
                "tdm",
                "intrawave",
                False,
                kernel_name_prefix=prefix,
                layout="rcr",
                pad_m=False,
                pad_n=False,
                pad_k=False,
                gpu_target=arch,
            )
            and te.is_tile_config_valid(
                *(t[k] for k in TILE_KEYS),
                dtype,
                dtype,
                out_dtype,
                name,
                "rcr",
                arch,
                kernel_name_prefix=prefix,
            )
        }
        if operator is not None:
            got["arch_filter"] = ArchFilter(arch).is_kernel_valid(
                datatype_a=dtype,
                datatype_b=dtype,
                datatype_c=out_dtype,
                pipeline=base,
                epilogue="tdm",
                scheduler="intrawave",
                layout="rcr",
                pad_m=False,
                pad_n=False,
                pad_k=False,
                num_slots=slots,
                ring_param=param,
                operator=operator,
                **t,
            )
        if variant == "preshuffle":
            # The preshuffle sweep gates its pipelines through this rule.
            got["preshuffle_gate"] = not preshuffle_pipeline_reject_reason(
                base, arch, "rcr", *pads, warp_k=t["warp_k"], scheduler="intrawave"
            )
        elif gemm_utils is not None:
            got["gemm_utils"] = gemm_utils._gfx1250_pipeline_supported(
                base,
                "intrawave",
                "tdm",
                False,
                t["warp_m"],
                t["warp_n"],
                t["warp_k"],
                arch,
                variant,
                layout="rcr",
                dtype=dtype,
                warp_tile_k=t["warp_tile_k"],
                warp_tile_m=t["warp_tile_m"],
                warp_tile_n=t["warp_tile_n"],
                tile_mn=(t["tile_m"], t["tile_n"]),
                ring=(slots, param),
            )
        if ctypes_utils is not None:
            config = ctypes_utils.KernelConfig(
                dtype_a=dtype,
                dtype_b=dtype,
                dtype_c=out_dtype,
                tile_m=t["tile_m"],
                tile_n=t["tile_n"],
                tile_k=t["tile_k"],
                wave_m=t["warp_m"],
                wave_n=t["warp_n"],
                wave_k=t["warp_k"],
                warp_m=t["warp_tile_m"],
                warp_n=t["warp_tile_n"],
                warp_k=t["warp_tile_k"],
                pipeline=name,
                scheduler="intrawave",
                epilogue="tdm",
                pad_m=False,
                pad_n=False,
                pad_k=False,
                gfx_arch=arch,
                variant=variant,
            )
            got["ctypes_utils"] = ctypes_utils.validate_kernel_config(config).is_valid
        return got

    def test_plain_gfx1250_accepted(self):
        """Control: the rejections below are not a tile every layer refuses."""
        for name in RA_PIPELINES:
            got = self._layers(ARCH, "standard", name, RA_TILE)
            self.assertTrue(all(got.values()), (name, got))

    def test_other_arch_rejected(self):
        for arch, variant, name in itertools.product(
            ("gfx950", "gfx942"), RA_VARIANTS, RA_PIPELINES
        ):
            got = self._layers(arch, variant, name, RA_TILE)
            self.assertFalse(any(got.values()), (arch, variant, name, got))

    def test_other_variants_rejected(self):
        for variant, name in itertools.product(RA_VARIANTS, RA_PIPELINES):
            if variant == "standard":
                continue
            got = self._layers(ARCH, variant, name, RA_TILE)
            self.assertFalse(any(got.values()), (variant, name, got))

    def test_warp_k2_rejected(self):
        # comp_tdm takes the same 1x2x2 tile, so warp_k is the only reason.
        control = self._layers(ARCH, "standard", "comp_tdm", RA_TILE_WARP_K2)
        self.assertTrue(all(control.values()), control)
        for variant, name in itertools.product(RA_VARIANTS, RA_PIPELINES):
            got = self._layers(ARCH, variant, name, RA_TILE_WARP_K2)
            self.assertFalse(any(got.values()), (variant, name, got))


if __name__ == "__main__":
    unittest.main()
