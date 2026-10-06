#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Arch-string handling in tile_engine's gemm_validation_utils.

This module is shared by seven GEMM instance builders, and the dispatcher
bridges gate on it too (dispatcher/python/gemm_utils.py imports it directly and
mirrors its warp gate to keep bridge and Old-TE instance sets identical). A
change to how it reads an architecture string is therefore never local, and the
targets that actually reach it are not always bare: CK's own CMakeLists
configures GPU_TARGETS as "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
for ASAN builds, and that string is what gets handed to --gpu_target.

These tests pin the two decisions taken while enabling gfx1250:

  1. Fragment-shape tests ARE normalized. Which MFMA/WMMA fragment the silicon
     has does not depend on xnack or sramecc.
  2. The WARP_SUPPORTED_COMBINATIONS lookup is NOT normalized. See
     TestWarpConfigurationLookupIsUnchangedOnGfx9 for the evidence.

Run: python3 -m pytest tests/test_gemm_validation_arch_normalization.py -v
"""

import importlib.util
import itertools
import logging
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CK_ROOT = DISPATCHER_DIR.parent

_TE_VALIDATION = CK_ROOT / "tile_engine" / "ops" / "gemm" / "gemm_validation_utils.py"


def _load_validation_module():
    spec = importlib.util.spec_from_file_location(
        "_te_gemm_validation_utils_arch_test", _TE_VALIDATION
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if not _TE_VALIDATION.exists():  # pragma: no cover - tile_engine not checked out
    gvu = None
else:
    gvu = _load_validation_module()


_SUFFIXES = {
    "gfx90a": "gfx90a:sramecc+:xnack-",
    "gfx942": "gfx942:sramecc+:xnack-",
    "gfx950": "gfx950:sramecc+:xnack-",
    "gfx1201": "gfx1201:xnack-",
    "gfx1250": "gfx1250:xnack-",
}


@unittest.skipIf(gvu is None, "tile_engine tree not present")
class TestFragmentShapeChecksAreNormalized(unittest.TestCase):
    """A feature suffix must not change which MFMA/WMMA fragment we believe in.

    _validate_fp8_mfma_warp_tile_k encodes ISA fragment shapes. gfx950 doubles
    the fp8 K-block relative to gfx90a/gfx942 (MFMA_F32_16x16x256_F8 and
    MFMA_F32_32x32x128_F8). Before normalization, "gfx950:sramecc+:xnack-" fell
    through the `== "gfx950"` test into the gfx942 branch, so on an ASAN-style
    configure the validator rejected gfx950's correct warp_tile_k and accepted
    gfx942's wrong one. Both directions of that are pinned here.
    """

    def test_suffixed_target_matches_bare_target_for_every_arch(self):
        for bare, suffixed in _SUFFIXES.items():
            for dtype, wt_m, wt_k in itertools.product(
                ("fp8", "bf8", "fp16", "bf16"), (16, 32, 64), (16, 32, 64, 128, 256)
            ):
                with self.subTest(arch=bare, dtype=dtype, m=wt_m, k=wt_k):
                    self.assertEqual(
                        gvu._validate_fp8_mfma_warp_tile_k(
                            wt_m, wt_m, wt_k, dtype, bare
                        )[0],
                        gvu._validate_fp8_mfma_warp_tile_k(
                            wt_m, wt_m, wt_k, dtype, suffixed
                        )[0],
                    )

    def test_suffixed_gfx950_uses_the_gfx950_k_block_not_the_gfx942_one(self):
        for dtype in ("fp8", "bf8"):
            for wt_m, good_k, gfx942_k in ((16, 128, 64), (32, 64, 32)):
                with self.subTest(dtype=dtype, warp_tile_m=wt_m):
                    ok, _ = gvu._validate_fp8_mfma_warp_tile_k(
                        wt_m, wt_m, good_k, dtype, _SUFFIXES["gfx950"]
                    )
                    self.assertTrue(ok, "gfx950 K-block must be accepted")
                    bad, _ = gvu._validate_fp8_mfma_warp_tile_k(
                        wt_m, wt_m, gfx942_k, dtype, _SUFFIXES["gfx950"]
                    )
                    self.assertFalse(bad, "gfx942 K-block must not be accepted")

    def test_error_message_echoes_the_caller_supplied_target(self):
        # Normalization is internal; diagnostics should show what was passed in
        # so a mis-typed --gpu_target is visible in the message.
        _, msg = gvu._validate_fp8_mfma_warp_tile_k(
            32, 32, 999, "fp8", _SUFFIXES["gfx950"]
        )
        self.assertIn(_SUFFIXES["gfx950"], msg)

    def test_gfx1250_rejects_the_32x32_warp_tile_bare_and_suffixed(self):
        # There is no 32x32 WMMA fragment on gfx1250; a 32x32 warp tile compiles
        # and returns garbage, so this must be rejected however the target is
        # spelled.
        for target in ("gfx1250", _SUFFIXES["gfx1250"]):
            for dtype in ("fp8", "bf8"):
                with self.subTest(target=target, dtype=dtype):
                    ok, _ = gvu._validate_fp8_mfma_warp_tile_k(
                        32, 32, 32, dtype, target
                    )
                    self.assertFalse(ok)


@unittest.skipIf(gvu is None, "tile_engine tree not present")
class TestWarpConfigurationLookupIsUnchangedOnGfx9(unittest.TestCase):
    """The WARP_SUPPORTED_COMBINATIONS lookup stays raw for gfx9. On purpose.

    Normalizing it would activate a validator that a suffixed target currently
    bypasses -- which sounds like a fix until you look at what the table
    contains. dispatcher/codegen/arch_specs.json, the declared single JSON
    source of truth for arch data, lists seven warp maps for gfx942 and nine for
    gfx950; this table lists three for each. Turning it on for suffixed names
    therefore rejects configurations the repo's own spec calls valid.

    gfx1250 is the exception, and the only one: it is absent from
    arch_specs.json entirely, so this table is its sole listing, and it is the
    target this work exists for.

    These tests fail if someone "tidies up" the asymmetry.
    """

    _GRID = [1, 2, 4, 8]

    def _decisions(self, target):
        return {
            (m, n, k): gvu.validate_warp_configuration(m, n, k, target)
            for m, n, k in itertools.product(self._GRID, repeat=3)
        }

    def test_suffixed_gfx9_targets_stay_permissive(self):
        # develop's behaviour: an unrecognized key logs and allows. Pinned so the
        # gfx9 instance sets cannot move under an ASAN-style configure.
        for bare in ("gfx90a", "gfx942", "gfx950"):
            suffixed = _SUFFIXES[bare]
            with self.subTest(arch=suffixed):
                self.assertTrue(
                    all(self._decisions(suffixed).values()),
                    f"{suffixed} must remain unchecked by this table",
                )

    def test_bare_gfx9_targets_still_enforce_their_three_maps(self):
        for bare in ("gfx90a", "gfx942", "gfx950"):
            decisions = self._decisions(bare)
            with self.subTest(arch=bare):
                self.assertEqual(
                    {c for c, ok in decisions.items() if ok},
                    {(1, 4, 1), (2, 2, 1), (4, 1, 1)},
                )

    def test_the_table_is_a_strict_subset_of_arch_specs_json(self):
        """Why the gfx9 lookup is left alone, asserted rather than asserted-in-prose.

        If the two ever agree, the argument above evaporates and the raw lookup
        should be revisited -- so this test is written to fail at that point.
        """
        import json

        specs_path = DISPATCHER_DIR / "codegen" / "arch_specs.json"
        if not specs_path.exists():
            self.skipTest("arch_specs.json not present")
        specs = json.load(open(specs_path))["architectures"]
        for arch in ("gfx942", "gfx950"):
            table = {tuple(c) for c in gvu.WARP_SUPPORTED_COMBINATIONS[arch]}
            spec = {tuple(c) for c in specs[arch]["warp_configs"]}
            with self.subTest(arch=arch):
                self.assertTrue(
                    table < spec,
                    f"{arch}: gemm_validation_utils lists {sorted(table)}, "
                    f"arch_specs.json lists {sorted(spec)}. If these now agree, "
                    f"normalizing the gfx9 lookup is no longer a coverage loss.",
                )

    def test_suffixed_gfx1250_is_checked_like_bare_gfx1250(self):
        self.assertEqual(
            self._decisions(_SUFFIXES["gfx1250"]), self._decisions("gfx1250")
        )

    def test_suffixed_gfx1250_rejects_maps_absent_from_the_table(self):
        # develop allowed all of these on "gfx1250:xnack-" because the lookup
        # missed. 1x1x1 is the example the review asked about.
        for combo in ((1, 1, 1), (1, 1, 2), (8, 8, 8), (4, 4, 1)):
            with self.subTest(combo=combo):
                self.assertFalse(
                    gvu.validate_warp_configuration(*combo, _SUFFIXES["gfx1250"])
                )

    def test_gfx1201_suffix_is_not_swept_up_by_the_gfx1250_exception(self):
        # The exception is exact-gfx1250, not gfx12-family. gfx1201 keeps
        # develop's permissive behaviour under a suffix.
        self.assertTrue(all(self._decisions(_SUFFIXES["gfx1201"]).values()))


@unittest.skipIf(gvu is None, "tile_engine tree not present")
class TestGroupedQuantWarpRestrictionScope(unittest.TestCase):
    """Exercise operator routing, not just the shared quant validator."""

    PREFIXES = ("gemm_rowcolquant", "grouped_gemm_rowcolquant", "grouped_gemm_tensorquant")

    @staticmethod
    def config(arch, dtype, warp):
        return dict(
            tile_m=128, tile_n=128, tile_k=256,
            warp_m=warp[0], warp_n=warp[1], warp_k=warp[2],
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            a_datatype=dtype, b_datatype=dtype, c_datatype="fp16",
            pipeline="compv3", layout="rcr", gpu_target=arch,
        )

    def test_grouped_limits_do_not_filter_plain_rowcolquant(self):
        for arch, dtype, prefix, warp in itertools.product(
            ("gfx1250", "gfx1250:xnack-"), ("fp8", "bf8"), self.PREFIXES,
            ((4, 2, 1), (1, 2, 2), (1, 4, 1)),
        ):
            with self.subTest(arch=arch, dtype=dtype, prefix=prefix, warp=warp):
                # Eight warps and warp_k=2 are grouped-bridge restrictions.
                # Four warps with warp_k=1 remain valid for all three operators.
                expected = prefix == "gemm_rowcolquant" or warp == (1, 4, 1)
                self.assertEqual(gvu.is_tile_config_valid(
                    **self.config(arch, dtype, warp), kernel_name_prefix=prefix,
                ), expected)

    def test_shared_fragment_validation_still_applies_to_all_quant_operators(self):
        for arch, dtype, prefix in itertools.product(
            ("gfx1250", "gfx1250:xnack-"), ("fp8", "bf8"), self.PREFIXES,
        ):
            with self.subTest(arch=arch, dtype=dtype, prefix=prefix):
                cfg = self.config(arch, dtype, (1, 4, 1))
                cfg.update(warp_tile_m=32, warp_tile_n=32)
                self.assertFalse(gvu.is_tile_config_valid(**cfg, kernel_name_prefix=prefix))


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    sys.exit(unittest.main())
