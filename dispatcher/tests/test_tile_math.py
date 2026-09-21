#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for codegen/tile_math.py.

The reference lookup tables TILE_TO_WAVE_WARP and _TILE_WTILK_TO_VECS are 
used as ground-truth oracles. The mathematical functions in tile_math.py 
must generate at least the set of configurations present in those tables 
(no false negatives).

Run:
    python3 -m pytest dispatcher/tests/test_tile_math.py -v
or:
    cd projects/composablekernel/dispatcher
    python3 -m pytest tests/test_tile_math.py -v
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from arch_specs_generated import WARP_SUPPORTED_COMBINATIONS, ELEMENT_SIZE_MAP  # noqa: E402
from grouped_conv.tile_math import (                                              # noqa: E402
    get_valid_wave_warp_pairs,
    get_valid_vec_sizes,
    get_vec_sizes_for_wave_warp,
    dtype_keys_for_warp_tile_k,
    WARP_SIZE,
)

# =============================================================================
# Reference oracle tables (ground truth extracted from CK profiler configs)
#
# They live here as static test fixtures: the math functions in tile_math.py must generate
# at least every entry present in these tables (zero false negatives).
# =============================================================================

TILE_TO_WAVE_WARP: Dict[Tuple[int, int, int], List[Tuple[Tuple, Tuple]]] = {
    (16, 16, 32):  [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 16, 64):  [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 16, 128): [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 32, 64):  [((1, 2, 1), (16, 16, 8)),  ((1, 2, 1), (16, 16, 16))],
    (16, 64, 32):  [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 64, 64):  [((1, 2, 1), (16, 16, 8)),  ((1, 2, 1), (16, 16, 16))],
    (16, 128, 32): [((1, 1, 1), (16, 16, 16))],
    (16, 128, 64): [((1, 2, 1), (16, 16, 8)),  ((1, 2, 1), (16, 16, 16))],
    (16, 256, 32): [((1, 1, 1), (16, 16, 16))],
    (16, 256, 64): [((1, 4, 1), (16, 16, 16))],
    (32, 16, 64):  [((2, 1, 1), (16, 16, 8)),  ((2, 1, 1), (16, 16, 16))],
    (32, 32, 32):  [((1, 1, 1), (32, 32, 8))],
    (32, 64, 16):  [((1, 1, 1), (32, 32, 4))],
    (32, 64, 32):  [((1, 1, 1), (32, 32, 8))],
    (32, 64, 64):  [((1, 2, 1), (32, 32, 8))],
    (32, 128, 16): [((1, 2, 1), (32, 32, 4))],
    (32, 128, 32): [((1, 1, 1), (32, 32, 8)), ((1, 2, 1), (32, 32, 8))],
    (32, 128, 64): [((1, 2, 1), (32, 32, 8))],
    (32, 256, 64): [((1, 4, 1), (32, 32, 8))],
    (64, 16, 16):  [((1, 1, 1), (16, 16, 4)), ((1, 1, 1), (16, 16, 16)),
                    ((4, 1, 1), (16, 16, 4)), ((4, 1, 1), (16, 16, 16))],
    (64, 16, 32):  [((1, 1, 1), (16, 16, 8)), ((1, 1, 1), (16, 16, 16)),
                    ((4, 1, 1), (16, 16, 8)), ((4, 1, 1), (16, 16, 16))],
    (64, 16, 64):  [((2, 1, 1), (16, 16, 8)), ((2, 1, 1), (16, 16, 16)),
                    ((4, 1, 1), (16, 16, 16))],
    (64, 32, 16):  [((1, 1, 1), (32, 32, 4))],
    (64, 32, 32):  [((1, 1, 1), (32, 32, 8))],
    (64, 32, 64):  [((2, 1, 1), (32, 32, 8))],
    (64, 64, 8):   [((2, 1, 1), (32, 32, 8))],
    (64, 64, 16):  [((1, 1, 1), (32, 32, 4))],
    (64, 64, 32):  [((1, 1, 1), (32, 32, 8)), ((2, 2, 1), (16, 16, 8)),
                    ((2, 2, 1), (16, 16, 16))],
    (64, 64, 64):  [((2, 2, 1), (16, 16, 16)), ((2, 2, 1), (32, 32, 8))],
    (64, 128, 16): [((1, 2, 1), (32, 32, 4)), ((2, 2, 1), (32, 32, 4))],
    (64, 128, 32): [((1, 2, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (64, 128, 64): [((2, 2, 1), (32, 32, 8))],
    (128, 16, 64): [((2, 1, 1), (16, 16, 8)), ((2, 1, 1), (16, 16, 16))],
    (128, 32, 16): [((2, 1, 1), (32, 32, 4)), ((4, 1, 1), (32, 32, 4)),
                    ((4, 1, 1), (32, 32, 8))],
    (128, 32, 32): [((1, 1, 1), (32, 32, 8)), ((2, 1, 1), (32, 32, 8)),
                    ((2, 1, 2), (32, 32, 8)), ((4, 1, 1), (32, 32, 8))],
    (128, 32, 64): [((2, 1, 1), (32, 32, 8)), ((4, 1, 1), (32, 32, 16))],
    (128, 64, 8):  [((2, 1, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (128, 64, 16): [((2, 1, 1), (32, 32, 4)), ((2, 2, 1), (32, 32, 4)),
                    ((2, 2, 1), (32, 32, 8))],
    (128, 64, 32): [((2, 1, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (128, 64, 64): [((2, 2, 1), (32, 32, 8))],
    (128, 128, 16):[((1, 2, 1), (32, 32, 4)), ((2, 2, 1), (32, 32, 4))],
    (128, 128, 32):[((1, 2, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (128, 128, 64):[((2, 2, 1), (32, 32, 8))],
    (128, 192, 16):[((2, 2, 1), (32, 32, 4))],
    (128, 256, 16):[((2, 2, 1), (32, 32, 4))],
    (128, 256, 32):[((2, 2, 1), (32, 32, 8))],
    (224, 256, 64):[((2, 2, 1), (16, 16, 16))],
    (256, 16, 64): [((4, 1, 1), (16, 16, 16))],
    (256, 32, 64): [((4, 1, 1), (32, 32, 8))],
    (256, 64, 8):  [((2, 2, 1), (32, 32, 8))],
    (256, 128, 16):[((2, 2, 1), (32, 32, 4))],
    (256, 128, 32):[((2, 2, 1), (32, 32, 8))],
    (256, 224, 64):[((2, 2, 1), (16, 16, 16))],
    (256, 256, 32):[((2, 2, 1), (16, 16, 16)), ((2, 2, 1), (32, 32, 8))],
}

_TILE_WTILK_TO_VECS: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int]]] = {
    (16, 16, 32, 8):   [(1, 1, 2)],
    (16, 16, 32, 16):  [(1, 1, 1), (1, 1, 2)],
    (16, 16, 64, 8):   [(4, 4, 4)],
    (16, 16, 64, 16):  [(1, 1, 1), (1, 4, 4), (4, 1, 1), (4, 4, 4), (8, 8, 4)],
    (16, 16, 128, 8):  [(4, 4, 4)],
    (16, 16, 128, 16): [(8, 8, 4)],
    (16, 32, 64, 8):   [(4, 4, 4)],
    (16, 32, 64, 16):  [(1, 1, 1), (1, 2, 4), (1, 4, 4), (2, 1, 1), (2, 2, 4), (2, 4, 4), (8, 8, 4)],
    (16, 64, 32, 8):   [(1, 4, 4), (4, 1, 1), (4, 4, 4)],
    (16, 64, 32, 16):  [(1, 8, 4), (8, 1, 1), (8, 8, 4)],
    (16, 64, 64, 8):   [(4, 4, 4)],
    (16, 64, 64, 16):  [(1, 1, 1), (1, 8, 4), (2, 1, 1), (2, 8, 4), (8, 8, 4)],
    (16, 128, 32, 16): [(4, 4, 1)],
    (16, 128, 64, 8):  [(4, 4, 4)],
    (16, 128, 64, 16): [(1, 8, 4), (2, 8, 4), (8, 8, 4)],
    (16, 256, 32, 16): [(8, 8, 1)],
    (16, 256, 64, 16): [(1, 8, 4), (2, 8, 4), (8, 8, 4)],
    (32, 16, 64, 8):   [(4, 4, 2)],
    (32, 16, 64, 16):  [(1, 1, 1), (1, 2, 2), (2, 1, 1), (2, 2, 2), (4, 1, 1), (4, 2, 2), (8, 8, 2)],
    (32, 32, 32, 8):   [(2, 2, 1), (2, 2, 2)],
    (32, 64, 16, 4):   [(1, 4, 4), (4, 4, 4)],
    (32, 64, 32, 8):   [(1, 1, 8), (2, 2, 1), (2, 8, 8), (4, 4, 1), (4, 4, 2), (4, 4, 4), (8, 8, 8)],
    (32, 64, 64, 8):   [(4, 4, 8), (8, 8, 8)],
    (32, 128, 16, 4):  [(4, 4, 4)],
    (32, 128, 32, 8):  [(1, 1, 8), (4, 4, 4), (8, 8, 1), (8, 8, 2), (8, 8, 8)],
    (32, 128, 64, 8):  [(4, 4, 8), (8, 8, 8)],
    (32, 256, 64, 8):  [(8, 8, 8)],
    (64, 16, 16, 4):   [(4, 1, 1)],
    (64, 16, 16, 16):  [(4, 1, 1)],
    (64, 16, 32, 8):   [(1, 4, 4), (4, 1, 1), (4, 4, 4), (8, 1, 1), (8, 2, 2)],
    (64, 16, 32, 16):  [(1, 8, 4), (8, 1, 1), (8, 2, 2), (8, 8, 4)],
    (64, 16, 64, 8):   [(4, 4, 2)],
    (64, 16, 64, 16):  [(1, 1, 1), (1, 2, 2), (8, 1, 1), (8, 2, 2), (8, 8, 2), (16, 1, 1), (16, 2, 2)],
    (64, 32, 16, 4):   [(4, 4, 1), (4, 4, 4)],
    (64, 32, 32, 8):   [(1, 1, 8), (4, 4, 1), (4, 4, 2), (4, 4, 4), (8, 8, 1), (8, 8, 8)],
    (64, 32, 64, 8):   [(4, 4, 4), (8, 8, 4)],
    (64, 64, 8, 8):    [(1, 1, 8)],
    (64, 64, 16, 4):   [(1, 1, 1), (4, 4, 4)],
    (64, 64, 32, 8):   [(1, 1, 1), (1, 1, 8), (1, 2, 1), (2, 1, 2), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
    (64, 64, 32, 16):  [(1, 2, 1), (2, 1, 2), (4, 4, 4), (8, 8, 8)],
    (64, 64, 64, 8):   [(1, 1, 1), (1, 4, 4), (2, 2, 2), (4, 1, 1), (4, 4, 4), (8, 8, 4), (8, 8, 8)],
    (64, 64, 64, 16):  [(2, 2, 4), (4, 1, 1), (8, 8, 2), (8, 8, 8)],
    (64, 128, 16, 4):  [(4, 4, 4)],
    (64, 128, 32, 8):  [(1, 1, 8), (1, 4, 4), (1, 8, 8), (4, 4, 4), (8, 8, 8)],
    (64, 128, 64, 8):  [(8, 8, 8)],
    (128, 16, 64, 8):  [(4, 4, 2)],
    (128, 16, 64, 16): [(8, 1, 1), (8, 2, 2), (8, 8, 2)],
    (128, 32, 16, 4):  [(4, 1, 1), (4, 2, 2), (4, 4, 4)],
    (128, 32, 16, 8):  [(4, 1, 1), (4, 2, 2)],
    (128, 32, 32, 8):  [(1, 1, 8), (4, 1, 1), (4, 4, 4), (8, 1, 1), (8, 2, 2), (8, 8, 1), (8, 8, 2), (8, 8, 8)],
    (128, 32, 64, 8):  [(4, 4, 4), (8, 8, 4)],
    (128, 32, 64, 16): [(16, 1, 1), (16, 2, 2), (16, 8, 8)],
    (128, 64, 8, 8):   [(1, 1, 8)],
    (128, 64, 16, 4):  [(4, 4, 4)],
    (128, 64, 16, 8):  [(1, 1, 8)],
    (128, 64, 32, 8):  [(1, 1, 8), (4, 4, 4), (8, 8, 8)],
    (128, 64, 64, 8):  [(8, 8, 8)],
    (128, 128, 16, 4): [(1, 1, 4), (4, 4, 4)],
    (128, 128, 32, 8): [(1, 1, 8), (4, 4, 4), (4, 4, 8), (8, 8, 8)],
    (128, 128, 64, 8): [(4, 4, 4), (4, 4, 8), (8, 8, 4), (8, 8, 8)],
    (128, 192, 16, 4): [(4, 4, 4)],
    (128, 256, 16, 4): [(4, 4, 4)],
    (128, 256, 32, 8): [(1, 1, 8), (4, 4, 4), (8, 4, 8), (8, 8, 8)],
    (224, 256, 64, 16):[(8, 8, 8)],
    (256, 16, 64, 16): [(8, 1, 1), (8, 2, 2), (8, 8, 2)],
    (256, 32, 64, 8):  [(8, 8, 4), (8, 8, 8)],
    (256, 64, 8, 8):   [(1, 1, 8)],
    (256, 128, 16, 4): [(4, 4, 4)],
    (256, 128, 32, 8): [(1, 1, 8), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
    (256, 224, 64, 16):[(8, 8, 8)],
    (256, 256, 32, 8): [(4, 4, 4), (8, 8, 4), (8, 8, 8)],
    (256, 256, 32, 16):[(8, 8, 8)],
}


# =============================================================================
# TestGetValidWaveWarpPairs
# =============================================================================

class TestGetValidWaveWarpPairs(unittest.TestCase):
    """Tests for get_valid_wave_warp_pairs()."""

    # --- Spot checks ---

    def test_spot_check_128_64_32(self):
        """Reference pairs for (128,64,32) must all be in the math result."""
        tile = (128, 64, 32)
        ref_pairs = TILE_TO_WAVE_WARP[tile]
        math_pairs = set(get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32"))
        for pair in ref_pairs:
            self.assertIn(
                pair, math_pairs,
                f"Missing reference pair {pair} for tile {tile}",
            )

    def test_spot_check_64_64_32(self):
        """Reference pairs for (64,64,32) must all be in the union of dtype results."""
        tile = (64, 64, 32)
        ref_pairs = TILE_TO_WAVE_WARP[tile]
        # Union across dtype_keys — reference table is dtype-agnostic
        math_pairs = set(get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32"))
        math_pairs |= set(get_valid_wave_warp_pairs(*tile, "fp32_fp32_fp32"))
        math_pairs |= set(get_valid_wave_warp_pairs(*tile, "fp16_fp16_fp32"))
        for pair in ref_pairs:
            self.assertIn(
                pair, math_pairs,
                f"Missing reference pair {pair} for tile {tile}",
            )

    def test_spot_check_128_32_32_fp32(self):
        """Reference pairs for (128,32,32) include fp32 warp tiles."""
        tile = (128, 32, 32)
        ref_pairs = TILE_TO_WAVE_WARP[tile]
        # fp32 uses warp_tile_k in {4, 8}; bf16 uses {8, 16}
        # Try both dtype_keys and require the union covers all reference pairs
        math_pairs = set(get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32"))
        math_pairs |= set(get_valid_wave_warp_pairs(*tile, "fp32_fp32_fp32"))
        for pair in ref_pairs:
            self.assertIn(
                pair, math_pairs,
                f"Missing reference pair {pair} for tile {tile}",
            )

    # --- Full coverage of reference table ---

    def test_coverage_all_reference_tiles(self):
        """Every (tile, pair) in TILE_TO_WAVE_WARP must be generated by the math.

        The math is called with both bf16_bf16_fp32 and fp32_fp32_fp32 dtype_keys
        and the union must contain every reference pair.  This is the zero-false-
        negatives requirement.
        """
        DTYPE_KEYS = ["bf16_bf16_fp32", "fp32_fp32_fp32", "fp16_fp16_fp32"]
        missing = []

        for tile, ref_pairs in TILE_TO_WAVE_WARP.items():
            # Union across dtype_keys (reference table is dtype-agnostic)
            math_pairs: set = set()
            for dk in DTYPE_KEYS:
                math_pairs |= set(get_valid_wave_warp_pairs(*tile, dk))

            for pair in ref_pairs:
                if pair not in math_pairs:
                    missing.append((tile, pair))

        if missing:
            lines = [f"  tile={t}  pair={p}" for t, p in missing]
            self.fail(
                f"{len(missing)} reference pairs not generated by math:\n"
                + "\n".join(lines)
            )

    # --- Structural validity of math output ---

    def test_no_invalid_wave_combos(self):
        """Every wave combo returned must be in WARP_SUPPORTED_COMBINATIONS for gfx942."""
        valid_waves = {tuple(c) for c in WARP_SUPPORTED_COMBINATIONS["gfx942"]}
        tile = (128, 64, 32)
        for wave, _warp in get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32"):
            self.assertIn(
                wave, valid_waves,
                f"Wave combo {wave} not in gfx942 supported combinations",
            )

    def test_no_invalid_wave_combos_all_tiles(self):
        """All tiles in reference table: every returned wave must be arch-valid."""
        valid_waves = {tuple(c) for c in WARP_SUPPORTED_COMBINATIONS["gfx942"]}
        for tile in TILE_TO_WAVE_WARP:
            for wave, _warp in get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32"):
                self.assertIn(
                    wave, valid_waves,
                    f"tile={tile}: wave {wave} not in gfx942 supported combinations",
                )

    def test_tile_divisibility_enforced(self):
        """A tile not divisible by any warp_tile_m must return no pairs."""
        # (13, 17, 32) — prime dimensions, won't divide any MFMA shape
        pairs = get_valid_wave_warp_pairs(13, 17, 32, "bf16_bf16_fp32")
        self.assertEqual(pairs, [], "Expected no pairs for prime-dimension tile")

    def test_warp_tile_divides_tile_m_and_n(self):
        """For every returned pair, warp_tile_m | tile_m and warp_tile_n | tile_n."""
        for tile in TILE_TO_WAVE_WARP:
            tm, tn, tk = tile
            for _wave, (wt_m, wt_n, wt_k) in get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32"):
                self.assertEqual(tm % wt_m, 0, f"tile_m={tm} not divisible by warp_m={wt_m}")
                self.assertEqual(tn % wt_n, 0, f"tile_n={tn} not divisible by warp_n={wt_n}")

    # --- Special cases ---

    def test_wave_k2_special_case(self):
        """(128,32,32) must produce the ((2,1,2), (32,32,8)) pair (wave_k=2)."""
        tile = (128, 32, 32)
        math_pairs = set(get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32"))
        expected = ((2, 1, 2), (32, 32, 8))
        self.assertIn(
            expected, math_pairs,
            f"Expected wave_k=2 pair {expected} not found for tile {tile}",
        )

    def test_pipeline_async_constraint(self):
        """pipeline='basic_async_v1' must only return pairs with wave_n==2 and warp_tile_n==16."""
        tile = (64, 64, 32)
        pairs = get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32", pipeline="basic_async_v1")
        for wave, (wt_m, wt_n, wt_k) in pairs:
            _wm, wn, _wk = wave
            self.assertEqual(wn, 2, f"async pipeline: expected wave_n=2, got {wn}")
            self.assertEqual(wt_n, 16, f"async pipeline: expected warp_tile_n=16, got {wt_n}")

    def test_no_duplicates(self):
        """The returned list must not contain duplicate pairs."""
        for tile in list(TILE_TO_WAVE_WARP.keys()):
            pairs = get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32")
            self.assertEqual(
                len(pairs), len(set(pairs)),
                f"tile={tile}: duplicate pairs in result",
            )

    def test_gfx950_returns_superset_of_gfx942(self):
        """gfx950 supports more wave combos, so it must return >= as many pairs as gfx942."""
        tile = (128, 64, 32)
        pairs_942 = set(get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32", arch="gfx942"))
        pairs_950 = set(get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32", arch="gfx950"))
        self.assertTrue(
            pairs_942.issubset(pairs_950),
            "gfx942 pairs should be a subset of gfx950 pairs",
        )


# =============================================================================
# TestGetValidVecSizes
# =============================================================================

class TestGetValidVecSizes(unittest.TestCase):
    """Tests for get_valid_vec_sizes()."""

    # --- Spot checks ---

    def test_spot_check_128_64_32_wave221_wt_32328(self):
        """Reference vecs for (128,64,32) wave=(2,2,1) warp=(32,32,8) bf16."""
        vecs = get_valid_vec_sizes(128, 64, 32, 2, 2, 1, 32, 32, 8, "bf16_bf16_fp32")
        # Reference: [(1,1,8), (4,4,4), (8,8,8)] from _TILE_WTILK_TO_VECS[(128,64,32,8)]
        ref = _TILE_WTILK_TO_VECS[(128, 64, 32, 8)]
        math_set = set(vecs)
        for v in ref:
            self.assertIn(v, math_set, f"Reference vec {v} missing for (128,64,32) wave=(2,2,1)")

    def test_spot_check_64_64_32_wave111_wt_32328(self):
        """Reference vecs for (64,64,32) wave=(1,1,1) warp=(32,32,8) bf16."""
        vecs = get_valid_vec_sizes(64, 64, 32, 1, 1, 1, 32, 32, 8, "bf16_bf16_fp32")
        ref = _TILE_WTILK_TO_VECS[(64, 64, 32, 8)]
        math_set = set(vecs)
        for v in ref:
            self.assertIn(v, math_set, f"Reference vec {v} missing for (64,64,32) wave=(1,1,1)")

    # --- Structural validity ---

    def test_pixel_budget_respected(self):
        """vec_a/vec_b must be compatible with their pixel budget.

        Compatible means the vector divides the per-thread pixel budget OR is an
        exact multiple of it (the wide load decomposes into v/pixels sub-loads).
        """
        configs = [
            (128, 64, 32, 2, 2, 1, 32, 32, 8, "bf16_bf16_fp32"),
            (64, 64, 32, 1, 1, 1, 32, 32, 8, "bf16_bf16_fp32"),
            (256, 128, 32, 2, 2, 1, 32, 32, 8, "bf16_bf16_fp32"),
        ]
        for cfg in configs:
            tm, tn, tk, wm, wn, wk, wt_m, wt_n, wt_k, dk = cfg
            block_size = WARP_SIZE * wm * wn * wk
            pixels_a = tm * tk // block_size
            pixels_b = tn * tk // block_size
            for va, vb, vc in get_valid_vec_sizes(*cfg):
                self.assertTrue(pixels_a % va == 0 or va % pixels_a == 0,
                    f"cfg={cfg}: pixels_a={pixels_a} incompatible with vec_a={va}")
                self.assertTrue(pixels_b % vb == 0 or vb % pixels_b == 0,
                    f"cfg={cfg}: pixels_b={pixels_b} incompatible with vec_b={vb}")

    def test_fp32_vec_c_8_admitted(self):
        """fp32 tiles admit vec_c=8 (relaxed 32-byte ceiling). Category (A)."""
        vecs = get_valid_vec_sizes(128, 128, 32, 2, 2, 1, 32, 32, 8, "fp32_fp32_fp32")
        self.assertIn((4, 4, 8), set(vecs),
            "fp32 (128,128,32) should admit vec triple (4,4,8)")

    def test_bf16_asymmetric_small_pixel_vec8(self):
        """Asymmetric small-pixel tile admits a vec=8 triple. Category (B)."""
        # (16,256,64) wave=(1,4,1): pixels_a = 16*64/256 = 4 < 8, but 8 % 4 == 0.
        vecs = get_valid_vec_sizes(16, 256, 64, 1, 4, 1, 16, 16, 16, "bf16_bf16_fp32")
        self.assertTrue(any(8 in (va, vb) for va, vb, vc in vecs),
            "bf16 (16,256,64) wave=(1,4,1) should admit a vec=8 triple")

    def test_bf16_vec_c_capped_at_16(self):
        """bf16 vec_c never exceeds 16 (LDS 256-bit ceiling still enforced)."""
        for va, vb, vc in get_valid_vec_sizes(128, 128, 32, 2, 2, 1, 32, 32, 8, "bf16_bf16_fp32"):
            self.assertLessEqual(vc, 16, f"bf16: vec_c={vc} > 16")

    def test_lds_validity_bf16(self):
        """All returned vecs must satisfy the power-of-2 bit-width constraint for bf16.

        Standard LDS allows 8–128 bits; some bwd_data global loads allow 256 bits.
        The constraint is: bits = vec * sizeof * 8 must be a power of 2 up to 256.
        """
        sizeof_bf16 = 2.0
        for va, vb, vc in get_valid_vec_sizes(128, 64, 32, 2, 2, 1, 32, 32, 8, "bf16_bf16_fp32"):
            for v in (va, vb, vc):
                bits = int(v * sizeof_bf16 * 8)
                self.assertGreater(bits, 0, f"vec={v}: bits must be > 0")
                self.assertLessEqual(bits, 256, f"vec={v}: bits={bits} > 256")
                self.assertEqual(bits & (bits - 1), 0, f"vec={v}: bits={bits} not power of 2")

    def test_dtype_max_vec_fp32(self):
        """For fp32, vec_a and vec_b must be <= 8 (32 bytes / 4 bytes, allowing 2×16-byte loads)."""
        for va, vb, vc in get_valid_vec_sizes(128, 64, 32, 2, 2, 1, 32, 32, 4, "fp32_fp32_fp32"):
            self.assertLessEqual(va, 8, f"fp32: vec_a={va} > 8")
            self.assertLessEqual(vb, 8, f"fp32: vec_b={vb} > 8")

    def test_dtype_max_vec_bf16(self):
        """For bf16, vec_a and vec_b must be <= 16 (32 bytes / 2 bytes per element)."""
        for va, vb, vc in get_valid_vec_sizes(128, 64, 32, 2, 2, 1, 32, 32, 8, "bf16_bf16_fp32"):
            self.assertLessEqual(va, 16, f"bf16: vec_a={va} > 16")
            self.assertLessEqual(vb, 16, f"bf16: vec_b={vb} > 16")

    def test_vec_c_divisibility(self):
        """tile_n must be divisible by vec_c (output is vectorized along N)."""
        tile_n = 64
        wave_n = 2
        warp_tile_n = 16
        for va, vb, vc in get_valid_vec_sizes(
            128, tile_n, 32, 2, wave_n, 1, 16, warp_tile_n, 16, "bf16_bf16_fp32"
        ):
            self.assertEqual(
                tile_n % vc, 0,
                f"tile_n={tile_n} not divisible by vec_c={vc}",
            )

    def test_empty_result_for_non_divisible_block(self):
        """If tile_m * tile_k is not divisible by block_size, return []."""
        # tile_m=13 (prime) → pixels_a won't be integer → expect empty
        result = get_valid_vec_sizes(13, 16, 32, 1, 1, 1, 16, 16, 16, "bf16_bf16_fp32")
        self.assertEqual(result, [])


# =============================================================================
# TestGetVecSizesForWaveWarp (wrapper)
# =============================================================================

class TestGetVecSizesForWaveWarp(unittest.TestCase):
    """Tests for the get_vec_sizes_for_wave_warp() convenience wrapper."""

    def test_spot_check_128_64_32_wt8(self):
        """Reference vecs for (128,64,32,8) in _TILE_WTILK_TO_VECS must all appear."""
        ref = _TILE_WTILK_TO_VECS[(128, 64, 32, 8)]
        math_set = set(get_vec_sizes_for_wave_warp(128, 64, 32, 8, "bf16_bf16_fp32"))
        for v in ref:
            self.assertIn(v, math_set,
                f"Reference vec {v} missing from get_vec_sizes_for_wave_warp(128,64,32,8)")

    def test_spot_check_64_64_32_wt8(self):
        """Reference vecs for (64,64,32,8) in _TILE_WTILK_TO_VECS must all appear."""
        ref = _TILE_WTILK_TO_VECS[(64, 64, 32, 8)]
        math_set = set(get_vec_sizes_for_wave_warp(64, 64, 32, 8, "bf16_bf16_fp32"))
        for v in ref:
            self.assertIn(v, math_set,
                f"Reference vec {v} missing from get_vec_sizes_for_wave_warp(64,64,32,8)")

    def test_result_is_sorted(self):
        """Returned list must be in sorted order."""
        result = get_vec_sizes_for_wave_warp(128, 64, 32, 8, "bf16_bf16_fp32")
        self.assertEqual(result, sorted(result))

    def test_no_duplicates(self):
        """Returned list must not contain duplicates."""
        result = get_vec_sizes_for_wave_warp(128, 64, 32, 8, "bf16_bf16_fp32")
        self.assertEqual(len(result), len(set(result)))


# =============================================================================
# TestMathVsReferenceStatistics — coverage tests (the key correctness tests)
# =============================================================================

# Map warp_tile_k → dtype_keys to try when verifying _TILE_WTILK_TO_VECS
# Derived from warp_gemm_dispatcher.hpp; we try all plausible dtypes per warp_tile_k.
_DTYPE_KEYS_ALL = ["bf16_bf16_fp32", "fp32_fp32_fp32", "fp16_fp16_fp32", "fp8_fp8_fp32"]


class TestMathVsReferenceStatistics(unittest.TestCase):
    """Comprehensive coverage: math must generate >= all reference entries."""

    def test_wave_warp_no_false_negatives(self):
        """TILE_TO_WAVE_WARP: every reference pair must be in the math output.

        The math is queried with multiple dtype_keys and the union must cover
        every reference pair.  This guarantees zero false negatives.
        """
        missing = []
        for tile, ref_pairs in TILE_TO_WAVE_WARP.items():
            math_pairs: set = set()
            for dk in _DTYPE_KEYS_ALL:
                math_pairs |= set(get_valid_wave_warp_pairs(*tile, dk))
            for pair in ref_pairs:
                if pair not in math_pairs:
                    missing.append((tile, pair))

        if missing:
            lines = [f"  tile={t}  pair={p}" for t, p in missing[:20]]
            self.fail(
                f"{len(missing)} reference wave/warp pairs not generated by math "
                f"(showing first 20):\n" + "\n".join(lines)
            )

    def test_vec_no_false_negatives(self):
        """_TILE_WTILK_TO_VECS: every reference vec must be in the math output.

        For each (tile_m, tile_n, tile_k, warp_tile_k) key, the math is queried
        with all plausible dtype_keys (inferred from warp_tile_k) and the union
        must contain every reference vec triple.
        """
        missing = []
        for (tm, tn, tk, wtk), ref_vecs in _TILE_WTILK_TO_VECS.items():
            dtype_keys = dtype_keys_for_warp_tile_k(wtk)
            if not dtype_keys:
                dtype_keys = _DTYPE_KEYS_ALL  # fallback: try all

            math_vecs: set = set()
            for dk in dtype_keys:
                math_vecs |= set(get_vec_sizes_for_wave_warp(tm, tn, tk, wtk, dk))

            for v in ref_vecs:
                if v not in math_vecs:
                    missing.append(((tm, tn, tk, wtk), v))

        if missing:
            lines = [f"  key={k}  vec={v}" for k, v in missing[:30]]
            self.fail(
                f"{len(missing)} reference vec triples not generated by math "
                f"(showing first 30):\n" + "\n".join(lines)
            )

    def test_extra_wave_warp_pairs_are_structurally_valid(self):
        """Pairs the math generates but reference doesn't have must still be valid.

        Extra pairs are not failures — they represent valid configs not yet in
        the JSON profiler files.  But they must satisfy structural constraints.
        """
        valid_waves = {tuple(c) for c in WARP_SUPPORTED_COMBINATIONS["gfx942"]}
        invalid_extras = []

        for tile in TILE_TO_WAVE_WARP:
            ref_set = set(TILE_TO_WAVE_WARP[tile])
            math_pairs = get_valid_wave_warp_pairs(*tile, "bf16_bf16_fp32")
            extras = [p for p in math_pairs if p not in ref_set]
            for wave, (wt_m, wt_n, wt_k) in extras:
                tm, tn, _tk = tile
                # Wave must be arch-valid
                if wave not in valid_waves:
                    invalid_extras.append((tile, wave, "not in arch wave combos"))
                # Tile must be divisible by warp tile
                elif tm % wt_m != 0 or tn % wt_n != 0:
                    invalid_extras.append((tile, (wt_m, wt_n, wt_k), "divisibility violated"))

        if invalid_extras:
            lines = [f"  {e}" for e in invalid_extras[:10]]
            self.fail(
                f"{len(invalid_extras)} extra pairs are structurally invalid:\n"
                + "\n".join(lines)
            )

    def test_coverage_rate_wave_warp(self):
        """Log coverage statistics for TILE_TO_WAVE_WARP (informational)."""
        total = 0
        covered = 0
        for tile, ref_pairs in TILE_TO_WAVE_WARP.items():
            math_pairs: set = set()
            for dk in _DTYPE_KEYS_ALL:
                math_pairs |= set(get_valid_wave_warp_pairs(*tile, dk))
            for pair in ref_pairs:
                total += 1
                if pair in math_pairs:
                    covered += 1
        rate = covered / total * 100 if total else 0
        print(f"\n[wave/warp coverage] {covered}/{total} = {rate:.1f}%")
        self.assertEqual(covered, total, f"Coverage {rate:.1f}% < 100%")

    def test_coverage_rate_vec(self):
        """Log coverage statistics for _TILE_WTILK_TO_VECS (informational)."""
        total = 0
        covered = 0
        for (tm, tn, tk, wtk), ref_vecs in _TILE_WTILK_TO_VECS.items():
            dtype_keys = dtype_keys_for_warp_tile_k(wtk) or _DTYPE_KEYS_ALL
            math_vecs: set = set()
            for dk in dtype_keys:
                math_vecs |= set(get_vec_sizes_for_wave_warp(tm, tn, tk, wtk, dk))
            for v in ref_vecs:
                total += 1
                if v in math_vecs:
                    covered += 1
        rate = covered / total * 100 if total else 0
        print(f"\n[vec coverage] {covered}/{total} = {rate:.1f}%")
        self.assertEqual(covered, total, f"Vec coverage {rate:.1f}% < 100%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
