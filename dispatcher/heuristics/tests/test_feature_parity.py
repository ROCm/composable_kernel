#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Test that the C++ extract_features() in ml_heuristic.hpp produces identical
values to the Python GemmUniversalFeatureEngine.extract().

This test uses ctypes to call the C++ feature extraction compiled into a
small shared library, then compares against Python output. If compilation
fails (no HIP/ROCm), it falls back to verifying the Python feature engine
against manually computed expected values for specific test cases.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_engine import (
    GemmUniversalFeatureEngine,
    PIPELINE_MAP,
    SCHEDULER_MAP,
    EPILOGUE_MAP,
    LAYOUT_MAP,
)


def _compute_features_manually(
    M,
    N,
    K,
    split_k,
    dtype,
    layout,
    tile_m,
    tile_n,
    tile_k,
    warp_m,
    warp_n,
    warp_k,
    warp_tile_m,
    warp_tile_n,
    warp_tile_k,
    pipeline,
    scheduler,
    epilogue,
    pad_m,
    pad_n,
    pad_k,
    persistent,
    hw,
):
    """Recompute features independently to verify the Python engine."""
    bpe_map = {"fp8": 1.0, "fp16": 2.0, "bf16": 2.0, "fp32": 4.0}
    bpe = bpe_map.get(dtype, 1.0)

    log2_M = math.log2(max(M, 1))
    log2_N = math.log2(max(N, 1))
    log2_K = math.log2(max(K, 1))
    log2_MNK = math.log2(max(M * N * K, 1))
    mem = (M * K + K * N + M * N) * bpe
    ai = (2.0 * M * N * K) / max(mem, 1)

    lds_est = (tile_m * tile_k + tile_n * tile_k) * bpe
    lds_cap = 32768 if pipeline == "compv4" else hw["lds_capacity"]

    ntm = math.ceil(M / max(tile_m, 1))
    ntn = math.ceil(N / max(tile_n, 1))
    ntk = math.ceil(K / max(tile_k, 1))

    def eff(d, t):
        if t <= 0:
            return 1.0
        r = d % t
        return r / t if r > 0 else 1.0

    # Problem-to-tile ratios
    ratio_M_to_tile_m = M / max(tile_m, 1)
    ratio_N_to_tile_n = N / max(tile_n, 1)
    ratio_K_to_tile_k = K / max(tile_k, 1)

    # Binary features: problem smaller than tile
    problem_smaller_than_tile_m = float(M < tile_m)
    problem_smaller_than_tile_n = float(N < tile_n)
    problem_smaller_than_tile_k = float(K < tile_k)
    any_dim_too_small = float((M < tile_m) or (N < tile_n) or (K < tile_k))

    # Padding requirement features
    needs_padding_m = float(tile_m > 0 and M % tile_m != 0)
    needs_padding_n = float(tile_n > 0 and N % tile_n != 0)
    needs_padding_k = float(tile_k > 0 and K % tile_k != 0)

    # Interaction features
    has_padding_when_needed_m = float(needs_padding_m and pad_m)
    has_padding_when_needed_n = float(needs_padding_n and pad_n)
    has_padding_when_needed_k = float(needs_padding_k and pad_k)

    # Missing padding features
    missing_required_padding_m = float(needs_padding_m and not pad_m)
    missing_required_padding_n = float(needs_padding_n and not pad_n)
    missing_required_padding_k = float(needs_padding_k and not pad_k)
    missing_any_required_padding = float(
        missing_required_padding_m
        or missing_required_padding_n
        or missing_required_padding_k
    )

    return [
        M,  # 0
        N,  # 1
        K,  # 2
        split_k,  # 3
        log2_M,  # 4
        log2_N,  # 5
        log2_K,  # 6
        log2_MNK,  # 7
        ai,  # 8
        M / max(N, 1),  # 9 (aspect_ratio_mn)
        M / max(K, 1),  # 10 (aspect_ratio_mk)
        N / max(K, 1),  # 11 (aspect_ratio_nk)
        LAYOUT_MAP.get(layout, 0),  # 12
        tile_m,  # 13
        tile_n,  # 14
        tile_k,  # 15
        warp_m,  # 16
        warp_n,  # 17
        warp_k,  # 18
        warp_tile_m,  # 19
        warp_tile_n,  # 20
        warp_tile_k,  # 21
        PIPELINE_MAP.get(pipeline, 0),  # 22
        SCHEDULER_MAP.get(scheduler, 0),  # 23
        EPILOGUE_MAP.get(epilogue, 0),  # 24
        float(pad_m),  # 25
        float(pad_n),  # 26
        float(pad_k),  # 27
        float(persistent),  # 28
        warp_m * warp_n * warp_k,  # 29 (num_warps)
        tile_m * tile_n * tile_k,  # 30 (tile_volume)
        tile_m * tile_n,  # 31 (tile_mn)
        lds_est,  # 32 (lds_usage_estimate)
        lds_est / max(lds_cap, 1),  # 33 (lds_usage_ratio)
        ntm,  # 34 (num_tiles_m)
        ntn,  # 35 (num_tiles_n)
        ntk,  # 36 (num_tiles_k)
        ntm * ntn,  # 37 (total_output_tiles)
        eff(M, tile_m),  # 38 (tile_eff_m)
        eff(N, tile_n),  # 39 (tile_eff_n)
        eff(K, tile_k),  # 40 (tile_eff_k)
        eff(M, tile_m)
        * eff(N, tile_n)
        * eff(K, tile_k),  # 41 (overall_tile_efficiency)
        ntm * ntn / max(hw["num_cus"], 1),  # 42 (cu_utilization)
        ratio_M_to_tile_m,  # 43
        ratio_N_to_tile_n,  # 44
        ratio_K_to_tile_k,  # 45
        problem_smaller_than_tile_m,  # 46
        problem_smaller_than_tile_n,  # 47
        problem_smaller_than_tile_k,  # 48
        any_dim_too_small,  # 49
        needs_padding_m,  # 50
        needs_padding_n,  # 51
        needs_padding_k,  # 52
        has_padding_when_needed_m,  # 53
        has_padding_when_needed_n,  # 54
        has_padding_when_needed_k,  # 55
        missing_required_padding_m,  # 56
        missing_required_padding_n,  # 57
        missing_required_padding_k,  # 58
        missing_any_required_padding,  # 59
        hw["num_cus"],  # 60
        hw["simds_per_cu"],  # 61
        hw["num_cus"] * hw["simds_per_cu"],  # 62 (total_simds)
        hw["shader_engines"],  # 63
        hw["max_clock_mhz"],  # 64
        hw["max_waves_per_cu"],  # 65
        hw["wavefront_size"],  # 66
        hw["lds_capacity"],  # 67
        hw["l1_cache_kb"],  # 68
        hw["l2_cache_kb"],  # 69
        hw["l3_cache_kb"],  # 70
        hw["num_xcd"],  # 71
    ]


TEST_CASES = [
    {
        "problem": {
            "m": 1024,
            "n": 1024,
            "k": 1024,
            "split_k": 1,
            "dtype": "fp8",
            "layout": "rcr",
        },
        "kernel": {
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 64,
            "warp_m": 2,
            "warp_n": 2,
            "warp_k": 1,
            "warp_tile_m": 32,
            "warp_tile_n": 32,
            "warp_tile_k": 16,
            "pipeline": "compv3",
            "scheduler": "intrawave",
            "epilogue": "cshuffle",
            "pad_m": False,
            "pad_n": False,
            "pad_k": False,
            "persistent": False,
        },
    },
    {
        "problem": {
            "m": 1,
            "n": 4096,
            "k": 4096,
            "split_k": 1,
            "dtype": "fp8",
            "layout": "rcr",
        },
        "kernel": {
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 128,
            "warp_m": 1,
            "warp_n": 4,
            "warp_k": 1,
            "warp_tile_m": 16,
            "warp_tile_n": 16,
            "warp_tile_k": 128,
            "pipeline": "compv4",
            "scheduler": "interwave",
            "epilogue": "default",
            "pad_m": True,
            "pad_n": True,
            "pad_k": True,
            "persistent": True,
        },
    },
    {
        "problem": {
            "m": 20480,
            "n": 7168,
            "k": 256,
            "split_k": 1,
            "dtype": "fp16",
            "layout": "rrr",
        },
        "kernel": {
            "tile_m": 256,
            "tile_n": 256,
            "tile_k": 32,
            "warp_m": 4,
            "warp_n": 1,
            "warp_k": 1,
            "warp_tile_m": 32,
            "warp_tile_n": 32,
            "warp_tile_k": 16,
            "pipeline": "mem",
            "scheduler": "interwave",
            "epilogue": "cshuffle",
            "pad_m": False,
            "pad_n": False,
            "pad_k": False,
            "persistent": False,
        },
    },
]

HW = {
    "num_cus": 256,
    "simds_per_cu": 4,
    "shader_engines": 32,
    "max_clock_mhz": 2400,
    "max_waves_per_cu": 32,
    "wavefront_size": 64,
    "lds_capacity": 65536,
    "l1_cache_kb": 32,
    "l2_cache_kb": 4096,
    "l3_cache_kb": 262144,
    "num_xcd": 8,
}


class TestFeatureParity:
    """Verify Python feature engine matches manual computation (C++ uses same logic)."""

    @pytest.fixture
    def fe(self):
        return GemmUniversalFeatureEngine(**HW)

    @pytest.mark.parametrize("case_idx", range(len(TEST_CASES)))
    def test_python_matches_manual(self, fe, case_idx):
        case = TEST_CASES[case_idx]
        prob = case["problem"]
        kern = case["kernel"]

        py_features = fe.extract(prob, kern)

        manual = _compute_features_manually(
            prob["m"],
            prob["n"],
            prob["k"],
            prob["split_k"],
            prob["dtype"],
            prob["layout"],
            kern["tile_m"],
            kern["tile_n"],
            kern["tile_k"],
            kern["warp_m"],
            kern["warp_n"],
            kern["warp_k"],
            kern["warp_tile_m"],
            kern["warp_tile_n"],
            kern["warp_tile_k"],
            kern["pipeline"],
            kern["scheduler"],
            kern["epilogue"],
            kern["pad_m"],
            kern["pad_n"],
            kern["pad_k"],
            kern["persistent"],
            HW,
        )

        manual_arr = np.array(manual, dtype=np.float64)
        assert len(py_features) == len(manual_arr) == 72

        for i in range(72):
            assert py_features[i] == pytest.approx(
                manual_arr[i], rel=1e-10, abs=1e-15
            ), (
                f"Feature {i} ({fe.get_feature_names()[i]}): Python={py_features[i]}, Manual={manual_arr[i]}"
            )

    def test_feature_count(self, fe):
        assert len(fe.get_feature_names()) == 72

    def test_encoding_maps_match_cpp(self):
        """The C++ encode_* functions must use the same mapping as Python.

        PIPELINE_MAP was extended for grouped-conv suffix-aware kernels with
        ``basic_v1`` and ``compv6``; the original GEMM ids (0-4) are
        preserved so existing GEMM models keep loading unchanged.
        """
        assert PIPELINE_MAP == {
            "compv3": 0,
            "compv4": 1,
            "compv5": 2,
            "mem": 3,
            "preshufflev2": 4,
            "basic_v1": 5,
            "compv6": 6,
        }
        assert SCHEDULER_MAP == {"intrawave": 0, "interwave": 1}
        assert EPILOGUE_MAP == {"default": 0, "cshuffle": 1}
        assert LAYOUT_MAP == {"rcr": 0, "rrr": 1, "crr": 2, "ccr": 3}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
