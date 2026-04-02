#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Feature engineering for CK Tile kernel performance prediction.

Provides a strict FeatureEngine interface with per-op subclasses.
All feature engines produce a consistent numpy array for LightGBM.
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "bf8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

LAYOUT_MAP = {"rcr": 0, "rrr": 1, "crr": 2, "ccr": 3}
PIPELINE_MAP = {"compv3": 0, "compv4": 1, "compv5": 2, "mem": 3, "preshufflev2": 4}
SCHEDULER_MAP = {"intrawave": 0, "interwave": 1}
EPILOGUE_MAP = {"default": 0, "cshuffle": 1}


class FeatureEngine(ABC):
    """Abstract base for per-op feature extraction."""

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Ordered list of feature names matching the output array columns."""
        ...

    @abstractmethod
    def get_categorical_features(self) -> list[str]:
        """Feature names that should be treated as categorical by LightGBM."""
        ...

    @abstractmethod
    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        """Extract a single feature vector from a (problem, kernel) pair."""
        ...

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized batch extraction from a DataFrame. Override for speed."""
        names = self.get_feature_names()
        result = np.zeros((len(df), len(names)), dtype=np.float64)
        for i in range(len(df)):
            row = df.iloc[i]
            prob = row.to_dict()
            kern = row.to_dict()
            result[i] = self.extract(prob, kern)
        return result

    def get_parameter_space(self) -> dict[str, list]:
        """Valid discrete values for each kernel parameter (for surrogate search)."""
        return {}

    def get_constraints(self) -> list:
        """Multi-param constraint functions returning True if config is valid."""
        return []

    def validate_config(self, config: dict) -> bool:
        """Check all constraints. Returns True if the config is valid."""
        ps = self.get_parameter_space()
        for k, valid_vals in ps.items():
            if k in config and config[k] not in valid_vals:
                return False
        for constraint in self.get_constraints():
            if not constraint(config):
                return False
        return True

    def project_to_valid(self, config: dict) -> dict:
        """Snap a config to the nearest valid discrete point."""
        ps = self.get_parameter_space()
        result = dict(config)
        for k, valid_vals in ps.items():
            if k not in result:
                continue
            v = result[k]
            if isinstance(valid_vals[0], (int, float)):
                result[k] = min(valid_vals, key=lambda x: abs(x - v))
            elif v not in valid_vals:
                result[k] = valid_vals[0]
        return result


class GemmUniversalFeatureEngine(FeatureEngine):
    """Feature engine for gemm_universal kernels."""

    def __init__(
        self,
        num_cus: int = 256,
        lds_capacity: int = 65536,
        max_clock_mhz: int = 2400,
        simds_per_cu: int = 4,
        shader_engines: int = 32,
        max_waves_per_cu: int = 32,
        wavefront_size: int = 64,
        l1_cache_kb: int = 32,
        l2_cache_kb: int = 4096,
        l3_cache_kb: int = 262144,
        num_xcd: int = 8,
    ):
        self._hw = {
            "num_cus": num_cus,
            "lds_capacity": lds_capacity,
            "max_clock_mhz": max_clock_mhz,
            "simds_per_cu": simds_per_cu,
            "shader_engines": shader_engines,
            "max_waves_per_cu": max_waves_per_cu,
            "wavefront_size": wavefront_size,
            "l1_cache_kb": l1_cache_kb,
            "l2_cache_kb": l2_cache_kb,
            "l3_cache_kb": l3_cache_kb,
            "num_xcd": num_xcd,
            "total_simds": num_cus * simds_per_cu,
        }

    def get_feature_names(self) -> list[str]:
        return [
            # Problem features
            "M",
            "N",
            "K",
            "split_k",
            "log2_M",
            "log2_N",
            "log2_K",
            "log2_MNK",
            "arithmetic_intensity",
            "aspect_ratio_mn",
            "aspect_ratio_mk",
            "aspect_ratio_nk",
            "layout",
            # Kernel features
            "tile_m",
            "tile_n",
            "tile_k",
            "warp_m",
            "warp_n",
            "warp_k",
            "warp_tile_m",
            "warp_tile_n",
            "warp_tile_k",
            "pipeline",
            "scheduler",
            "epilogue",
            "pad_m",
            "pad_n",
            "pad_k",
            "persistent",
            "num_warps",
            "tile_volume",
            "tile_mn",
            "lds_usage_estimate",
            "lds_usage_ratio",
            # Interaction features
            "num_tiles_m",
            "num_tiles_n",
            "num_tiles_k",
            "total_output_tiles",
            "tile_eff_m",
            "tile_eff_n",
            "tile_eff_k",
            "overall_tile_efficiency",
            "cu_utilization",
            # P0 FIX: Problem-to-tile ratio features
            "ratio_M_to_tile_m",
            "ratio_N_to_tile_n",
            "ratio_K_to_tile_k",
            "problem_smaller_than_tile_m",
            "problem_smaller_than_tile_n",
            "problem_smaller_than_tile_k",
            "any_dim_too_small",
            # P1 FIX: Padding requirement interaction features
            "needs_padding_m",
            "needs_padding_n",
            "needs_padding_k",
            "has_padding_when_needed_m",
            "has_padding_when_needed_n",
            "has_padding_when_needed_k",
            "missing_required_padding_m",
            "missing_required_padding_n",
            "missing_required_padding_k",
            "missing_any_required_padding",
            # Hardware features
            "hw_num_cus",
            "hw_simds_per_cu",
            "hw_total_simds",
            "hw_shader_engines",
            "hw_max_clock_mhz",
            "hw_max_waves_per_cu",
            "hw_wavefront_size",
            "hw_lds_capacity",
            "hw_l1_cache_kb",
            "hw_l2_cache_kb",
            "hw_l3_cache_kb",
            "hw_num_xcd",
        ]

    def get_categorical_features(self) -> list[str]:
        return ["layout", "pipeline", "scheduler", "epilogue"]

    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        M = int(problem.get("m", problem.get("M", 0)))
        N = int(problem.get("n", problem.get("N", 0)))
        K = int(problem.get("k", problem.get("K", 0)))
        split_k = int(problem.get("split_k", 1))
        dtype = str(problem.get("dtype", "fp8"))
        bpe = DTYPE_BYTES.get(dtype, 1.0)

        log2_M = math.log2(max(M, 1))
        log2_N = math.log2(max(N, 1))
        log2_K = math.log2(max(K, 1))
        log2_MNK = math.log2(max(M * N * K, 1))

        mem_bytes = (M * K + K * N + M * N) * bpe
        ai = (2.0 * M * N * K) / max(mem_bytes, 1)

        ar_mn = M / max(N, 1)
        ar_mk = M / max(K, 1)
        ar_nk = N / max(K, 1)

        layout_code = LAYOUT_MAP.get(str(problem.get("layout", "rcr")), 0)

        tile_m = int(kernel.get("tile_m", 128))
        tile_n = int(kernel.get("tile_n", 128))
        tile_k = int(kernel.get("tile_k", 64))
        warp_m = int(kernel.get("warp_m", 2))
        warp_n = int(kernel.get("warp_n", 2))
        warp_k = int(kernel.get("warp_k", 1))
        warp_tile_m = int(kernel.get("warp_tile_m", 32))
        warp_tile_n = int(kernel.get("warp_tile_n", 32))
        warp_tile_k = int(kernel.get("warp_tile_k", 16))

        pipeline_code = PIPELINE_MAP.get(str(kernel.get("pipeline", "compv4")), 0)
        scheduler_code = SCHEDULER_MAP.get(str(kernel.get("scheduler", "intrawave")), 0)
        epilogue_code = EPILOGUE_MAP.get(str(kernel.get("epilogue", "cshuffle")), 0)

        pad_m = float(kernel.get("pad_m", False))
        pad_n = float(kernel.get("pad_n", False))
        pad_k = float(kernel.get("pad_k", False))
        persistent = float(kernel.get("persistent", False))

        num_warps = warp_m * warp_n * warp_k
        tile_volume = tile_m * tile_n * tile_k
        tile_mn = tile_m * tile_n

        lds_est = (tile_m * tile_k + tile_n * tile_k) * bpe
        lds_cap = self._hw["lds_capacity"]
        if str(kernel.get("pipeline", "")).startswith("compv4"):
            lds_cap = 32768
        lds_ratio = lds_est / max(lds_cap, 1)

        num_tiles_m = math.ceil(M / max(tile_m, 1))
        num_tiles_n = math.ceil(N / max(tile_n, 1))
        num_tiles_k = math.ceil(K / max(tile_k, 1))
        total_output_tiles = num_tiles_m * num_tiles_n

        rem_m = M % tile_m if tile_m > 0 else 0
        tile_eff_m = rem_m / tile_m if rem_m > 0 else 1.0
        rem_n = N % tile_n if tile_n > 0 else 0
        tile_eff_n = rem_n / tile_n if rem_n > 0 else 1.0
        rem_k = K % tile_k if tile_k > 0 else 0
        tile_eff_k = rem_k / tile_k if rem_k > 0 else 1.0
        overall_eff = tile_eff_m * tile_eff_n * tile_eff_k

        cu_util = total_output_tiles / max(self._hw["num_cus"], 1)

        # P0 FIX: Problem-to-tile ratio features (avoid oversized tiles for tiny problems)
        ratio_M_to_tile_m = M / max(tile_m, 1)
        ratio_N_to_tile_n = N / max(tile_n, 1)
        ratio_K_to_tile_k = K / max(tile_k, 1)

        # Binary features: is problem dimension smaller than tile?
        problem_smaller_than_tile_m = float(M < tile_m)
        problem_smaller_than_tile_n = float(N < tile_n)
        problem_smaller_than_tile_k = float(K < tile_k)
        any_dim_too_small = float((M < tile_m) or (N < tile_n) or (K < tile_k))

        # P1 FIX: Padding requirement features (does this kernel have padding when needed?)
        needs_padding_m = float(M % tile_m != 0) if tile_m > 0 else 0.0
        needs_padding_n = float(N % tile_n != 0) if tile_n > 0 else 0.0
        needs_padding_k = float(K % tile_k != 0) if tile_k > 0 else 0.0

        # Interaction features: kernel has padding capability when problem needs it
        has_padding_when_needed_m = float(needs_padding_m and pad_m)
        has_padding_when_needed_n = float(needs_padding_n and pad_n)
        has_padding_when_needed_k = float(needs_padding_k and pad_k)

        # Critical feature: missing required padding (kernel will likely fail)
        missing_required_padding_m = float(needs_padding_m and not pad_m)
        missing_required_padding_n = float(needs_padding_n and not pad_n)
        missing_required_padding_k = float(needs_padding_k and not pad_k)
        missing_any_required_padding = float(
            missing_required_padding_m
            or missing_required_padding_n
            or missing_required_padding_k
        )

        hw = self._hw
        return np.array(
            [
                M,
                N,
                K,
                split_k,
                log2_M,
                log2_N,
                log2_K,
                log2_MNK,
                ai,
                ar_mn,
                ar_mk,
                ar_nk,
                layout_code,
                tile_m,
                tile_n,
                tile_k,
                warp_m,
                warp_n,
                warp_k,
                warp_tile_m,
                warp_tile_n,
                warp_tile_k,
                pipeline_code,
                scheduler_code,
                epilogue_code,
                pad_m,
                pad_n,
                pad_k,
                persistent,
                num_warps,
                tile_volume,
                tile_mn,
                lds_est,
                lds_ratio,
                num_tiles_m,
                num_tiles_n,
                num_tiles_k,
                total_output_tiles,
                tile_eff_m,
                tile_eff_n,
                tile_eff_k,
                overall_eff,
                cu_util,
                # P0 FIX: New ratio and binary features
                ratio_M_to_tile_m,
                ratio_N_to_tile_n,
                ratio_K_to_tile_k,
                problem_smaller_than_tile_m,
                problem_smaller_than_tile_n,
                problem_smaller_than_tile_k,
                any_dim_too_small,
                # P1 FIX: Padding requirement interaction features
                needs_padding_m,
                needs_padding_n,
                needs_padding_k,
                has_padding_when_needed_m,
                has_padding_when_needed_n,
                has_padding_when_needed_k,
                missing_required_padding_m,
                missing_required_padding_n,
                missing_required_padding_k,
                missing_any_required_padding,
                hw["num_cus"],
                hw["simds_per_cu"],
                hw["total_simds"],
                hw["shader_engines"],
                hw["max_clock_mhz"],
                hw["max_waves_per_cu"],
                hw["wavefront_size"],
                hw["lds_capacity"],
                hw["l1_cache_kb"],
                hw["l2_cache_kb"],
                hw["l3_cache_kb"],
                hw["num_xcd"],
            ],
            dtype=np.float64,
        )

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized batch extraction -- much faster than row-by-row."""
        n = len(df)
        names = self.get_feature_names()
        result = np.zeros((n, len(names)), dtype=np.float64)

        M = df["m"].values.astype(np.float64)
        N = df["n"].values.astype(np.float64)
        K = df["k"].values.astype(np.float64)
        split_k = df["split_k"].fillna(1).values.astype(np.float64)

        dtype_col = df["dtype"].fillna("fp8")
        bpe = dtype_col.map(DTYPE_BYTES).fillna(1.0).values

        result[:, 0] = M
        result[:, 1] = N
        result[:, 2] = K
        result[:, 3] = split_k
        result[:, 4] = np.log2(np.maximum(M, 1))
        result[:, 5] = np.log2(np.maximum(N, 1))
        result[:, 6] = np.log2(np.maximum(K, 1))
        result[:, 7] = np.log2(np.maximum(M * N * K, 1))

        mem = (M * K + K * N + M * N) * bpe
        result[:, 8] = (2.0 * M * N * K) / np.maximum(mem, 1)
        result[:, 9] = M / np.maximum(N, 1)
        result[:, 10] = M / np.maximum(K, 1)
        result[:, 11] = N / np.maximum(K, 1)

        result[:, 12] = df["layout"].map(LAYOUT_MAP).fillna(0).values

        tile_m = df["tile_m"].fillna(128).values.astype(np.float64)
        tile_n = df["tile_n"].fillna(128).values.astype(np.float64)
        tile_k = df["tile_k"].fillna(64).values.astype(np.float64)
        warp_m = df["warp_m"].fillna(2).values.astype(np.float64)
        warp_n = df["warp_n"].fillna(2).values.astype(np.float64)
        warp_k = df["warp_k"].fillna(1).values.astype(np.float64)
        warp_tile_m = df["warp_tile_m"].fillna(32).values.astype(np.float64)
        warp_tile_n = df["warp_tile_n"].fillna(32).values.astype(np.float64)
        warp_tile_k = df["warp_tile_k"].fillna(16).values.astype(np.float64)

        result[:, 13] = tile_m
        result[:, 14] = tile_n
        result[:, 15] = tile_k
        result[:, 16] = warp_m
        result[:, 17] = warp_n
        result[:, 18] = warp_k
        result[:, 19] = warp_tile_m
        result[:, 20] = warp_tile_n
        result[:, 21] = warp_tile_k

        result[:, 22] = df["pipeline"].map(PIPELINE_MAP).fillna(0).values
        result[:, 23] = df["scheduler"].map(SCHEDULER_MAP).fillna(0).values
        result[:, 24] = df["epilogue"].map(EPILOGUE_MAP).fillna(0).values

        result[:, 25] = df["pad_m"].fillna(False).astype(float).values
        result[:, 26] = df["pad_n"].fillna(False).astype(float).values
        result[:, 27] = df["pad_k"].fillna(False).astype(float).values
        result[:, 28] = df["persistent"].fillna(False).astype(float).values

        num_warps = warp_m * warp_n * warp_k
        result[:, 29] = num_warps
        result[:, 30] = tile_m * tile_n * tile_k
        result[:, 31] = tile_m * tile_n

        lds_est = (tile_m * tile_k + tile_n * tile_k) * bpe
        result[:, 32] = lds_est
        lds_cap = np.full(n, self._hw["lds_capacity"], dtype=np.float64)
        is_compv4 = df["pipeline"].fillna("").str.startswith("compv4")
        lds_cap[is_compv4] = 32768
        result[:, 33] = lds_est / np.maximum(lds_cap, 1)

        ntm = np.ceil(M / np.maximum(tile_m, 1))
        ntn = np.ceil(N / np.maximum(tile_n, 1))
        ntk = np.ceil(K / np.maximum(tile_k, 1))
        result[:, 34] = ntm
        result[:, 35] = ntn
        result[:, 36] = ntk
        result[:, 37] = ntm * ntn

        rem_m = np.mod(M, np.maximum(tile_m, 1))
        result[:, 38] = np.where(rem_m > 0, rem_m / tile_m, 1.0)
        rem_n = np.mod(N, np.maximum(tile_n, 1))
        result[:, 39] = np.where(rem_n > 0, rem_n / tile_n, 1.0)
        rem_k = np.mod(K, np.maximum(tile_k, 1))
        result[:, 40] = np.where(rem_k > 0, rem_k / tile_k, 1.0)
        result[:, 41] = result[:, 38] * result[:, 39] * result[:, 40]

        result[:, 42] = (ntm * ntn) / max(self._hw["num_cus"], 1)

        # P0 FIX: Problem-to-tile ratio features
        result[:, 43] = M / np.maximum(tile_m, 1)  # ratio_M_to_tile_m
        result[:, 44] = N / np.maximum(tile_n, 1)  # ratio_N_to_tile_n
        result[:, 45] = K / np.maximum(tile_k, 1)  # ratio_K_to_tile_k

        # Binary features: is problem smaller than tile?
        result[:, 46] = (M < tile_m).astype(float)  # problem_smaller_than_tile_m
        result[:, 47] = (N < tile_n).astype(float)  # problem_smaller_than_tile_n
        result[:, 48] = (K < tile_k).astype(float)  # problem_smaller_than_tile_k
        result[:, 49] = ((M < tile_m) | (N < tile_n) | (K < tile_k)).astype(
            float
        )  # any_dim_too_small

        # P1 FIX: Padding requirement features
        pad_m_bool = df["pad_m"].fillna(False).astype(bool).values
        pad_n_bool = df["pad_n"].fillna(False).astype(bool).values
        pad_k_bool = df["pad_k"].fillna(False).astype(bool).values

        needs_padding_m = (np.mod(M, np.maximum(tile_m, 1)) != 0)
        needs_padding_n = (np.mod(N, np.maximum(tile_n, 1)) != 0)
        needs_padding_k = (np.mod(K, np.maximum(tile_k, 1)) != 0)

        result[:, 50] = needs_padding_m.astype(float)
        result[:, 51] = needs_padding_n.astype(float)
        result[:, 52] = needs_padding_k.astype(float)

        # Interaction features: kernel has padding when problem needs it
        result[:, 53] = (needs_padding_m & pad_m_bool).astype(float)  # has_padding_when_needed_m
        result[:, 54] = (needs_padding_n & pad_n_bool).astype(float)  # has_padding_when_needed_n
        result[:, 55] = (needs_padding_k & pad_k_bool).astype(float)  # has_padding_when_needed_k

        # Critical feature: missing required padding
        result[:, 56] = (needs_padding_m & ~pad_m_bool).astype(float)  # missing_required_padding_m
        result[:, 57] = (needs_padding_n & ~pad_n_bool).astype(float)  # missing_required_padding_n
        result[:, 58] = (needs_padding_k & ~pad_k_bool).astype(float)  # missing_required_padding_k
        result[:, 59] = ((needs_padding_m & ~pad_m_bool) | (needs_padding_n & ~pad_n_bool) | (needs_padding_k & ~pad_k_bool)).astype(float)  # missing_any_required_padding

        # Hardware profile features
        hw = self._hw
        result[:, 60] = hw["num_cus"]
        result[:, 61] = hw["simds_per_cu"]
        result[:, 62] = hw["total_simds"]
        result[:, 63] = hw["shader_engines"]
        result[:, 64] = hw["max_clock_mhz"]
        result[:, 65] = hw["max_waves_per_cu"]
        result[:, 66] = hw["wavefront_size"]
        result[:, 67] = hw["lds_capacity"]
        result[:, 68] = hw["l1_cache_kb"]
        result[:, 69] = hw["l2_cache_kb"]
        result[:, 70] = hw["l3_cache_kb"]
        result[:, 71] = hw["num_xcd"]

        return result

    def get_parameter_space(self) -> dict[str, list]:
        return {
            "tile_m": [32, 64, 128, 192, 256],
            "tile_n": [32, 64, 128, 192, 256],
            "tile_k": [32, 64, 128, 256],
            "warp_m": [1, 2, 4],
            "warp_n": [1, 2, 4],
            "warp_k": [1],
            "warp_tile_m": [4, 16, 32, 64],
            "warp_tile_n": [4, 16, 32, 64],
            "warp_tile_k": [8, 16, 32, 64, 128],
            "pipeline": list(PIPELINE_MAP.keys()),
            "scheduler": list(SCHEDULER_MAP.keys()),
            "epilogue": list(EPILOGUE_MAP.keys()),
            "pad_m": [True, False],
            "pad_n": [True, False],
            "pad_k": [True, False],
            "persistent": [True, False],
        }

    def get_constraints(self) -> list:
        lds_cap = self._hw["lds_capacity"]

        def _lds_constraint(cfg):
            tm = cfg.get("tile_m", 128)
            tn = cfg.get("tile_n", 128)
            tk = cfg.get("tile_k", 64)
            bpe = 1.0  # fp8 default
            est = (tm * tk + tn * tk) * bpe
            cap = (
                32768 if str(cfg.get("pipeline", "")).startswith("compv4") else lds_cap
            )
            return est <= cap

        def _warp_constraint(cfg):
            wm = cfg.get("warp_m", 2)
            wn = cfg.get("warp_n", 2)
            wk = cfg.get("warp_k", 1)
            return (wm * wn * wk) in [2, 4, 8]

        return [_lds_constraint, _warp_constraint]
