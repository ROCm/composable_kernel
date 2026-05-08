#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Feature engineering for grouped convolution kernel performance prediction.

Extends the FeatureEngine interface to support grouped convolution operations.
Follows the same pattern as GEMM: hardware parameters are read from the data
(hw_* columns) with fallback defaults for gfx950.
"""

import math
import numpy as np
import pandas as pd

from feature_engine import FeatureEngine, DTYPE_BYTES, PIPELINE_MAP


class GroupedConvFeatureEngine(FeatureEngine):
    """Feature engine for grouped_conv kernels.

    Hardware parameters are initialized from defaults but can be overridden
    by reading from data columns (hw_num_cus, hw_max_clock_mhz, etc.)
    """

    def __init__(
        self,
        num_cus: int = 256,  # gfx950 MI300 default
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
            # Problem features (30 -> 38 with Tier-1 additions -> 46 with 3D support)
            "N",
            "C",
            "K",
            "G",
            "Hi",
            "Wi",
            "Y",
            "X",
            "stride_h",
            "stride_w",
            "pad_h",
            "pad_w",
            "Ho",
            "Wo",  # Computed output dimensions
            "log2_N",
            "log2_C",
            "log2_K",
            "log2_G",
            "log2_Hi",
            "log2_Wi",
            "log2_spatial",  # log2(Hi * Wi) for 2D, log2(Di * Hi * Wi) for 3D
            "log2_filter",  # log2(Y * X) for 2D, log2(Z * Y * X) for 3D
            "log2_output",  # log2(Ho * Wo) for 2D, log2(Do * Ho * Wo) for 3D
            "arithmetic_intensity",
            "filter_area",  # Y * X for 2D, Z * Y * X for 3D
            "is_1x1_conv",
            "is_3x3_conv",
            "channels_per_group",  # C / G
            "aspect_ratio_hw",  # Hi / Wi
            "aspect_ratio_filter",  # Y / X
            # 3D-specific features (8 new)
            "is_3d",  # 1.0 if 3D conv, 0.0 if 2D
            "Di",  # Depth input (1 for 2D)
            "Z",  # Filter depth (1 for 2D)
            "Do",  # Depth output (1 for 2D)
            "stride_d",  # Depth stride (1 for 2D)
            "pad_d",  # Depth padding (0 for 2D)
            "dilation_h",  # Height dilation
            "dilation_w",  # Width dilation
            # Tier-1 Group-specific features (8)
            "log2_channels_per_group",
            "log2_output_channels_per_group",
            "is_depthwise",
            "group_density",
            "is_small_group",
            "channels_product_per_group",
            "batch_group_product",
            "is_small_batch_grouped",
            # Kernel features (15 -> 21 with Tier-1 additions)
            "block_size",
            "gemm_m_per_block",
            "gemm_n_per_block",
            "pipeline",
            "num_warps",  # Estimated from block_size
            "tile_volume",  # gemm_m * gemm_n * block_size
            "tile_mn",  # gemm_m * gemm_n
            "lds_usage_estimate",
            "lds_usage_ratio",
            "block_tile_ratio_m",  # gemm_m / block_size
            "block_tile_ratio_n",  # gemm_n / block_size
            "block_efficiency",  # Degree to which block is square-like
            "is_compv3",
            "is_compv4",
            "is_compv5",
            # Suffix-aware kernel features (6 new)
            "is_intrawave",  # 1.0 if wave_mode == "intrawave", 0.0 if "interwave"
            "has_dsb",  # 1.0 if double smem buffer suffix present
            "has_si",  # 1.0 if store-immediate suffix present
            "is_basic",  # 1.0 if pipeline starts with "basic_v"
            "is_compv6",  # 1.0 if pipeline == "compv6"
            "is_mem",  # 1.0 if pipeline == "mem"
            # Interaction features (18)
            "gemm_m_output",  # Effective GEMM M: N * Ho * Wo
            "gemm_n_output",  # Effective GEMM N: K
            "gemm_k_output",  # Effective GEMM K: (C/G) * Y * X
            "num_tiles_m",
            "num_tiles_n",
            "num_tiles_k",
            "total_output_tiles",
            "tile_eff_m",
            "tile_eff_n",
            "tile_eff_k",
            "overall_tile_efficiency",
            "cu_utilization",
            "ratio_gemm_m_to_tile_m",
            "ratio_gemm_n_to_tile_n",
            "ratio_gemm_k_to_tile_k",
            "problem_smaller_than_tile_m",
            "problem_smaller_than_tile_n",
            "problem_smaller_than_tile_k",
            # Hardware features (12)
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
        return ["pipeline"]

    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        # Problem features - 2D and 3D
        N = int(problem.get("N", 1))
        C = int(problem.get("C", 64))
        K = int(problem.get("K", 64))
        G = int(problem.get("G", 1))
        Hi = int(problem.get("Hi", 32))
        Wi = int(problem.get("Wi", 32))
        Di = int(problem.get("Di", 1))  # 3D support
        Y = int(problem.get("Y", 1))
        X = int(problem.get("X", 1))
        Z = int(problem.get("Z", 1))  # 3D support
        stride_h = int(problem.get("stride_h", 1))
        stride_w = int(problem.get("stride_w", 1))
        stride_d = int(problem.get("stride_d", 1))  # 3D support
        pad_h = int(problem.get("pad_h", 0))
        pad_w = int(problem.get("pad_w", 0))
        pad_d = int(problem.get("pad_d", 0))  # 3D support
        dilation_h = int(problem.get("dilation_h", 1))
        dilation_w = int(problem.get("dilation_w", 1))
        dilation_d = int(problem.get("dilation_d", 1))  # 3D support

        # Determine if 3D convolution
        is_3d = float(Di > 1 or Z > 1 or pad_d > 0)

        # Compute output dimensions (match GroupedConvProblem.Ho/Wo/Do formula)
        eff_y = (Y - 1) * dilation_h + 1
        eff_x = (X - 1) * dilation_w + 1
        eff_z = (Z - 1) * dilation_d + 1
        Ho = (Hi + 2 * pad_h - eff_y) // stride_h + 1
        Wo = (Wi + 2 * pad_w - eff_x) // stride_w + 1
        Do = (Di + 2 * pad_d - eff_z) // stride_d + 1 if is_3d else 1

        # Log features (adjusted for 3D)
        log2_N = math.log2(max(N, 1))
        log2_C = math.log2(max(C, 1))
        log2_K = math.log2(max(K, 1))
        log2_G = math.log2(max(G, 1))
        log2_Hi = math.log2(max(Hi, 1))
        log2_Wi = math.log2(max(Wi, 1))
        # For 3D: spatial includes depth dimension
        spatial_volume = Di * Hi * Wi if is_3d else Hi * Wi
        filter_volume = Z * Y * X if is_3d else Y * X
        output_volume = Do * Ho * Wo if is_3d else Ho * Wo
        log2_spatial = math.log2(max(spatial_volume, 1))
        log2_filter = math.log2(max(filter_volume, 1))
        log2_output = math.log2(max(output_volume, 1))

        # Arithmetic intensity (FLOPs / bytes) - adjusted for 3D
        dtype = str(problem.get("dtype", "bf16"))
        bpe = DTYPE_BYTES.get(dtype, 2.0)

        # FLOPs: N * K * output_volume * (C/G) * filter_volume * 2 (MAC)
        flops = N * K * output_volume * (C / max(G, 1)) * filter_volume * 2

        # Bytes: input + filter + output (adjusted for 3D)
        input_bytes = N * C * spatial_volume * bpe
        filter_bytes = K * (C / max(G, 1)) * filter_volume * bpe
        output_bytes = N * K * output_volume * bpe
        bytes_transferred = input_bytes + filter_bytes + output_bytes
        ai = flops / max(bytes_transferred, 1)

        # Derived problem features (adjusted for 3D)
        filter_area = filter_volume  # Y * X for 2D, Z * Y * X for 3D
        is_1x1_conv = float(Y == 1 and X == 1 and Z == 1)
        is_3x3_conv = (
            float(Y == 3 and X == 3 and Z == 3) if is_3d else float(Y == 3 and X == 3)
        )
        channels_per_group = C / max(G, 1)
        aspect_ratio_hw = Hi / max(Wi, 1)
        aspect_ratio_filter = Y / max(X, 1)

        # Tier-1 Group-specific features (8)
        output_channels_per_group = K / max(G, 1)
        log2_channels_per_group = math.log2(max(channels_per_group, 1))
        log2_output_channels_per_group = math.log2(max(output_channels_per_group, 1))
        is_depthwise = float(G == C and G == K)
        group_density = G / max(C, 1)
        is_small_group = float(
            channels_per_group < 16 or output_channels_per_group < 16
        )
        channels_product_per_group = channels_per_group * output_channels_per_group
        batch_group_product = N * G
        is_small_batch_grouped = float(N < 8 and G > 1)

        # Kernel features
        block_size = int(kernel.get("block_size", 16))
        gemm_m_per_block = int(kernel.get("gemm_m_per_block", 64))
        gemm_n_per_block = int(kernel.get("gemm_n_per_block", 64))
        pipeline_str = str(kernel.get("pipeline", "compv3"))
        pipeline_code = PIPELINE_MAP.get(pipeline_str, 0)

        # Estimate warps (assuming 256 thread block)
        num_warps = block_size / 4.0

        tile_volume = gemm_m_per_block * gemm_n_per_block * block_size
        tile_mn = gemm_m_per_block * gemm_n_per_block

        # LDS usage estimate
        lds_est = (gemm_m_per_block * block_size + gemm_n_per_block * block_size) * bpe
        lds_cap = self._hw["lds_capacity"]
        if pipeline_str.startswith("compv4"):
            lds_cap = 32768
        lds_ratio = lds_est / max(lds_cap, 1)

        # Kernel derived features
        block_tile_ratio_m = gemm_m_per_block / max(block_size, 1)
        block_tile_ratio_n = gemm_n_per_block / max(block_size, 1)
        block_efficiency = min(gemm_m_per_block, gemm_n_per_block) / max(
            gemm_m_per_block, gemm_n_per_block, 1
        )
        is_compv3 = float(pipeline_str == "compv3")
        is_compv4 = float(pipeline_str == "compv4")
        is_compv5 = float(pipeline_str == "compv5")

        # Suffix-aware kernel features (6 new)
        wave_mode_str = str(kernel.get("wave_mode", "intrawave"))
        is_intrawave = float(wave_mode_str == "intrawave")
        has_dsb = float(int(kernel.get("has_dsb", 0)))
        has_si = float(int(kernel.get("has_si", 0)))
        is_basic = float(pipeline_str.startswith("basic_v"))
        is_compv6 = float(pipeline_str == "compv6")
        is_mem = float(pipeline_str == "mem")

        # Interaction features - Map conv to GEMM dimensions (adjusted for 3D)
        # GEMM M: N * output_volume (N * Do * Ho * Wo for 3D, N * Ho * Wo for 2D)
        # GEMM N: K (output channels)
        # GEMM K: (C/G) * filter_volume ((C/G) * Z * Y * X for 3D, (C/G) * Y * X for 2D)
        gemm_m = N * output_volume
        gemm_n = K
        gemm_k = int(channels_per_group * filter_volume)

        num_tiles_m = math.ceil(gemm_m / max(gemm_m_per_block, 1))
        num_tiles_n = math.ceil(gemm_n / max(gemm_n_per_block, 1))
        num_tiles_k = math.ceil(gemm_k / max(block_size, 1))
        total_output_tiles = num_tiles_m * num_tiles_n

        rem_m = gemm_m % gemm_m_per_block if gemm_m_per_block > 0 else 0
        tile_eff_m = rem_m / gemm_m_per_block if rem_m > 0 else 1.0
        rem_n = gemm_n % gemm_n_per_block if gemm_n_per_block > 0 else 0
        tile_eff_n = rem_n / gemm_n_per_block if rem_n > 0 else 1.0
        rem_k = gemm_k % block_size if block_size > 0 else 0
        tile_eff_k = rem_k / block_size if rem_k > 0 else 1.0
        overall_eff = tile_eff_m * tile_eff_n * tile_eff_k

        cu_util = total_output_tiles / max(self._hw["num_cus"], 1)

        # Problem-to-tile ratios
        ratio_gemm_m_to_tile_m = gemm_m / max(gemm_m_per_block, 1)
        ratio_gemm_n_to_tile_n = gemm_n / max(gemm_n_per_block, 1)
        ratio_gemm_k_to_tile_k = gemm_k / max(block_size, 1)

        problem_smaller_than_tile_m = float(gemm_m < gemm_m_per_block)
        problem_smaller_than_tile_n = float(gemm_n < gemm_n_per_block)
        problem_smaller_than_tile_k = float(gemm_k < block_size)

        hw = self._hw
        return np.array(
            [
                # Problem features (30)
                N,
                C,
                K,
                G,
                Hi,
                Wi,
                Y,
                X,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                Ho,
                Wo,
                log2_N,
                log2_C,
                log2_K,
                log2_G,
                log2_Hi,
                log2_Wi,
                log2_spatial,
                log2_filter,
                log2_output,
                ai,
                filter_area,
                is_1x1_conv,
                is_3x3_conv,
                channels_per_group,
                aspect_ratio_hw,
                aspect_ratio_filter,
                # 3D-specific features (8)
                is_3d,
                Di,
                Z,
                Do,
                stride_d,
                pad_d,
                dilation_h,
                dilation_w,
                # Tier-1 Group-specific features (8)
                log2_channels_per_group,
                log2_output_channels_per_group,
                is_depthwise,
                group_density,
                is_small_group,
                channels_product_per_group,
                batch_group_product,
                is_small_batch_grouped,
                # Kernel features (15)
                block_size,
                gemm_m_per_block,
                gemm_n_per_block,
                pipeline_code,
                num_warps,
                tile_volume,
                tile_mn,
                lds_est,
                lds_ratio,
                block_tile_ratio_m,
                block_tile_ratio_n,
                block_efficiency,
                is_compv3,
                is_compv4,
                is_compv5,
                # Suffix-aware kernel features (6)
                is_intrawave,
                has_dsb,
                has_si,
                is_basic,
                is_compv6,
                is_mem,
                # Interaction features (18)
                gemm_m,
                gemm_n,
                gemm_k,
                num_tiles_m,
                num_tiles_n,
                num_tiles_k,
                total_output_tiles,
                tile_eff_m,
                tile_eff_n,
                tile_eff_k,
                overall_eff,
                cu_util,
                ratio_gemm_m_to_tile_m,
                ratio_gemm_n_to_tile_n,
                ratio_gemm_k_to_tile_k,
                problem_smaller_than_tile_m,
                problem_smaller_than_tile_n,
                problem_smaller_than_tile_k,
                # Hardware features (12)
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

        # Extract problem features (2D and 3D)
        N = df["N"].values.astype(np.float64)
        C = df["C"].values.astype(np.float64)
        K = df["K"].values.astype(np.float64)
        G = df["G"].values.astype(np.float64)
        Hi = df["Hi"].values.astype(np.float64)
        Wi = df["Wi"].values.astype(np.float64)
        Y = df["Y"].values.astype(np.float64)
        X = df["X"].values.astype(np.float64)
        stride_h = df["stride_h"].values.astype(np.float64)
        stride_w = df["stride_w"].values.astype(np.float64)
        pad_h = df["pad_h"].values.astype(np.float64)
        pad_w = df["pad_w"].values.astype(np.float64)

        # 3D parameters (default to 1 for 2D convolutions)
        Di = df.get("Di", pd.Series(np.ones(n))).values.astype(np.float64)
        Z = df.get("Z", pd.Series(np.ones(n))).values.astype(np.float64)
        stride_d = df.get("stride_d", pd.Series(np.ones(n))).values.astype(np.float64)
        pad_d = df.get("pad_d", pd.Series(np.zeros(n))).values.astype(np.float64)

        # Dilation defaults to 1 if not present (standard convolution)
        dilation_h = df.get("dilation_h", pd.Series(np.ones(n))).values.astype(
            np.float64
        )
        dilation_w = df.get("dilation_w", pd.Series(np.ones(n))).values.astype(
            np.float64
        )
        dilation_d = df.get("dilation_d", pd.Series(np.ones(n))).values.astype(
            np.float64
        )

        # Determine if 3D convolution
        is_3d = ((Di > 1) | (Z > 1) | (pad_d > 0)).astype(np.float64)

        # Compute output dimensions (match GroupedConvProblem.Ho/Wo/Do formula)
        eff_y = (Y - 1) * dilation_h + 1
        eff_x = (X - 1) * dilation_w + 1
        eff_z = (Z - 1) * dilation_d + 1
        Ho = (Hi + 2 * pad_h - eff_y) // stride_h + 1
        Wo = (Wi + 2 * pad_w - eff_x) // stride_w + 1
        Do = np.where(is_3d, (Di + 2 * pad_d - eff_z) // stride_d + 1, 1.0)

        # Log features (adjusted for 3D)
        log2_N = np.log2(np.maximum(N, 1))
        log2_C = np.log2(np.maximum(C, 1))
        log2_K = np.log2(np.maximum(K, 1))
        log2_G = np.log2(np.maximum(G, 1))
        log2_Hi = np.log2(np.maximum(Hi, 1))
        log2_Wi = np.log2(np.maximum(Wi, 1))
        # For 3D: spatial includes depth dimension
        spatial_volume = np.where(is_3d, Di * Hi * Wi, Hi * Wi)
        filter_volume = np.where(is_3d, Z * Y * X, Y * X)
        output_volume = np.where(is_3d, Do * Ho * Wo, Ho * Wo)
        log2_spatial = np.log2(np.maximum(spatial_volume, 1))
        log2_filter = np.log2(np.maximum(filter_volume, 1))
        log2_output = np.log2(np.maximum(output_volume, 1))

        # Arithmetic intensity (vectorized per-row for mixed-dtype batches)
        if "dtype" in df.columns:
            bpe = df["dtype"].map(DTYPE_BYTES).fillna(2.0).values.astype(np.float64)
        else:
            bpe = np.full(n, 2.0, dtype=np.float64)  # Default to bf16 bpe=2

        # FLOPs and arithmetic intensity (adjusted for 3D)
        flops = N * K * output_volume * (C / np.maximum(G, 1)) * filter_volume * 2
        input_bytes = N * C * spatial_volume * bpe
        filter_bytes = K * (C / np.maximum(G, 1)) * filter_volume * bpe
        output_bytes = N * K * output_volume * bpe
        bytes_transferred = input_bytes + filter_bytes + output_bytes
        ai = flops / np.maximum(bytes_transferred, 1)

        # Derived problem features (adjusted for 3D)
        filter_area = filter_volume  # Y * X for 2D, Z * Y * X for 3D
        is_1x1_conv = np.where(
            is_3d,
            ((Y == 1) & (X == 1) & (Z == 1)).astype(np.float64),
            ((Y == 1) & (X == 1)).astype(np.float64),
        )
        is_3x3_conv = np.where(
            is_3d,
            ((Y == 3) & (X == 3) & (Z == 3)).astype(np.float64),
            ((Y == 3) & (X == 3)).astype(np.float64),
        )
        channels_per_group = C / np.maximum(G, 1)
        aspect_ratio_hw = Hi / np.maximum(Wi, 1)
        aspect_ratio_filter = Y / np.maximum(X, 1)

        # Tier-1 Group-specific features (8)
        output_channels_per_group = K / np.maximum(G, 1)
        log2_channels_per_group = np.log2(np.maximum(channels_per_group, 1))
        log2_output_channels_per_group = np.log2(
            np.maximum(output_channels_per_group, 1)
        )
        is_depthwise = ((G == C) & (G == K)).astype(np.float64)
        group_density = G / np.maximum(C, 1)
        is_small_group = (
            (channels_per_group < 16) | (output_channels_per_group < 16)
        ).astype(np.float64)
        channels_product_per_group = channels_per_group * output_channels_per_group
        batch_group_product = N * G
        is_small_batch_grouped = ((N < 8) & (G > 1)).astype(np.float64)

        # Kernel features
        block_size = df["block_size"].values.astype(np.float64)
        gemm_m_per_block = df["gemm_m_per_block"].values.astype(np.float64)
        gemm_n_per_block = df["gemm_n_per_block"].values.astype(np.float64)
        pipeline_code = (
            df["pipeline"].map(PIPELINE_MAP).fillna(0).values.astype(np.float64)
        )

        num_warps = block_size / 4.0
        tile_volume = gemm_m_per_block * gemm_n_per_block * block_size
        tile_mn = gemm_m_per_block * gemm_n_per_block

        # LDS usage
        lds_est = (gemm_m_per_block * block_size + gemm_n_per_block * block_size) * bpe
        lds_cap = np.full(n, self._hw["lds_capacity"], dtype=np.float64)
        is_compv4 = (df["pipeline"] == "compv4").values
        lds_cap[is_compv4] = 32768
        lds_ratio = lds_est / np.maximum(lds_cap, 1)

        # Kernel derived features
        block_tile_ratio_m = gemm_m_per_block / np.maximum(block_size, 1)
        block_tile_ratio_n = gemm_n_per_block / np.maximum(block_size, 1)
        block_efficiency = np.minimum(gemm_m_per_block, gemm_n_per_block) / np.maximum(
            np.maximum(gemm_m_per_block, gemm_n_per_block), 1
        )
        is_compv3_arr = (df["pipeline"] == "compv3").values.astype(np.float64)
        is_compv4_arr = (df["pipeline"] == "compv4").values.astype(np.float64)
        is_compv5_arr = (df["pipeline"] == "compv5").values.astype(np.float64)

        # Suffix-aware kernel features (6 new). Use df.get() with sensible defaults
        # so old parquets without these columns still load.
        wave_mode_series = df.get(
            "wave_mode", pd.Series(["intrawave"] * n, index=df.index)
        )
        is_intrawave_arr = (wave_mode_series == "intrawave").values.astype(np.float64)
        has_dsb_arr = (
            df.get("has_dsb", pd.Series(np.zeros(n), index=df.index))
            .fillna(0)
            .values.astype(np.float64)
        )
        has_si_arr = (
            df.get("has_si", pd.Series(np.zeros(n), index=df.index))
            .fillna(0)
            .values.astype(np.float64)
        )
        is_basic_arr = (
            df["pipeline"]
            .astype(str)
            .str.startswith("basic_v")
            .values.astype(np.float64)
        )
        is_compv6_arr = (df["pipeline"] == "compv6").values.astype(np.float64)
        is_mem_arr = (df["pipeline"] == "mem").values.astype(np.float64)

        # Interaction features (adjusted for 3D)
        # GEMM M: N * output_volume (N * Do * Ho * Wo for 3D, N * Ho * Wo for 2D)
        # GEMM N: K (output channels)
        # GEMM K: channels_per_group * filter_volume
        gemm_m = N * output_volume
        gemm_n = K
        gemm_k = (channels_per_group * filter_volume).astype(np.int64)

        num_tiles_m = np.ceil(gemm_m / np.maximum(gemm_m_per_block, 1))
        num_tiles_n = np.ceil(gemm_n / np.maximum(gemm_n_per_block, 1))
        num_tiles_k = np.ceil(gemm_k / np.maximum(block_size, 1))
        total_output_tiles = num_tiles_m * num_tiles_n

        rem_m = np.where(gemm_m_per_block > 0, gemm_m % gemm_m_per_block, 0)
        tile_eff_m = np.where(rem_m > 0, rem_m / gemm_m_per_block, 1.0)
        rem_n = np.where(gemm_n_per_block > 0, gemm_n % gemm_n_per_block, 0)
        tile_eff_n = np.where(rem_n > 0, rem_n / gemm_n_per_block, 1.0)
        rem_k = np.where(block_size > 0, gemm_k % block_size, 0)
        tile_eff_k = np.where(rem_k > 0, rem_k / block_size, 1.0)
        overall_eff = tile_eff_m * tile_eff_n * tile_eff_k

        cu_util = total_output_tiles / max(self._hw["num_cus"], 1)

        # Problem-to-tile ratios
        ratio_gemm_m_to_tile_m = gemm_m / np.maximum(gemm_m_per_block, 1)
        ratio_gemm_n_to_tile_n = gemm_n / np.maximum(gemm_n_per_block, 1)
        ratio_gemm_k_to_tile_k = gemm_k / np.maximum(block_size, 1)

        problem_smaller_than_tile_m = (gemm_m < gemm_m_per_block).astype(np.float64)
        problem_smaller_than_tile_n = (gemm_n < gemm_n_per_block).astype(np.float64)
        problem_smaller_than_tile_k = (gemm_k < block_size).astype(np.float64)

        hw = self._hw

        # Assemble feature matrix column by column
        idx = 0
        result[:, idx] = N
        idx += 1
        result[:, idx] = C
        idx += 1
        result[:, idx] = K
        idx += 1
        result[:, idx] = G
        idx += 1
        result[:, idx] = Hi
        idx += 1
        result[:, idx] = Wi
        idx += 1
        result[:, idx] = Y
        idx += 1
        result[:, idx] = X
        idx += 1
        result[:, idx] = stride_h
        idx += 1
        result[:, idx] = stride_w
        idx += 1
        result[:, idx] = pad_h
        idx += 1
        result[:, idx] = pad_w
        idx += 1
        result[:, idx] = Ho
        idx += 1
        result[:, idx] = Wo
        idx += 1
        result[:, idx] = log2_N
        idx += 1
        result[:, idx] = log2_C
        idx += 1
        result[:, idx] = log2_K
        idx += 1
        result[:, idx] = log2_G
        idx += 1
        result[:, idx] = log2_Hi
        idx += 1
        result[:, idx] = log2_Wi
        idx += 1
        result[:, idx] = log2_spatial
        idx += 1
        result[:, idx] = log2_filter
        idx += 1
        result[:, idx] = log2_output
        idx += 1
        result[:, idx] = ai
        idx += 1
        result[:, idx] = filter_area
        idx += 1
        result[:, idx] = is_1x1_conv
        idx += 1
        result[:, idx] = is_3x3_conv
        idx += 1
        result[:, idx] = channels_per_group
        idx += 1
        result[:, idx] = aspect_ratio_hw
        idx += 1
        result[:, idx] = aspect_ratio_filter
        idx += 1
        # 3D-specific features (8)
        result[:, idx] = is_3d
        idx += 1
        result[:, idx] = Di
        idx += 1
        result[:, idx] = Z
        idx += 1
        result[:, idx] = Do
        idx += 1
        result[:, idx] = stride_d
        idx += 1
        result[:, idx] = pad_d
        idx += 1
        result[:, idx] = dilation_h
        idx += 1
        result[:, idx] = dilation_w
        idx += 1
        # Tier-1 Group-specific features (8)
        result[:, idx] = log2_channels_per_group
        idx += 1
        result[:, idx] = log2_output_channels_per_group
        idx += 1
        result[:, idx] = is_depthwise
        idx += 1
        result[:, idx] = group_density
        idx += 1
        result[:, idx] = is_small_group
        idx += 1
        result[:, idx] = channels_product_per_group
        idx += 1
        result[:, idx] = batch_group_product
        idx += 1
        result[:, idx] = is_small_batch_grouped
        idx += 1
        # Kernel features
        result[:, idx] = block_size
        idx += 1
        result[:, idx] = gemm_m_per_block
        idx += 1
        result[:, idx] = gemm_n_per_block
        idx += 1
        result[:, idx] = pipeline_code
        idx += 1
        result[:, idx] = num_warps
        idx += 1
        result[:, idx] = tile_volume
        idx += 1
        result[:, idx] = tile_mn
        idx += 1
        result[:, idx] = lds_est
        idx += 1
        result[:, idx] = lds_ratio
        idx += 1
        result[:, idx] = block_tile_ratio_m
        idx += 1
        result[:, idx] = block_tile_ratio_n
        idx += 1
        result[:, idx] = block_efficiency
        idx += 1
        result[:, idx] = is_compv3_arr
        idx += 1
        result[:, idx] = is_compv4_arr
        idx += 1
        result[:, idx] = is_compv5_arr
        idx += 1
        # Suffix-aware kernel features (6)
        result[:, idx] = is_intrawave_arr
        idx += 1
        result[:, idx] = has_dsb_arr
        idx += 1
        result[:, idx] = has_si_arr
        idx += 1
        result[:, idx] = is_basic_arr
        idx += 1
        result[:, idx] = is_compv6_arr
        idx += 1
        result[:, idx] = is_mem_arr
        idx += 1
        result[:, idx] = gemm_m
        idx += 1
        result[:, idx] = gemm_n
        idx += 1
        result[:, idx] = gemm_k
        idx += 1
        result[:, idx] = num_tiles_m
        idx += 1
        result[:, idx] = num_tiles_n
        idx += 1
        result[:, idx] = num_tiles_k
        idx += 1
        result[:, idx] = total_output_tiles
        idx += 1
        result[:, idx] = tile_eff_m
        idx += 1
        result[:, idx] = tile_eff_n
        idx += 1
        result[:, idx] = tile_eff_k
        idx += 1
        result[:, idx] = overall_eff
        idx += 1
        result[:, idx] = cu_util
        idx += 1
        result[:, idx] = ratio_gemm_m_to_tile_m
        idx += 1
        result[:, idx] = ratio_gemm_n_to_tile_n
        idx += 1
        result[:, idx] = ratio_gemm_k_to_tile_k
        idx += 1
        result[:, idx] = problem_smaller_than_tile_m
        idx += 1
        result[:, idx] = problem_smaller_than_tile_n
        idx += 1
        result[:, idx] = problem_smaller_than_tile_k
        idx += 1
        result[:, idx] = hw["num_cus"]
        idx += 1
        result[:, idx] = hw["simds_per_cu"]
        idx += 1
        result[:, idx] = hw["total_simds"]
        idx += 1
        result[:, idx] = hw["shader_engines"]
        idx += 1
        result[:, idx] = hw["max_clock_mhz"]
        idx += 1
        result[:, idx] = hw["max_waves_per_cu"]
        idx += 1
        result[:, idx] = hw["wavefront_size"]
        idx += 1
        result[:, idx] = hw["lds_capacity"]
        idx += 1
        result[:, idx] = hw["l1_cache_kb"]
        idx += 1
        result[:, idx] = hw["l2_cache_kb"]
        idx += 1
        result[:, idx] = hw["l3_cache_kb"]
        idx += 1
        result[:, idx] = hw["num_xcd"]
        idx += 1

        return result
