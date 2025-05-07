/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>

#include "hstu_attention_fwd_type_config.hpp"

template <ck_tile::index_t MaxK>
struct HstuAttentionFwdBlockTile;

// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct HstuAttentionFwdBlockTile<32>
{
    using type        = ck_tile::sequence<64, 64, 16, 32, 32, 32>;
    using gemm0_warps = ck_tile::sequence<2, 1, 1>;
    using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct HstuAttentionFwdBlockTile<64>
{
    using type        = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionFwdBlockTile<128>
{
    using type        = ck_tile::sequence<128, 32, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionFwdBlockTile<256>
{
    using type        = ck_tile::sequence<128, 128, 32, 256, 32, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

using HstuAttentionFwdWarpTile1 = ck_tile::sequence<16, 16, 16>;

template <ck_tile::index_t MaxK>
struct HstuAttentionFwdShape;

template <>
struct HstuAttentionFwdShape<32>
{
    using Type = ck_tile::TileFmhaShape<typename HstuAttentionFwdBlockTile<32>::type,
                                        typename HstuAttentionFwdBlockTile<32>::gemm0_warps,
                                        HstuAttentionFwdWarpTile1,
                                        typename HstuAttentionFwdBlockTile<32>::gemm1_warps,
                                        HstuAttentionFwdWarpTile1,
                                        IsVLayoutRowMajor>;
};

template <>
struct HstuAttentionFwdShape<64>
{
    using Type = ck_tile::TileFmhaShape<typename HstuAttentionFwdBlockTile<64>::type,
                                        typename HstuAttentionFwdBlockTile<64>::gemm0_warps,
                                        HstuAttentionFwdWarpTile1,
                                        typename HstuAttentionFwdBlockTile<64>::gemm1_warps,
                                        HstuAttentionFwdWarpTile1,
                                        IsVLayoutRowMajor>;
};

template <>
struct HstuAttentionFwdShape<128>
{
    using Type = ck_tile::TileFmhaShape<typename HstuAttentionFwdBlockTile<128>::type,
                                        typename HstuAttentionFwdBlockTile<128>::gemm0_warps,
                                        HstuAttentionFwdWarpTile1,
                                        typename HstuAttentionFwdBlockTile<128>::gemm1_warps,
                                        HstuAttentionFwdWarpTile1,
                                        IsVLayoutRowMajor>;
};

template <>
struct HstuAttentionFwdShape<256>
{
    using Type = ck_tile::TileFmhaShape<typename HstuAttentionFwdBlockTile<256>::type,
                                        typename HstuAttentionFwdBlockTile<256>::gemm0_warps,
                                        HstuAttentionFwdWarpTile1,
                                        typename HstuAttentionFwdBlockTile<256>::gemm1_warps,
                                        HstuAttentionFwdWarpTile1,
                                        IsVLayoutRowMajor>;
};
