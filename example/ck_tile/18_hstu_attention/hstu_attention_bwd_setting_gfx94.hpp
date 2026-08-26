// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_bwd_tile_setting_define.hpp"

#if defined(BUILD_HSTU_FOR_GFX94)
using WarpTile_16x16x16 = ck_tile::sequence<16, 16, 16>;
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 32>;
using WarpTile_32x32x16 = ck_tile::sequence<32, 32, 16>;

template <ck_tile::index_t MaxK>
struct HstuAttentionNoSoftmaxBwdBlockTileForKernel1;

// Tile-sizes: M N0 N0Sub K1 MaxK
//
template <>
struct HstuAttentionNoSoftmaxBwdBlockTileForKernel1<64>
{
    using type             = ck_tile::sequence<128, 64, 32, 32, 64>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxBwdBlockTileForKernel1<96>
{
    using type             = ck_tile::sequence<128, 64, 32, 32, 96>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxBwdBlockTileForKernel1<128>
{
    using type             = ck_tile::sequence<128, 32, 16, 16, 128>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxBwdBlockTileForKernel1<256>
{
    using type             = ck_tile::sequence<64, 32, 16, 16, 256>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionWithSoftmaxBwdBlockTileForKernel1;

// Tile-sizes: M N0 N0Sub K1 MaxK
//
template <>
struct HstuAttentionWithSoftmaxBwdBlockTileForKernel1<64>
{
    using type             = ck_tile::sequence<128, 64, 32, 32, 64>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxBwdBlockTileForKernel1<96>
{
    using type             = ck_tile::sequence<128, 64, 32, 32, 96>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxBwdBlockTileForKernel1<128>
{
    using type             = ck_tile::sequence<128, 32, 16, 16, 128>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxBwdBlockTileForKernel1<256>
{
    using type             = ck_tile::sequence<64, 32, 16, 16, 256>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionNoSoftmaxBwdTileSettingForKernel1;

template <>
struct HstuAttentionNoSoftmaxBwdTileSettingForKernel1<64>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<64>::type,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<64>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<64>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionNoSoftmaxBwdTileSettingForKernel1<96>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<96>::type,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<96>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<96>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionNoSoftmaxBwdTileSettingForKernel1<128>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<128>::type,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<128>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<128>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionNoSoftmaxBwdTileSettingForKernel1<256>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<256>::type,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<256>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxBwdBlockTileForKernel1<256>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionWithSoftmaxBwdTileSettingForKernel1;

template <>
struct HstuAttentionWithSoftmaxBwdTileSettingForKernel1<64>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<64>::type,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<64>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<64>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionWithSoftmaxBwdTileSettingForKernel1<96>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<96>::type,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<96>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<96>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionWithSoftmaxBwdTileSettingForKernel1<128>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<128>::type,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<128>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<128>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionWithSoftmaxBwdTileSettingForKernel1<256>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<256>::type,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<256>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxBwdBlockTileForKernel1<256>::gemm4_warps,
        WarpTile_16x16x16>;
};

/////////////////////////////////////////////////////////////////////////////////////////////
template <ck_tile::index_t MaxK>
struct HstuAttentionBwdBlockTileForKernel2;

// Tile-sizes: M N0 M0Sub K1 MaxK
//
template <>
struct HstuAttentionBwdBlockTileForKernel2<64>
{
    using type             = ck_tile::sequence<32, 128, 16, 16, 64>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<96>
{
    using type             = ck_tile::sequence<32, 128, 16, 16, 96>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<128>
{
    using type             = ck_tile::sequence<32, 64, 16, 16, 128>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<256>
{
    using type             = ck_tile::sequence<32, 64, 16, 16, 256>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionBwdTileSettingForKernel2;

template <>
struct HstuAttentionBwdTileSettingForKernel2<64>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
        typename HstuAttentionBwdBlockTileForKernel2<64>::type,
        typename HstuAttentionBwdBlockTileForKernel2<64>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<64>::gemm1_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<64>::gemm3_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel2<96>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
        typename HstuAttentionBwdBlockTileForKernel2<96>::type,
        typename HstuAttentionBwdBlockTileForKernel2<96>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<96>::gemm1_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<96>::gemm3_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel2<128>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
        typename HstuAttentionBwdBlockTileForKernel2<128>::type,
        typename HstuAttentionBwdBlockTileForKernel2<128>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<128>::gemm1_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<128>::gemm3_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel2<256>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
        typename HstuAttentionBwdBlockTileForKernel2<256>::type,
        typename HstuAttentionBwdBlockTileForKernel2<256>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<256>::gemm1_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<256>::gemm3_warps,
        WarpTile_16x16x16>;
};
#endif
