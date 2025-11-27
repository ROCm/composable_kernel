// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_tile_setting_define.hpp"

using HstuAttentionFwdWarpTile1 = ck_tile::sequence<16, 16, 16>;
using HstuAttentionFwdWarpTile2 = ck_tile::sequence<16, 16, 32>;

#if !defined(BUILD_HSTU_FOR_GFX95_ONLY)
template <ck_tile::index_t MaxK>
struct HstuAttentionNoSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<32>
{
    using type        = ck_tile::sequence<64, 64, 32, 32, 32>;
    using gemm0_warps = ck_tile::sequence<2, 1, 1>;
    using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<64>
{
    using type        = ck_tile::sequence<128, 64, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128>
{
    using type        = ck_tile::sequence<128, 32, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<256>
{
    using type        = ck_tile::sequence<128, 32, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionWithSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<32>
{
    using type        = ck_tile::sequence<64, 64, 32, 32, 32>;
    using gemm0_warps = ck_tile::sequence<2, 1, 1>;
    using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<64>
{
    using type        = ck_tile::sequence<128, 64, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128>
{
    using type        = ck_tile::sequence<128, 64, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<256>
{
    using type        = ck_tile::sequence<128, 32, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionNoSoftmaxFwdTileSetting;

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<32>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<32>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<32>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionNoSoftmaxFwdBlockTile<32>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<256>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionWithSoftmaxFwdTileSetting;

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<32>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<32>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<32>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionWithSoftmaxFwdBlockTile<32>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<256>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm0_warps,
        HstuAttentionFwdWarpTile1,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};
#endif

#if defined(BUILD_HSTU_FOR_GFX95_ONLY)
template <ck_tile::index_t MaxK>
struct HstuAttentionNoSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<32>
{
    using type        = ck_tile::sequence<64, 64, 32, 16, 32>;
    using gemm0_warps = ck_tile::sequence<2, 1, 1>;
    using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<64>
{
    using type        = ck_tile::sequence<128, 64, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128>
{
    using type        = ck_tile::sequence<128, 32, 128, 32, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<256>
{
    using type        = ck_tile::sequence<128, 32, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionWithSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<32>
{
    using type        = ck_tile::sequence<64, 64, 32, 16, 32>;
    using gemm0_warps = ck_tile::sequence<2, 1, 1>;
    using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<64>
{
    using type        = ck_tile::sequence<128, 64, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128>
{
    using type        = ck_tile::sequence<128, 64, 128, 32, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<256>
{
    using type        = ck_tile::sequence<128, 32, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionNoSoftmaxFwdTileSetting;

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<32>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<32>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<32>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionNoSoftmaxFwdBlockTile<32>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128>::gemm1_warps,
        HstuAttentionFwdWarpTile2>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<256>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionWithSoftmaxFwdTileSetting;

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<32>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<32>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<32>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionWithSoftmaxFwdBlockTile<32>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128>::gemm1_warps,
        HstuAttentionFwdWarpTile2>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<256>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm0_warps,
        HstuAttentionFwdWarpTile2,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm1_warps,
        HstuAttentionFwdWarpTile1>;
};
#endif
