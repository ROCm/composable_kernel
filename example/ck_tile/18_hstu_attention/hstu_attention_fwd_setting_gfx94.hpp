// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_fwd_tile_setting_define.hpp"

#if defined(BUILD_HSTU_FOR_GFX94)
using WarpTile_16x16x16 = ck_tile::sequence<16, 16, 16>;
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 32>;
using WarpTile_32x32x16 = ck_tile::sequence<32, 32, 16>;

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionNoSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<64, 64>
{
    using type        = ck_tile::sequence<64, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<64, 128>
{
    using type        = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<96, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 128, 32, 96>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128, 64>
{
    using type        = ck_tile::sequence<64, 32, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128, 128>
{
    using type        = ck_tile::sequence<128, 32, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<256, MTile>
{
    using type        = ck_tile::sequence<128, 32, 16, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionWithSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<64, 64>
{
    using type        = ck_tile::sequence<64, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<64, 128>
{
    using type        = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<96, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 128, 32, 96>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128, 64>
{
    using type        = ck_tile::sequence<64, 64, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128, 128>
{
    using type        = ck_tile::sequence<128, 64, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<256, MTile>
{
    using type        = ck_tile::sequence<128, 32, 16, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionNoSoftmaxFwdTileSetting;

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<64, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<64, 64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64, 64>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64, 64>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<64, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<64, 128>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64, 128>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64, 128>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<96, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::gemm1_warps,
        WarpTile_16x16x16>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<96, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<96, 128>;

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<256, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm1_warps,
        WarpTile_16x16x16>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<256, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<256, 128>;

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionWithSoftmaxFwdTileSetting;

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<64, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<64, 64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64, 64>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64, 64>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<64, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<64, 128>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64, 128>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64, 128>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<96, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::gemm0_warps,
        WarpTile_32x32x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::gemm1_warps,
        WarpTile_32x32x16>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<96, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<96, 128>;

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<256, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm1_warps,
        WarpTile_16x16x16>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<256, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<256, 128>;
#endif
