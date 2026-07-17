// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_bwd_type_config.hpp"
#include "hstu_attention_tile_setting_define.hpp"
#include "hstu_attention_host_util.hpp"

using WarpTile_16x16x16 = ck_tile::sequence<16, 16, 16>;
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 32>;
using WarpTile_32x32x16 = ck_tile::sequence<32, 32, 16>;

#if !defined(BUILD_HSTU_FOR_GFX95)
template <ck_tile::index_t MaxK>
struct HstuAttentionBwdBlockTileForKernel1;

// Tile-sizes: M N0 N0Sub MaxK
//
template <>
struct HstuAttentionBwdBlockTileForKernel1<64>
{
    using type             = ck_tile::sequence<128, 64, 32, 64>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel1<96>
{
    using type             = ck_tile::sequence<128, 64, 32, 96>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel1<128>
{
    using type             = ck_tile::sequence<64, 32, 16, 128>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel1<256>
{
    using type             = ck_tile::sequence<64, 32, 16, 256>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionBwdTileSettingForKernel1;

template <>
struct HstuAttentionBwdTileSettingForKernel1<64>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<64>::type,
        typename HstuAttentionBwdBlockTileForKernel1<64>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel1<64>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel1<96>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<96>::type,
        typename HstuAttentionBwdBlockTileForKernel1<96>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel1<96>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel1<128>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<128>::type,
        typename HstuAttentionBwdBlockTileForKernel1<128>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel1<128>::gemm4_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel1<256>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<256>::type,
        typename HstuAttentionBwdBlockTileForKernel1<256>::gemm0gemm2_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel1<256>::gemm4_warps,
        WarpTile_16x16x16>;
};

/////////////////////////////////////////////////////////////////////////////////////////////
template <ck_tile::index_t MaxK>
struct HstuAttentionBwdBlockTileForKernel2;

// Tile-sizes: M N0 K0 K1 MaxK
//
template <>
struct HstuAttentionBwdBlockTileForKernel2<64>
{
    using type             = ck_tile::sequence<64, 64, 32, 32, 64>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<96>
{
    using type             = ck_tile::sequence<64, 64, 32, 32, 96>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<128>
{
    using type             = ck_tile::sequence<64, 64, 32, 32, 128>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<256>
{
    using type             = ck_tile::sequence<32, 64, 32, 32, 256>;
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

#if defined(BUILD_HSTU_FOR_GFX95)
template <ck_tile::index_t MaxK>
struct HstuAttentionBwdBlockTileForKernel1;

// Tile-sizes: M N0 N0Sub MaxK
//
template <>
struct HstuAttentionBwdBlockTileForKernel1<64>
{
    using type             = ck_tile::sequence<128, 64, 32, 64>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel1<96>
{
    using type             = ck_tile::sequence<128, 64, 32, 96>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel1<128>
{
    using type             = ck_tile::sequence<64, 64, 32, 128>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel1<256>
{
    using type             = ck_tile::sequence<64, 64, 32, 256>;
    using gemm0gemm2_warps = ck_tile::sequence<4, 1, 1>;
    using gemm4_warps      = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK>
struct HstuAttentionBwdTileSettingForKernel1;

template <>
struct HstuAttentionBwdTileSettingForKernel1<64>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<64>::type,
        typename HstuAttentionBwdBlockTileForKernel1<64>::gemm0gemm2_warps,
        WarpTile_16x16x32,
        typename HstuAttentionBwdBlockTileForKernel1<64>::gemm4_warps,
        WarpTile_16x16x32>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel1<96>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<96>::type,
        typename HstuAttentionBwdBlockTileForKernel1<96>::gemm0gemm2_warps,
        WarpTile_16x16x32,
        typename HstuAttentionBwdBlockTileForKernel1<96>::gemm4_warps,
        WarpTile_16x16x32>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel1<128>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<128>::type,
        typename HstuAttentionBwdBlockTileForKernel1<128>::gemm0gemm2_warps,
        WarpTile_16x16x32,
        typename HstuAttentionBwdBlockTileForKernel1<128>::gemm4_warps,
        WarpTile_16x16x32>;
};

template <>
struct HstuAttentionBwdTileSettingForKernel1<256>
{
    using Type = ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
        typename HstuAttentionBwdBlockTileForKernel1<256>::type,
        typename HstuAttentionBwdBlockTileForKernel1<256>::gemm0gemm2_warps,
        WarpTile_16x16x32,
        typename HstuAttentionBwdBlockTileForKernel1<256>::gemm4_warps,
        WarpTile_16x16x32>;
};

/////////////////////////////////////////////////////////////////////////////////////////////
template <ck_tile::index_t MaxK>
struct HstuAttentionBwdBlockTileForKernel2;

// Tile-sizes: M N0 K0 K1 MaxK
//
template <>
struct HstuAttentionBwdBlockTileForKernel2<64>
{
    using type             = ck_tile::sequence<64, 128, 32, 32, 64>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<96>
{
    using type             = ck_tile::sequence<64, 128, 32, 32, 96>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<128>
{
    using type             = ck_tile::sequence<64, 128, 32, 32, 128>;
    using gemm0gemm2_warps = ck_tile::sequence<1, 4, 1>;
    using gemm1_warps      = ck_tile::sequence<4, 1, 1>;
    using gemm3_warps      = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionBwdBlockTileForKernel2<256>
{
    using type             = ck_tile::sequence<64, 64, 32, 32, 256>;
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
        WarpTile_16x16x32,
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
        WarpTile_16x16x32,
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
        WarpTile_16x16x32,
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
        WarpTile_16x16x32,
        typename HstuAttentionBwdBlockTileForKernel2<256>::gemm1_warps,
        WarpTile_16x16x16,
        typename HstuAttentionBwdBlockTileForKernel2<256>::gemm3_warps,
        WarpTile_16x16x16>;
};
#endif
