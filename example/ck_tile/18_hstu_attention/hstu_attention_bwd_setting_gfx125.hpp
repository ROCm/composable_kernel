// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_bwd_tile_setting_define.hpp"

#if defined(BUILD_HSTU_FOR_GFX125)
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 32>;

using HstuAttentionBwdKernel1Gemm0Gemm2Warps = ck_tile::sequence<4, 1, 1>;
using HstuAttentionBwdKernel1Gemm4Warps      = ck_tile::sequence<4, 1, 1>;

using HstuAttentionBwdKernel2Gemm0Gemm2Warps = ck_tile::sequence<1, 4, 1>;
using HstuAttentionBwdKernel2Gemm1Warps      = ck_tile::sequence<4, 1, 1>;
using HstuAttentionBwdKernel2Gemm3Warps      = ck_tile::sequence<4, 1, 1>;

// Kernel1 Tile-sizes: M N0 N0Sub K1 MaxK
//
using HstuAttentionBwdKernel1BlockTile_Hdim64_M128_N0_64_Sub32_K1_32 =
    ck_tile::sequence<128, 64, 32, 32, 64>;
using HstuAttentionBwdKernel1BlockTile_Hdim96_M128_N0_64_Sub32_K1_32 =
    ck_tile::sequence<128, 64, 32, 32, 96>;
using HstuAttentionBwdKernel1BlockTile_Hdim128_M128_N0_64_Sub32_K1_32 =
    ck_tile::sequence<128, 64, 32, 32, 128>;
using HstuAttentionBwdKernel1BlockTile_Hdim256_M128_N0_64_Sub32_K1_32 =
    ck_tile::sequence<128, 64, 32, 32, 256>;

// Kernel2 Tile-sizes: M0 N M0Sub K1 MaxK
//
using HstuAttentionBwdKernel2BlockTile_Hdim64_M0_64_N128_Sub32_K1_32 =
    ck_tile::sequence<64, 128, 32, 32, 64>;
using HstuAttentionBwdKernel2BlockTile_Hdim96_M0_64_N64_Sub32_K1_32 =
    ck_tile::sequence<64, 64, 32, 32, 96>;
using HstuAttentionBwdKernel2BlockTile_Hdim128_M0_64_N64_Sub32_K1_32 =
    ck_tile::sequence<64, 64, 32, 32, 128>;
using HstuAttentionBwdKernel2BlockTile_Hdim256_M0_64_N64_Sub32_K1_32 =
    ck_tile::sequence<64, 64, 32, 32, 256>;

template <ck_tile::index_t MaxK, bool kUseSoftmax>
static constexpr auto GetHstuAttentionBwdKernel1TileSetting()
{
    // NoSoftmax and WithSoftmax share the same Kernel1 block tiles on gfx125.
    if constexpr(MaxK == 64)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
            HstuAttentionBwdKernel1BlockTile_Hdim64_M128_N0_64_Sub32_K1_32,
            HstuAttentionBwdKernel1Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel1Gemm4Warps,
            WarpTile_16x16x32>{};
    }
    else if constexpr(MaxK == 96)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
            HstuAttentionBwdKernel1BlockTile_Hdim96_M128_N0_64_Sub32_K1_32,
            HstuAttentionBwdKernel1Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel1Gemm4Warps,
            WarpTile_16x16x32>{};
    }
    else if constexpr(MaxK == 128)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
            HstuAttentionBwdKernel1BlockTile_Hdim128_M128_N0_64_Sub32_K1_32,
            HstuAttentionBwdKernel1Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel1Gemm4Warps,
            WarpTile_16x16x32>{};
    }
    else if constexpr(MaxK == 256)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel1<
            HstuAttentionBwdKernel1BlockTile_Hdim256_M128_N0_64_Sub32_K1_32,
            HstuAttentionBwdKernel1Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel1Gemm4Warps,
            WarpTile_16x16x32>{};
    }
    else
    {
        static_assert(false, "MaxK size not supported!");
    }
}

template <ck_tile::index_t MaxK>
static constexpr auto GetHstuAttentionBwdKernel2TileSetting()
{
    if constexpr(MaxK == 64)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
            HstuAttentionBwdKernel2BlockTile_Hdim64_M0_64_N128_Sub32_K1_32,
            HstuAttentionBwdKernel2Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm1Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm3Warps,
            WarpTile_16x16x32>{};
    }
    else if constexpr(MaxK == 96)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
            HstuAttentionBwdKernel2BlockTile_Hdim96_M0_64_N64_Sub32_K1_32,
            HstuAttentionBwdKernel2Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm1Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm3Warps,
            WarpTile_16x16x32>{};
    }
    else if constexpr(MaxK == 128)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
            HstuAttentionBwdKernel2BlockTile_Hdim128_M0_64_N64_Sub32_K1_32,
            HstuAttentionBwdKernel2Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm1Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm3Warps,
            WarpTile_16x16x32>{};
    }
    else if constexpr(MaxK == 256)
    {
        return ck_tile::HstuAttentionBwdTileSettingClassForKernel2<
            HstuAttentionBwdKernel2BlockTile_Hdim256_M0_64_N64_Sub32_K1_32,
            HstuAttentionBwdKernel2Gemm0Gemm2Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm1Warps,
            WarpTile_16x16x32,
            HstuAttentionBwdKernel2Gemm3Warps,
            WarpTile_16x16x32>{};
    }
    else
    {
        static_assert(false, "MaxK size not supported!");
    }
}
#endif
