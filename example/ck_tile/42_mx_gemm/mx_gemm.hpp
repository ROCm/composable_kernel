// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

// GEMM config with 16x16 warp tile
struct MxGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 512;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = true; // comp_async uses double buffer
    static constexpr bool Preshuffle                = false;
    static constexpr ck_tile::index_t BContiguousItemsPerAccess = 16;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;

    using AScaleDataType = ck_tile::e8m0_t;
    using BScaleDataType = ck_tile::e8m0_t;
};

struct MX_GemmConfigEightWaves : MxGemmConfig
{
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2; // NWarps == 2 for ping-pong!
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128 * N_Warp;
    static constexpr ck_tile::index_t K_Tile = 128 * K_Warp;

    static constexpr int kBlockPerCu = 2;
};

struct MX_GemmConfig16 : MxGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 256;
};

struct MXfp4_GemmConfig16_Preshuffle : MxGemmConfig
{
    // Smallest possible N_Tile is 512 for fp4 preshuffle
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 512;
    static constexpr ck_tile::index_t K_Tile = 256;
    static constexpr auto Scheduler          = ck_tile::GemmPipelineScheduler::Default;
    static constexpr bool Preshuffle         = true;
    static constexpr ck_tile::index_t BContiguousItemsPerAccess = 32;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
};

struct MXfp8_GemmConfig16_Preshuffle : MxGemmConfig
{
    // For FP8 Preshuffle:
    // The theoretical functional minimum is N_Tile =  N_Warp * N_Warp_Tile * NXdlPack = 4*16*2 =
    // 128 . For better performance, we would choose N_Repeat = 2 which would yield N_Tile = 128 * 2
    // = 256 . Note: If we use fewer waves, the minimum theoretical N_Tile can be even smaller,
    // reduced to N_Tile = 32 for 1 single wave.
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr auto Scheduler        = ck_tile::GemmPipelineScheduler::Default;
    static constexpr bool Preshuffle       = true;
    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
};
