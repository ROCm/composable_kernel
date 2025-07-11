// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// #define FINEGRADE_LOADSTORE
#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {
template <typename Problem>
struct FlatmmLoopConfig
{
    static constexpr index_t PrefetchStages = 2;

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }
};

// Base class containing all common functionality for both pipeline versions

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct FlatmmPipelineAGmemBGmemCRegBase
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockFlatmm =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockFlatmm<Problem>())>;

    static constexpr auto config =
        BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr index_t GetVectorSizeA() { return Problem::VectorSizeA; }
    static constexpr index_t GetVectorSizeB() { return Problem::VectorSizeB; }
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t kLdsAlignmentInBytes = 16;

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
    static constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

    static constexpr index_t KFlatPerBlockPerIter = flatKPerWarp;
    static constexpr index_t NFlatPerBlockPerIter = flatNPerWarp;

    static constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
    static constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

    static constexpr index_t K1               = Problem::VectorLoadSize / sizeof(ADataType);
    static constexpr index_t ACopyLoadNum     = kMPerBlock * kKPerBlock / BlockSize / K1;
    static constexpr index_t ACopyLoadNumPerK = ACopyLoadNum / KIterPerWarp;
    static constexpr index_t ACopyPerLoadM    = kMPerBlock / ACopyLoadNum;
    static constexpr index_t BloadGap         = MIterPerWarp / 2;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;

    static constexpr auto warp_m = WarpTile::at(idxM);
    static constexpr auto warp_n = WarpTile::at(idxN);
    static constexpr auto warp_k = WarpTile::at(idxK);

    // MFMA configuration structure
    struct MfmaConfig
    {
        int mfma_per_wg;
        int dsread_per_wg;
    };
    static constexpr MfmaConfig GetMfmaConfig()
    {
        // K1 per Mfma = 0.5 cases: mfma_per_wg = 2, dsread_per_wg = 1
        if constexpr((warp_m == 16 && warp_n == 16 && warp_k == 32 &&
                      std::is_same_v<ADataType, fp8_t>) ||
                     (warp_m == 32 && warp_n == 32 && warp_k == 16 &&
                      std::is_same_v<ADataType, fp8_t>) ||
                     (warp_m == 16 && warp_n == 16 && warp_k == 16 &&
                      std::is_same_v<ADataType, fp16_t>) ||
                     (warp_m == 32 && warp_n == 32 && warp_k == 8 &&
                      std::is_same_v<ADataType, fp16_t>))
        {
            return {2, 1};
        }
        // K1 per Mfma = 2 cases: mfma_per_wg = 1, dsread_per_wg = 2
        else if constexpr((warp_m == 16 && warp_n == 16 && warp_k == 128 &&
                           std::is_same_v<ADataType, fp8_t>) ||
                          (warp_m == 32 && warp_n == 32 && warp_k == 64 &&
                           std::is_same_v<ADataType, fp8_t>))
        {
            return {1, 2};
        }
        // K1 per Mfma = 1 cases: mfma_per_wg = 1, dsread_per_wg = 1
        else if constexpr((warp_m == 16 && warp_n == 16 && warp_k == 32 &&
                           std::is_same_v<ADataType, fp16_t>) ||
                          (warp_m == 32 && warp_n == 32 && warp_k == 16 &&
                           std::is_same_v<ADataType, fp16_t>) ||
                          (warp_m == 16 && warp_n == 16 && warp_k == 128 /* &&
                           std::is_same_v<ADataType, fp4_t> */) ||
                          (warp_m == 32 && warp_n == 32 && warp_k == 64 /* &&
                           std::is_same_v<ADataType, fp4_t> */))
        {
            return {1, 1};
        }
        // Default configuration
        else
        {
            return {1, 1};
        }
    }

    // Get MFMA configuration values
    static constexpr auto mfma_config   = GetMfmaConfig();
    static constexpr auto mfma_per_wg   = mfma_config.mfma_per_wg;
    static constexpr auto dsread_per_wg = mfma_config.dsread_per_wg;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegBase", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    virtual ~FlatmmPipelineAGmemBGmemCRegBase() = default;

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return PipelinePolicy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto HotLoopScheduler()
    {

        constexpr index_t KPerLoad               = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t A_Buffer_Load_Inst_Num = kMPerBlock * kKPerBlock / BlockSize / KPerLoad;
        constexpr index_t A_LDS_Read_Inst_Num    = MIterPerWarp * KIterPerWarp;
        constexpr index_t B_Buffer_Load_Inst_Num = NIterPerWarp * KIterPerWarp;

        // Keypoint of pipeline optimize is workload balance in time
        // instruction schedule example(128X256X256, 1X4, 16X16X128):
        // Iter MNK     MFMA    ds_read ds_write    A_load  b_load
        // -1   M6N3:   60      2       -           -       -
        // -1   M7N0:   61      -       -           -       -
        // -1   M7N1:   62      -       -           -       -
        // -1   M7N2:   63      -       -           -       -
        // -1   M7N3:   64      4       -           -       -
        //  0   M0N0K0:  1      -       -           -       -
        //  0   M0N1:    2      -       -           -       2
        //  0   M0N2:    3      -       -           -       -
        //  0   M0N3:    4      6       -           -       -
        //  0   M1N0:    5      -       -           -       -
        //  0   M1N1:    6      -       -           -       4
        //  0   M1N2:    7      -       -           -       -
        //  0   M1N3:    8      8       -           -       -
        //  0   M2N0:    9      -       -           -       -
        //  0   M2N1:   10      -       -           -       6
        //  0   M2N2:   11      -       -           -       -
        //  0   M2N3:   12     10       -           -       -
        //  0   M3N0:   13      -       1           -       -
        //  0   M3N1:   14      -       -           -       8
        //  0   M3N2:   15      -       -           -       -
        //  0   M3N3:   16     12       -           -       -
        //  0   M4N0:   17      -       2           -       -
        //  0   M4N1:   18      -       -           -       -
        //  0   M4N2:   19      -       -           1       -
        //  0   M4N3:   20     14       -           -       -
        //  0   M5N0:   21      -       3           -       -
        //  0   M5N1:   22      -       -           -       -
        //  0   M5N2:   23      -       -           2       -
        //  0   M5N3:   24     16       -           -       -
        //  0   M6N0:   25      -       4           -       -
        //  0   M6N1:   26      -       -           -       -
        //  0   M6N2:   27      -       -           3       -
        //  0   M6N3:   28     17       -           -       -
        //  0   M7N0:   29      -       -           -       -
        //  0   M7N1:   30      -       -           -       -
        //  0   M7N2:   31      -       -           4       -
        //  0   M7N3:   32     18       -           -       -
        //  0   M0N0K1: 33      -       -           -       -
        //  0   M0N1:   34      -       -           -       10
        //  0   M0N2:   35      -       -           -       -
        //  0   M0N3:   36     20       -           -       -
        //  0   M1N0:   37      -       -           -       -
        //  0   M1N1:   38      -       -           -       12
        //  0   M1N2:   39      -       -           -       -
        //  0   M1N3:   40     22       -           -       -
        //  0   M2N0:   41      -       -           -       -
        //  0   M2N1:   42      -       -           -       14
        //  0   M2N2:   43      -       -           -       -
        //  0   M2N3:   44     24       -           -       -
        //  0   M3N0:   45      -       5           -       -
        //  0   M3N1:   46      -       -           -       16
        //  0   M3N2:   47      -       -           -       -
        //  0   M3N3:   48     26       -           -       -
        //  0   M4N0:   49      -       6           -       -
        //  0   M4N1:   50      -       -           -       -
        //  0   M4N2:   51      -       -           5       -
        //  0   M4N3:   52     28       -           -       -
        //  0   M5N0:   53      -       7           -       -
        //  0   M5N1:   54      -       -           -       -
        //  0   M5N2:   55      -       -           6       -
        //  0   M5N3:   56     30       -           -       -
        //  0   M6N0:   57      -       8           -       -
        //  0   M6N1:   58      -       -           -       -
        //  0   M6N2:   59      -       -           7       -
        //  0   M6N3:   60      2       -           -       -
        //  0   M7N0:   61      -       -           -       -
        //  0   M7N1:   62      -       -           -       -
        //  0   M7N2:   63      -       -           8       -
        //  0   M7N3:   64      4       -           -       -

        if constexpr(warp_m == 16 && warp_n == 16)
        {
// MFMA -> VMEM READ -> MFMA -> DS Read -> MFMA
// hiding the glbal memory VMEM latency
#if defined(__gfx950__)
            if constexpr(kMPerBlock == 128 && kNPerBlock == 256 && kKPerBlock == 256)
            {
                static_for<0, 2, 1>{}([&](auto j) {
                    ignore = j;
                    static_for<0, 3, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read

                    static_for<0, 3, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                });

                __builtin_amdgcn_sched_barrier(0);
            }
            else
            {
                static_for<0, 2, 1>{}([&](auto j) {
                    ignore = j;
                    static_for<0, 3, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read

                    static_for<0, 3, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    });
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                });

                __builtin_amdgcn_sched_barrier(0);
            }
// MFMA → MFMA → MFMA → MFMA → DS Read
// For other device engine we need more agressive MFMA with DS writes interleaved
#else
            if constexpr(kMPerBlock == 128 && kNPerBlock == 256 && kKPerBlock == 256)
            {
                static_for<0, 2, 1>{}([&](auto j) {
                    ignore = j;
                    // Uses loops to amortize scheduling overhead
                    static_for<0, 4, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });

                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
            }
            else if constexpr(kMPerBlock == 16 && kNPerBlock == 64 && kKPerBlock == 256)
            {
                static_for<0, 1, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                });
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read

                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read

                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read

                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_barrier(0);
            }
            else if constexpr(kMPerBlock == 128 && kNPerBlock == 128 && kKPerBlock == 128)
            {
                // prioritize MFMA to avoid LDS write conflicts
                static_for<0, 2, 1>{}([&](auto j) {
                    ignore = j;
                    static_for<0, 2, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 2, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                    static_for<0, 1, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
            }
            else
            {
                static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                static_for<0, A_LDS_Read_Inst_Num - A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
                });
                static_for<0, B_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                });
                static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
                });
            }

#endif
        }
        else
        {
            if constexpr((A_LDS_Read_Inst_Num / 2 >
                          A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num))
            {
                static_for<0,
                           A_LDS_Read_Inst_Num / 2 - A_Buffer_Load_Inst_Num -
                               B_Buffer_Load_Inst_Num,
                           1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
            }
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, A_LDS_Read_Inst_Num / 2, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, B_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
            });
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto TailHotLoopScheduler()
    {
#if 0
        static_for<0, 2, 1>{}([&](auto j) {
            ignore = j;
            static_for<0, 3, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            });
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read

            static_for<0, 3, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            });
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
        });
        __builtin_amdgcn_sched_barrier(0);
#endif
    }
};
} // namespace ck_tile
