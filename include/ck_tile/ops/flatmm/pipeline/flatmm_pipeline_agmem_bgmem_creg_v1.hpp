// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// #define FINEGRADE_LOADSTORE
#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {
template <typename Problem>
struct BaseFlatmmPipelineAGmemBGmemCRegV1
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

    static constexpr index_t K1               = 16 / sizeof(ADataType);
    static constexpr index_t ACopyLoadNum     = kMPerBlock * kKPerBlock / BlockSize / K1;
    static constexpr index_t ACopyLoadNumPerK = ACopyLoadNum / KIterPerWarp;
    static constexpr index_t AcopyPerLoadM    = kMPerBlock / ACopyLoadNum;
    static constexpr index_t BloadGap         = MIterPerWarp / 2;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;

    // MFMA configuration structure
    struct MfmaConfig
    {
        int mfma_per_wg;
        int dsread_per_wg;
    };
    static constexpr MfmaConfig GetMfmaConfig()
    {
        constexpr auto warp_m = WarpTile::at(idxM);
        constexpr auto warp_n = WarpTile::at(idxN);
        constexpr auto warp_k = WarpTile::at(idxK);

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

#if 0 // MI350 FP8 16X16 128*256*256
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
#endif
#if 0 // MI350 FP8 16X16 
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
#endif
#if 0 // MI300 FP8 16X16 128*128*128
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
#endif
#if 0 // MI300 FP8 16X16 128*256*128
            static_for<0, 2, 1>{}([&](auto j) {
                ignore = j;
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
#endif
#if 0 // MI300 FP8 16X16 16*64*256
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
#endif
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

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct FlatmmPipelineAGmemBGmemCRegV1
    : public FlatmmPipelineAGmemBGmemCRegBase<Problem, PipelinePolicy>
{
    using Base = FlatmmPipelineAGmemBGmemCRegBase<Problem, PipelinePolicy>;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::BlockGemmShape;
    using typename Base::CDataType;
    using typename Base::CLayout;

    using Base::config;
    using typename Base::BlockFlatmm;
    using typename Base::WG;

    using Base::GetVectorSizeA;
    using Base::GetVectorSizeB;
    using Base::GetVectorSizeC;
    using Base::kLdsAlignmentInBytes;
    using Base::kPadK;
    using Base::kPadM;
    using Base::kPadN;

    using Base::BlockSize;
    using Base::flatKPerWarp;
    using Base::flatNPerWarp;
    using Base::kKPerBlock;
    using Base::kMPerBlock;
    using Base::kNPerBlock;
    using Base::TransposeC;

    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::idxK;
    using Base::idxM;
    using Base::idxN;
    using typename Base::BlockTile;
    using typename Base::BlockWarps;
    using typename Base::WarpTile;

    using Base::KFlatPerBlockPerIter;
    using Base::KIterPerWarp;
    using Base::MIterPerWarp;
    using Base::NFlatPerBlockPerIter;
    using Base::NIterPerWarp;

    using Base::ACopyLoadNum;
    using Base::ACopyLoadNumPerK;
    using Base::AcopyPerLoadM;
    using Base::BloadGap;

    using Base::HasHotLoop;
    using Base::TailNum;

    using Base::MWarp;
    using Base::NWarp;

    using Base::KPerBlockPerIter;
    using Base::MPerBlockPerIter;

    using Base::K1;

    using Base::dsread_per_wg;
    using Base::mfma_per_wg;
    using typename Base::MfmaConfig;

    using Base::DoubleSmemBuffer;
    using Base::GetMfmaConfig;

    using Base::GetSmemSize;
    using Base::HotLoopScheduler;
    using Base::TailHotLoopScheduler;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV1", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp, typename AElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        index_t num_loop,
                                        void* p_smem_ping,
                                        void* p_smem_pong) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");
        static_assert(kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto MIter_2nd_last = (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;
        const index_t iMWarp          = get_warp_id() / NWarp;

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        __builtin_amdgcn_sched_barrier(0);

        // A tile in LDS
        ADataType* p_a_lds_ping = static_cast<ADataType*>(p_smem_ping);
        ADataType* p_a_lds_pong = static_cast<ADataType*>(p_smem_pong);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_lds_block_desc);
        auto a_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        auto a_copy_lds_window_ping =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        auto a_copy_lds_window_pong =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        // A LDS tile for block GEMM
        // auto a_lds_gemm_window = make_tile_window(
        //     a_lds_block_ping, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // ping-pong window for A LDS
        auto a_warp_window_ping_tmp =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        auto a_warp_window_pong_tmp =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_ping_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_ping;

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_pong_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_pong;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_ping(mIter)(kIter) = a_warp_window_ping_tmp;

                move_tile_window(a_warp_windows_ping(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_pong(mIter)(kIter) = a_warp_window_pong_tmp;

                move_tile_window(a_warp_windows_pong(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Block GEMM
        auto block_flatmm = BlockFlatmm();
        // Acc register tile
        auto c_block_tile = block_flatmm.MakeCBlockTile();

        // B flat DRAM window for load
        auto b_flat_distribution =
            PipelinePolicy::template MakeBFlatDramTileDistribution<Problem>();
        auto b_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        // pingpong buffer for B
        statically_indexed_array<
            statically_indexed_array<decltype(b_flat_dram_window), KIterPerWarp>,
            NIterPerWarp>
            b_flat_dram_windows;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_ping;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_pong;

        // Prefetch A0
        auto a_block_tile = load_tile(a_copy_dram_window);
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        // prefetch B
        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                 {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                b_warp_tensor_ping(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
            });
        });
        // move B window to next flat K
        move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

        // Prefill A0
        // if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
        // {
        //     auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
        //         PipelinePolicy::template MakeShuffledARegBlockDistribution<Problem>());
        //     shuffle_tile(a_shuffle_tmp, a_block_tile);
        //     const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_shuffle_tmp);
        //     store_tile(a_copy_lds_window_ping, a_block_tile_tmp);
        // }
        // else
        // {
        //     store_tile(a_copy_lds_window_ping, tile_elementwise_in(a_element_func,
        //     a_block_tile));
        // }
        auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
        store_tile(a_copy_lds_window_ping, a_block_tile_tmp);

        __builtin_amdgcn_sched_barrier(0);

        // Prefetch A1
        a_block_tile = load_tile(a_copy_dram_window);
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        // initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        block_sync_lds();

        // preload A00,A10 from lds
        constexpr auto m_preload = (MIterPerWarp * KIterPerWarp >= 2) ? 2 : 1;
        statically_indexed_array<decltype(load_tile(a_warp_windows_ping(number<0>{})(number<0>{}))),
                                 m_preload>
            a_warp_tensor_ping;
        statically_indexed_array<decltype(load_tile(a_warp_windows_pong(number<0>{})(number<0>{}))),
                                 m_preload>
            a_warp_tensor_pong;

        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MIterPerWarp;
            constexpr auto kIter = loadIter / MIterPerWarp;
            a_warp_tensor_ping(loadIter) =
                load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
        });
        __builtin_amdgcn_sched_barrier(0);

        // if(threadIdx.x==0){
        //     for(int i=0;i<a_block_tile.get_thread_buffer_size();i++) {
        //         printf("dteng--A buffer load: idx.x=%u, ablocktile=%f, buffer size=%d\n",
        //         threadIdx.x,
        //         type_convert<float>(a_block_tile.thread_buf_(i)),a_block_tile.get_thread_buffer_size());
        //     }
        // }
        // for(int i=0;i<a_warp_tensor_ping(number<0>{}).get_thread_buffer_size();i++) {
        //     printf("dteng--A lds load 00: idx.x=%u, awarptensor=%f, buffer size=%d\n",
        //     threadIdx.x,
        //     type_convert<float>(a_warp_tensor_ping(number<0>{}).thread_buf_(i)),a_warp_tensor_ping(number<0>{}).get_thread_buffer_size());
        // }

        index_t iCounter = (num_loop - 1) / 2;
        // if constexpr(HasMainLoop)
        // {
        while(iCounter > 0)
        {
            // prefetch B(2i+1)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_pong(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // Prefill A(2i+1)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_pong, a_block_tile_tmp);

            // Prefetch A(2i+2)
            a_block_tile = load_tile(a_copy_dram_window);
            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // GEMM 2i
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_ping(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_ping(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_ping);

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor_pong(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });
            HotLoopScheduler();

            // Next K

            // prefetch B(2i+2)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_ping(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // Prefill A(2i+2)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_ping, a_block_tile_tmp);

            // Prefetch A(2i+3)
            a_block_tile = load_tile(a_copy_dram_window);
            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // GEMM 2i+1
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_pong(number<AwarpIter>{}),
                             b_warp_tensor_pong(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_pong(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_pong(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_pong);

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor_ping(loadIter) =
                    load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
            });
            HotLoopScheduler();

            iCounter--;
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            // __builtin_amdgcn_sched_barrier(0);
            // prefetch B(loopK)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_pong(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // Prefill A(loopK)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_pong, a_block_tile_tmp);

            // GEMM loopK-1
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_ping(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_ping(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_ping);

            TailHotLoopScheduler();

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor_pong(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });

            // __builtin_amdgcn_sched_barrier(0);

            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_pong(number<AwarpIter>{}),
                             b_warp_tensor_pong(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_pong(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_pong(number<AmIter>{})(number<AkIter>{}));
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_pong);

            // TailHotLoopScheduler();
            // __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_ping(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_ping(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
        }
        // }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_flat_dram_block_window_tmp,
            num_loop,
            p_smem_ping,
            p_smem_pong);
    }
};

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct FlatmmPipelineAGmemBGmemCRegV2
    : public FlatmmPipelineAGmemBGmemCRegBase<Problem, PipelinePolicy>
{

    using Base = FlatmmPipelineAGmemBGmemCRegBase<Problem, PipelinePolicy>;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::BlockGemmShape;
    using typename Base::CDataType;
    using typename Base::CLayout;

    using Base::config;
    using typename Base::BlockFlatmm;
    using typename Base::WG;

    using Base::GetVectorSizeA;
    using Base::GetVectorSizeB;
    using Base::GetVectorSizeC;
    using Base::kLdsAlignmentInBytes;
    using Base::kPadK;
    using Base::kPadM;
    using Base::kPadN;

    using Base::BlockSize;
    using Base::flatKPerWarp;
    using Base::flatNPerWarp;
    using Base::kKPerBlock;
    using Base::kMPerBlock;
    using Base::kNPerBlock;
    using Base::TransposeC;

    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::idxK;
    using Base::idxM;
    using Base::idxN;
    using typename Base::BlockTile;
    using typename Base::BlockWarps;
    using typename Base::WarpTile;

    using Base::KFlatPerBlockPerIter;
    using Base::KIterPerWarp;
    using Base::MIterPerWarp;
    using Base::NFlatPerBlockPerIter;
    using Base::NIterPerWarp;

    using Base::ACopyLoadNum;
    using Base::ACopyLoadNumPerK;
    using Base::AcopyPerLoadM;
    using Base::BloadGap;

    using Base::HasHotLoop;
    using Base::TailNum;

    using Base::MWarp;
    using Base::NWarp;

    using Base::KPerBlockPerIter;
    using Base::MPerBlockPerIter;

    using Base::K1;

    using Base::dsread_per_wg;
    using Base::mfma_per_wg;
    using typename Base::MfmaConfig;

    using Base::DoubleSmemBuffer;
    using Base::GetMfmaConfig;

    using Base::GetSmemSize;
    using Base::HotLoopScheduler;
    using Base::TailHotLoopScheduler;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV2", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp, typename AElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        index_t num_loop,
                                        void* p_smem_ping,
                                        void* p_smem_pong) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");
        static_assert(kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto MIter_2nd_last = (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;
        const index_t iMWarp          = get_warp_id() / NWarp;

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        __builtin_amdgcn_sched_barrier(0);

        // A tile in LDS
        ADataType* p_a_lds_ping = static_cast<ADataType*>(p_smem_ping);
        ADataType* p_a_lds_pong = static_cast<ADataType*>(p_smem_pong);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_lds_block_desc);
        auto a_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_lds_block_desc);

        // A DRAM tile window for load

        auto a_copy_dram_window_tmp =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<AcopyPerLoadM>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeADramDistribution<Problem>());

        statically_indexed_array<decltype(a_copy_dram_window_tmp), ACopyLoadNum> a_copy_dram_window;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_copy_dram_window(AIter) = a_copy_dram_window_tmp;
            move_tile_window(a_copy_dram_window(AIter), {AIter * AcopyPerLoadM, 0});
        });

        auto a_copy_lds_window_ping_tmp =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<AcopyPerLoadM>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramDistribution<Problem>());

        statically_indexed_array<decltype(a_copy_lds_window_ping_tmp), ACopyLoadNum>
            a_copy_lds_window_ping;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_copy_lds_window_ping(AIter) = a_copy_lds_window_ping_tmp;
            move_tile_window(a_copy_lds_window_ping(AIter), {AIter * AcopyPerLoadM, 0});
        });

        auto a_copy_lds_window_pong_tmp =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<AcopyPerLoadM>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramDistribution<Problem>());

        statically_indexed_array<decltype(a_copy_lds_window_pong_tmp), ACopyLoadNum>
            a_copy_lds_window_pong;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_copy_lds_window_pong(AIter) = a_copy_lds_window_pong_tmp;
            move_tile_window(a_copy_lds_window_pong(AIter), {AIter * AcopyPerLoadM, 0});
        });

        // A LDS tile for block GEMM
        // auto a_lds_gemm_window = make_tile_window(
        //     a_lds_block_ping, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // ping-pong window for A LDS
        auto a_warp_window_ping_tmp =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        auto a_warp_window_pong_tmp =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_ping_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_ping;

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_pong_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_pong;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_ping(mIter)(kIter) = a_warp_window_ping_tmp;

                move_tile_window(a_warp_windows_ping(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_pong(mIter)(kIter) = a_warp_window_pong_tmp;

                move_tile_window(a_warp_windows_pong(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Block GEMM
        auto block_flatmm = BlockFlatmm();
        // Acc register tile
        auto c_block_tile = block_flatmm.MakeCBlockTile();

        // B flat DRAM window for load
        auto b_flat_distribution =
            PipelinePolicy::template MakeBFlatDramTileDistribution<Problem>();
        auto b_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        // pingpong buffer for B
        statically_indexed_array<
            statically_indexed_array<decltype(b_flat_dram_window), KIterPerWarp>,
            NIterPerWarp>
            b_flat_dram_windows;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_ping;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_pong;

        // Prefetch A0

        statically_indexed_array<decltype(load_tile(a_copy_dram_window(number<0>{}))), ACopyLoadNum>
            a_block_tile;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_block_tile(AIter) = load_tile(a_copy_dram_window(AIter));
            move_tile_window(a_copy_dram_window(AIter), {0, kKPerBlock});
        });

        // prefetch B
        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                 {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                b_warp_tensor_ping(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
            });
        });
        // move B window to next flat K
        move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

        // Prefill A0
        // if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
        // {
        //     auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
        //         PipelinePolicy::template MakeShuffledARegBlockDistribution<Problem>());
        //     shuffle_tile(a_shuffle_tmp, a_block_tile);
        //     const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_shuffle_tmp);
        //     store_tile(a_copy_lds_window_ping, a_block_tile_tmp);
        // }
        // else
        // {
        //     store_tile(a_copy_lds_window_ping, tile_elementwise_in(a_element_func,
        //     a_block_tile));
        // }

        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            store_tile(a_copy_lds_window_ping(AIter),
                       tile_elementwise_in(a_element_func, a_block_tile(AIter)));
        });

        __builtin_amdgcn_sched_barrier(0);

        // Prefetch A1

        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_block_tile(AIter) = load_tile(a_copy_dram_window(AIter));
            move_tile_window(a_copy_dram_window(AIter), {0, kKPerBlock});
        });

        // initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        block_sync_lds();

        // preload A00,A10 from lds
        constexpr auto m_preload = (MIterPerWarp * KIterPerWarp >= 2) ? 2 : 1;
        statically_indexed_array<decltype(load_tile(a_warp_windows_ping(number<0>{})(number<0>{}))),
                                 m_preload>
            a_warp_tensor_ping;
        statically_indexed_array<decltype(load_tile(a_warp_windows_pong(number<0>{})(number<0>{}))),
                                 m_preload>
            a_warp_tensor_pong;

        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MIterPerWarp;
            constexpr auto kIter = loadIter / MIterPerWarp;
            a_warp_tensor_ping(loadIter) =
                load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
        });
        __builtin_amdgcn_sched_barrier(0);

        // if(threadIdx.x==0){
        //     for(int i=0;i<a_block_tile.get_thread_buffer_size();i++) {
        //         printf("dteng--A buffer load: idx.x=%u, ablocktile=%f, buffer size=%d\n",
        //         threadIdx.x,
        //         type_convert<float>(a_block_tile.thread_buf_(i)),a_block_tile.get_thread_buffer_size());
        //     }
        // }
        // for(int i=0;i<a_warp_tensor_ping(number<0>{}).get_thread_buffer_size();i++) {
        //     printf("dteng--A lds load 00: idx.x=%u, awarptensor=%f, buffer size=%d\n",
        //     threadIdx.x,
        //     type_convert<float>(a_warp_tensor_ping(number<0>{}).thread_buf_(i)),a_warp_tensor_ping(number<0>{}).get_thread_buffer_size());
        // }

        index_t iCounter = (num_loop - 1) / 2;
        // if constexpr(HasMainLoop)
        // {
        while(iCounter > 0)
        {

            // GEMM 2i
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_ping(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        // prefetch B(2i+1)
                        constexpr auto curMNIter = mIter * NIterPerWarp + nIter;
                        if constexpr((curMNIter < NIterPerWarp * BloadGap) &&
                                     ((curMNIter % BloadGap) == 1))
                        {
                            constexpr auto BnIter                         = curMNIter / BloadGap;
                            constexpr auto BkIter                         = kIter;
                            b_flat_dram_windows(number<BnIter>{})(BkIter) = b_flat_dram_window;
                            move_tile_window(
                                b_flat_dram_windows(number<BnIter>{})(BkIter),
                                {BnIter * NFlatPerBlockPerIter, BkIter * KFlatPerBlockPerIter});
                            b_warp_tensor_pong(number<BnIter>{})(BkIter) =
                                load_tile(b_flat_dram_windows(number<BnIter>{})(BkIter));
                        }
                        // Prefill A(2i+1)
                        if constexpr((mIter >= (MIterPerWarp - 1 - ACopyLoadNumPerK)) &&
                                     (mIter < (MIterPerWarp - 1)) && ((nIter % NIterPerWarp) == 0))
                        {
                            constexpr auto AIter =
                                (mIter + ACopyLoadNumPerK + 1 + kIter * ACopyLoadNumPerK) %
                                ACopyLoadNum;
                            store_tile(
                                a_copy_lds_window_pong(number<AIter>{}),
                                tile_elementwise_in(a_element_func, a_block_tile(number<AIter>{})));
                        }
                        // Prefetch A(2i+2)
                        if constexpr((mIter >= (MIterPerWarp - 1 - ACopyLoadNumPerK + 1)) &&
                                     (mIter < (MIterPerWarp - 1 + 1)) &&
                                     ((nIter % NIterPerWarp) == (NIterPerWarp - 2)))
                        {
                            constexpr auto AIter =
                                (mIter + ACopyLoadNumPerK + kIter * ACopyLoadNumPerK) %
                                ACopyLoadNum;
                            a_block_tile(number<AIter>{}) =
                                load_tile(a_copy_dram_window(number<AIter>{}));
                            move_tile_window(a_copy_dram_window(number<AIter>{}), {0, kKPerBlock});
                        }
                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_ping(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_ping);

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor_pong(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });
            HotLoopScheduler();

            // Next K

            // GEMM 2i+1
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_pong(number<AwarpIter>{}),
                             b_warp_tensor_pong(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        // prefetch B(2i+2)
                        constexpr auto curMNIter = mIter * NIterPerWarp + nIter;
                        if constexpr((curMNIter < NIterPerWarp * BloadGap) &&
                                     ((curMNIter % BloadGap) == 1))
                        {
                            constexpr auto BnIter                         = curMNIter / BloadGap;
                            constexpr auto BkIter                         = kIter;
                            b_flat_dram_windows(number<BnIter>{})(BkIter) = b_flat_dram_window;
                            move_tile_window(
                                b_flat_dram_windows(number<BnIter>{})(BkIter),
                                {BnIter * NFlatPerBlockPerIter, BkIter * KFlatPerBlockPerIter});
                            b_warp_tensor_ping(number<BnIter>{})(BkIter) =
                                load_tile(b_flat_dram_windows(number<BnIter>{})(BkIter));
                        }
                        // Prefill A(2i+1)
                        if constexpr((mIter >= (MIterPerWarp - 1 - ACopyLoadNumPerK)) &&
                                     (mIter < (MIterPerWarp - 1)) && ((nIter % NIterPerWarp) == 0))
                        {
                            constexpr auto AIter =
                                (mIter + ACopyLoadNumPerK + 1 + kIter * ACopyLoadNumPerK) %
                                ACopyLoadNum;
                            store_tile(
                                a_copy_lds_window_ping(number<AIter>{}),
                                tile_elementwise_in(a_element_func, a_block_tile(number<AIter>{})));
                        }
                        // Prefetch A(2i+2)
                        if constexpr((mIter >= (MIterPerWarp - 1 - ACopyLoadNumPerK + 1)) &&
                                     (mIter < (MIterPerWarp - 1 + 1)) &&
                                     ((nIter % NIterPerWarp) == (NIterPerWarp - 2)))
                        {
                            constexpr auto AIter =
                                (mIter + ACopyLoadNumPerK + kIter * ACopyLoadNumPerK) %
                                ACopyLoadNum;
                            a_block_tile(number<AIter>{}) =
                                load_tile(a_copy_dram_window(number<AIter>{}));
                            move_tile_window(a_copy_dram_window(number<AIter>{}), {0, kKPerBlock});
                        }
                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_pong(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_pong(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_pong);

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor_ping(loadIter) =
                    load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
            });
            HotLoopScheduler();

            iCounter--;
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            // __builtin_amdgcn_sched_barrier(0);

            // GEMM loopK-1
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_ping(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        // prefetch B(loopK)
                        constexpr auto curMNIter = mIter * NIterPerWarp + nIter;
                        if constexpr((curMNIter < NIterPerWarp * BloadGap) &&
                                     ((curMNIter % BloadGap) == 1))
                        {
                            constexpr auto BnIter                         = curMNIter / BloadGap;
                            constexpr auto BkIter                         = kIter;
                            b_flat_dram_windows(number<BnIter>{})(BkIter) = b_flat_dram_window;
                            move_tile_window(
                                b_flat_dram_windows(number<BnIter>{})(BkIter),
                                {BnIter * NFlatPerBlockPerIter, BkIter * KFlatPerBlockPerIter});
                            b_warp_tensor_pong(number<BnIter>{})(BkIter) =
                                load_tile(b_flat_dram_windows(number<BnIter>{})(BkIter));
                        }
                        // Prefill A(loopK)
                        if constexpr((mIter >= (MIterPerWarp - 1 - ACopyLoadNumPerK)) &&
                                     (mIter < (MIterPerWarp - 1)) && ((nIter % NIterPerWarp) == 0))
                        {
                            constexpr auto AIter =
                                (mIter + ACopyLoadNumPerK + 1 + kIter * ACopyLoadNumPerK) %
                                ACopyLoadNum;
                            store_tile(
                                a_copy_lds_window_pong(number<AIter>{}),
                                tile_elementwise_in(a_element_func, a_block_tile(number<AIter>{})));
                        }

                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_ping(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_ping);

            TailHotLoopScheduler();

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor_pong(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });

            // __builtin_amdgcn_sched_barrier(0);

            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_pong(number<AwarpIter>{}),
                             b_warp_tensor_pong(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_pong(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_pong(number<AmIter>{})(number<AkIter>{}));
                    }
                });
            });
            // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_pong);

            // TailHotLoopScheduler();
            // __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor_ping(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());

                        // prefetch B(loopK)
                        constexpr auto curMNIter = mIter * NIterPerWarp + nIter;
                        if constexpr((curMNIter < NIterPerWarp * BloadGap) &&
                                     ((curMNIter % BloadGap) == 1))
                        {
                            constexpr auto BnIter                         = curMNIter / BloadGap;
                            constexpr auto BkIter                         = kIter;
                            b_flat_dram_windows(number<BnIter>{})(BkIter) = b_flat_dram_window;
                            move_tile_window(
                                b_flat_dram_windows(number<BnIter>{})(BkIter),
                                {BnIter * NFlatPerBlockPerIter, BkIter * KFlatPerBlockPerIter});
                            b_warp_tensor_pong(number<BnIter>{})(BkIter) =
                                load_tile(b_flat_dram_windows(number<BnIter>{})(BkIter));
                        }
                        // Prefill A(loopK)
                        if constexpr((mIter >= (MIterPerWarp - 1 - ACopyLoadNumPerK)) &&
                                     (mIter < (MIterPerWarp - 1)) && ((nIter % NIterPerWarp) == 0))
                        {
                            constexpr auto AIter =
                                (mIter + ACopyLoadNumPerK + 1 + kIter * ACopyLoadNumPerK) %
                                ACopyLoadNum;
                            store_tile(
                                a_copy_lds_window_pong(number<AIter>{}),
                                tile_elementwise_in(a_element_func, a_block_tile(number<AIter>{})));
                        }

                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor_ping(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
        }
        // }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_flat_dram_block_window_tmp,
            num_loop,
            p_smem_ping,
            p_smem_pong);
    }
};

} // namespace ck_tile
