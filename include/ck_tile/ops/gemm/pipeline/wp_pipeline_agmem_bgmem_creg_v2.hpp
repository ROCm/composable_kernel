// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseWeightPreshufflePipelineAGmemBGmemCRegV2
{
    static constexpr index_t PrefetchStages   = 2;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool, TailNumber tail_number)
    {
        if(tail_number == TailNumber::Odd)
        {
            run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Odd>{});
        }
        else if(tail_number == TailNumber::Even)
        {
            run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Even>{});
        }
    }
};

template <typename Problem, typename PipelinePolicy = UniversalWeightPreshufflePipelineAgBgCrPolicy>
struct WeightPreshufflePipelineAGmemBGmemCRegV2
    : public BaseWeightPreshufflePipelineAGmemBGmemCRegV2<Problem>
{
    using Base = BaseWeightPreshufflePipelineAGmemBGmemCRegV2<Problem>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockWeightPreshuffle =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockWeightPreshuffle<Problem>())>;

    static constexpr auto config =
        BlockWeightPreshuffle::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr index_t GetVectorSizeA()
    {
        return PipelinePolicy::template GetVectorSizeA<Problem>();
    }
    static constexpr index_t GetVectorSizeB()
    {
        return PipelinePolicy::template GetVectorSizeB<Problem>();
    }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t kLdsAlignmentInBytes = 16;
    static constexpr index_t NumWaveGroups        = Problem::NumWaveGroups;

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

    static constexpr index_t K1           = Problem::VectorLoadSize / sizeof(ADataType);
    static constexpr index_t ACopyLoadNum = kMPerBlock * kKPerBlock / BlockSize / K1;
    static constexpr auto TailNum         = Problem::TailNum;

    static constexpr auto warp_m = WarpTile::at(idxM);
    static constexpr auto warp_n = WarpTile::at(idxN);
    static constexpr auto warp_k = WarpTile::at(idxK);

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV2", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', WG::kM, WG::kN, WG::kK),
                      concat('x', GetVectorSizeA(), GetVectorSizeB()),
                      concat('x', kPadM, kPadN, kPadK));

        // clang-format on
    }

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;
    using Base::UsePersistentKernel;

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
// For other device engine we need more aggressive MFMA with DS writes interleaved
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

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp, typename AElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        index_t num_loop,
                                        void* p_smem_ping,
                                        void* p_smem_pong) const
    {
        using ComputeDataType = float;
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
        auto block_weight_preshuffle = BlockWeightPreshuffle();
        // Acc register tile
        auto c_block_tile = block_weight_preshuffle.MakeCBlockTile();

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

        index_t iCounter = (num_loop - 1) / 2;
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
            // TailHotLoopScheduler();

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
            // TailHotLoopScheduler();
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
        // apply softmax for c_block_tile
        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // m_local = rowmax(c_block_tile)
        auto m_local = block_tile_reduce<ComputeDataType>(
            c_block_tile, sequence<1>{}, f_max, std::numeric_limits<ComputeDataType>::lowest());

        block_tile_reduce_sync(m_local, f_max);

        // Pcompute{j} = sum(exp(x - m_local))
        auto p_compute =
            make_static_distributed_tensor<ComputeDataType>(c_block_tile.get_tile_distribution());

        constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();

        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                p_compute(i_j_idx) = exp(c_block_tile[i_j_idx] - m_local[i_idx]);
            });
        });

        // rowsum for p_compute{i, j}
        auto rowsum_p = block_tile_reduce<ComputeDataType>(
            p_compute, sequence<1>{}, f_sum, ComputeDataType{0});

        block_tile_reduce_sync(rowsum_p, f_sum);

        // softmax = p_compute{i, j} / rowsum_p
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                p_compute(i_j_idx) = p_compute[i_j_idx] / rowsum_p[i_idx];
            });
        });

        // -------------------------------------------------------------------------------------
        // Grouped topk part
        // for topk computing
        using WeightType = float;
        using IndexType = int32_t;
        struct ArgmaxPacket
        {
            WeightType value;
            IndexType arg;
        };

        int num_expert_group = 16;
        int topk_group = 2;
        int topk = 8;

        // auto value_block_tile =
        //     make_static_distributed_tensor<WeightType>(p_compute.get_tile_distribution());
        auto index_block_tile =
            make_static_distributed_tensor<WeightType>(p_compute.get_tile_distribution());

        auto x_tmp = p_compute;
        // Step1. calculate group score
        // int num_expert_group = 16;
        // int topk_group = 2;
        int expert_per_group = kNPerBlock / num_expert_group;
        constexpr auto p_compute_spans = decltype(p_compute)::get_distributed_spans();
        auto group_scores = x_tmp;
        // init group_scores to inf
        sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                group_scores(i_j_idx) = -numeric<WeightType>::infinity();
            });
        });
        for (index_t n_group = 0; n_group < num_expert_group; n_group++) {
            // get group value matrix (masked other groups)
            auto group_tmp = x_tmp;
            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        group_tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    group_tmp(i_j_idx) = ((col_id >= (n_group * expert_per_group)) && (col_id < ((n_group + 1) * expert_per_group))) ? x_tmp(i_j_idx) : -numeric<WeightType>::infinity();
                });
            });
            // get one column for group scores = rowmax(group_tmp)
            auto group_scores_col = block_tile_reduce<ComputeDataType>(
                group_tmp, sequence<1>{}, f_max, std::numeric_limits<ComputeDataType>::lowest());

            block_tile_reduce_sync(group_scores_col, f_max);
            // get all group scores
            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    group_scores(i_j_idx) = (col_id == n_group) ? group_scores_col(i_idx): group_scores(i_j_idx);
                });
            });
        }
                // // To Do - perf opt scheme to reshape x_tmp from 2d to 3d before reducemax
        // const auto x_tmp_3d = x_tmp.block_tile_reshape((kM, num_expert_group, kN/num_expert_group));
        // auto group_scores = block_tile_reduce<ComputeDataType>(
        //     x_tmp_3d, sequence<2>{}, f_max, std::numeric_limits<ComputeDataType>::lowest());
        // block_tile_reduce_sync(group_scores, f_max);

        // Step2: select topk group and cal mask score matrix
        // argmax for topk
        const auto f_argmax = [](ArgmaxPacket e0, ArgmaxPacket e1) {
            return e0.value > e1.value ? e0 : e1;
        };

        // topk_group_mask(1 for selected group scores, -inf for other group scores)
        auto topk_group_scores_mask = x_tmp;
        // init topk_group_scores_mask to -inf
        sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                topk_group_scores_mask(i_j_idx) = -numeric<WeightType>::infinity();
            });
        });

        for(index_t k_group = 0; k_group < topk_group; k_group++)
        {
            auto group_packet            = [&]() {
                auto tmp = make_static_distributed_tensor<ArgmaxPacket>(p_compute.get_tile_distribution());
                sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        ArgmaxPacket t;
                        t.value    = group_scores(i_j_idx);
                        t.arg      = tile_idx.at(number<1>{});
                        tmp(i_j_idx) = t;
                    });
                });
                return tmp;
            }();

            auto argmax_init = ArgmaxPacket{-numeric<WeightType>::infinity(), 0};
            auto group_r = block_tile_reduce<ArgmaxPacket>(group_packet, sequence<1>{}, f_argmax, argmax_init);

            block_tile_reduce_xor_sync(group_r, f_argmax);

            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    // auto row_id = tile_idx.at(number<0>{});
                    auto col_id = tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    auto k_group_idx       = group_r(i_idx).arg;
                    topk_group_scores_mask(i_j_idx) = ((col_id >= (k_group_idx * expert_per_group)) && (col_id < ((k_group_idx + 1) * expert_per_group))) ? 1 : topk_group_scores_mask(i_j_idx);          
                });
            });

            // update value
            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});

                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    group_scores(i_j_idx) = (col_id == group_r(i_idx).arg) ? -numeric<WeightType>::infinity()
                                                                    : group_scores(i_j_idx);
                });
            });
        }

        // Step3: mask score matrix
        auto x_tmp_masked = x_tmp;
        sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                x_tmp_masked(i_j_idx) = x_tmp(i_j_idx) * topk_group_scores_mask(i_j_idx);
            });
        });

        // Step4: select topk values from masked score matrix
        for(index_t i_k = 0; i_k < topk; i_k++)
        {
            auto packet            = [&]() {
                auto tmp = make_static_distributed_tensor<ArgmaxPacket>(p_compute.get_tile_distribution());
                sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        ArgmaxPacket t;
                        t.value    = x_tmp_masked(i_j_idx); // !!! we reference p_compute here
                        t.arg      = tile_idx.at(number<1>{});
                        tmp(i_j_idx) = t;
                    });
                });
                return tmp;
            }();

            auto argmax_init = ArgmaxPacket{-numeric<WeightType>::infinity(), 0};
            auto r = block_tile_reduce<ArgmaxPacket>(packet, sequence<1>{}, f_argmax, argmax_init);

            block_tile_reduce_xor_sync(r, f_argmax);

            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    // auto row_id = tile_idx.at(number<0>{});
                    auto col_id = tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ArgmaxPacket tmp       = r(i_idx);
                    // value_block_tile(i_j_idx)                = (col_id == i_k) ? tmp.value: value_block_tile(i_j_idx);
                    index_block_tile(i_j_idx)                = (col_id == i_k) ? tmp.arg: index_block_tile(i_j_idx);                 
                });
            });

            // update value
            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});

                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    x_tmp_masked(i_j_idx) = (col_id == r(i_idx).arg) ? -numeric<WeightType>::infinity()
                                                                    : x_tmp_masked(i_j_idx);
                });
            });
        }
        // cast DataType and apply CElementFunction
        // const auto value_cast_block_tile = tile_elementwise_in(
        //     [&](const auto& value) { return type_convert<WeightType>(value); },
        //     value_block_tile);
        // const auto index_cast_block_tile = tile_elementwise_in(
        //     [&](const auto& index) { return type_convert<IndexType>(index); },
        //     index_block_tile);

        // return value_cast_block_tile;
        return index_block_tile;
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
            [](const ADataType & a) { return a; },
            b_flat_dram_block_window_tmp,
            num_loop,
            p_smem_ping,
            p_smem_pong);
    }
};

} // namespace ck_tile
