
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseFlatmmPipelineAGmemBGmemCRegV0
{
    static constexpr index_t PrefetchStages  = 2;

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto TailHandler(const RunFunction& run_func, bool, TailNumber tail_num)
    {
        if (TailNumber::Even == tail_num) 
        {
            return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Even>{});
        }
        else if (TailNumber::Odd == tail_num)
        {
            return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Odd>{});
        }
        // assert(false);
        return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Empty>{});
        // return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Empty>{});
    }
};

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct FlatmmPipelineAGmemBGmemCRegV0
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
    
    static constexpr auto config = BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

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

    static constexpr index_t K1 = 16 / sizeof(ADataType);
    static constexpr index_t ACopyLoadNum = kMPerBlock * kKPerBlock / BlockSize / K1;
    static constexpr index_t ACopyLoadNumPerK = ACopyLoadNum / KIterPerWarp;
    static constexpr index_t AcopyPerLoadM = kMPerBlock / ACopyLoadNum;
    static constexpr index_t BloadGap = MIterPerWarp / 2;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;

    static constexpr auto warp_m = WarpTile::at(idxM);
    static constexpr auto warp_n = WarpTile::at(idxN);
    static constexpr auto warp_k = WarpTile::at(idxK);

    /*
    defined(USING_MFMA_16x16x32) && defined(ENABLE_FP8) // mi300 fp8 16c 0.5*K1
    defined(USING_MFMA_32x32x16) && defined(ENABLE_FP8) // mi300 fp8 32c 0.5*K1
    defined(USING_MFMA_16x16x16) && defined(ENABLE_FP16) // mi300 fp16 16c 0.5*K1
    defined(USING_MFMA_32x32x8) && defined(ENABLE_FP16) // mi300 fp16 32c 0.5*K1

    defined(USING_MFMA_16x16x128) && defined(ENABLE_FP8) // mi350 fp8 32c 2*K1
    defined(USING_MFMA_32x32x64) && defined(ENABLE_FP8) // mi350 fp8 64c 2*K1
    defined(USING_MFMA_16x16x32) && defined(ENABLE_FP16) // mi350 fp16 16c 1*K1
    defined(USING_MFMA_32x32x16) && defined(ENABLE_FP16) // mi350 fp16 32c 1*K1

    defined(USING_MFMA_16x16x128) && defined(ENABLE_FP4) // mi350 fp4 16c 1*K1
    defined(USING_MFMA_32x32x64) && defined(ENABLE_FP4) // mi350 fp4 32c 1*K1
    */
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
                            (warp_m == 16 && warp_n == 16 && warp_k == 128 /*&&
                            std::is_same_v<ADataType, fp4_t> */) ||
                            (warp_m == 32 && warp_n == 32 && warp_k == 64  /*&&
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

    static constexpr auto mfma_config   = GetMfmaConfig();
    static constexpr auto mfma_per_wg   = mfma_config.mfma_per_wg;
    static constexpr auto dsread_per_wg = mfma_config.dsread_per_wg;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV1", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

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
        // -1   M6N0:   57      -       8           -       -
        // -1   M6N1:   58      1       -           -       -
        // -1   M6N2:   59      -       -           7       -
        // -1   M6N3:   60      2       -           -       -
        // -1   M7N0:   61      -       -           -       -       
        // -1   M7N1:   62      3       -           -       -    
        // -1   M7N2:   63      -       -           8       -    
        // -1   M7N3:   64      4       -           -       -    
        //  0   M0N0K0:  1      -       -           -       -       
        //  0   M0N1:    2      5       -           -       2    
        //  0   M0N2:    3      -       -           -       -    
        //  0   M0N3:    4      6       -           -       -  
        //  0   M1N0:    5      -       -           -       -       
        //  0   M1N1:    6      7       -           -       4    
        //  0   M1N2:    7      -       -           -       -    
        //  0   M1N3:    8      8       -           -       - 
        //  0   M2N0:    9      -       -           -       -       
        //  0   M2N1:   10      9       -           -       6    
        //  0   M2N2:   11      -       -           -       -    
        //  0   M2N3:   12     10       -           -       -  
        //  0   M3N0:   13      -       1           -       -       
        //  0   M3N1:   14     11       -           -       8    
        //  0   M3N2:   15      -       -           -       -    
        //  0   M3N3:   16     12       -           -       -
        //  0   M4N0:   17      -       2           -       -       
        //  0   M4N1:   18     13       -           -       -    
        //  0   M4N2:   19      -       -           1       -    
        //  0   M4N3:   20     14       -           -       -
        //  0   M5N0:   21      -       3           -       -       
        //  0   M5N1:   22     15       -           -       -    
        //  0   M5N2:   23      -       -           2       -    
        //  0   M5N3:   24     16       -           -       -
        //  0   M6N0:   25      -       4           -       -       
        //  0   M6N1:   26     17       -           -       -    
        //  0   M6N2:   27      -       -           3       -    
        //  0   M6N3:   28     18       -           -       -
        //  0   M7N0:   29      -       -           -       -       
        //  0   M7N1:   30     19       -           -       -    
        //  0   M7N2:   31      -       -           4       -    
        //  0   M7N3:   32     20       -           -       -
        //  0   M0N0K1: 33      -       -           -       -       
        //  0   M0N1:   34     21       -           -       10    
        //  0   M0N2:   35      -       -           -       -    
        //  0   M0N3:   36     22       -           -       -  
        //  0   M1N0:   37      -       -           -       -       
        //  0   M1N1:   38     23       -           -       12   
        //  0   M1N2:   39      -       -           -       -    
        //  0   M1N3:   40     24       -           -       - 
        //  0   M2N0:   41      -       -           -       -       
        //  0   M2N1:   42     25       -           -       14   
        //  0   M2N2:   43      -       -           -       -    
        //  0   M2N3:   44     26       -           -       -  
        //  0   M3N0:   45      -       5           -       -       
        //  0   M3N1:   46     27       -           -       16   
        //  0   M3N2:   47      -       -           -       -    
        //  0   M3N3:   48     28       -           -       -
        //  0   M4N0:   49      -       6           -       -       
        //  0   M4N1:   50     29       -           -       -    
        //  0   M4N2:   51      -       -           5       -    
        //  0   M4N3:   52     30       -           -       -
        //  0   M5N0:   53      -       7           -       -       
        //  0   M5N1:   54     31       -           -       -    
        //  0   M5N2:   55      -       -           6       -    
        //  0   M5N3:   56     32       -           -       -
        //  0   M6N0:   57      -       8           -       -       
        //  0   M6N1:   58      1       -           -       -    
        //  0   M6N2:   59      -       -           7       -    
        //  0   M6N3:   60      2       -           -       -
        //  0   M7N0:   61      -       -           -       -       
        //  0   M7N1:   62      3       -           -       -    
        //  0   M7N2:   63      -       -           8       -    
        //  0   M7N3:   64      4       -           -       -
        
        #if 0
        constexpr auto dsread_num_perK = dsread_per_wg * MIterPerWarp;
        constexpr auto dswrite_num_perK = (dsread_num_perK + MWarp * NWarp - 1) / (MWarp * NWarp);
        constexpr auto dswrite_rep = (dswrite_num_perK + MIterPerWarp - 1) / MIterPerWarp;

        // index_t dsread_perM[MIterPerWarp];
        // index_t dswrite_perM[MIterPerWarp];
        index_t dsread_perM[MIterPerWarp];
        index_t dswrite_perM[MIterPerWarp];
        index_t load_perM[MIterPerWarp];
        
        constexpr int dswrite_inst = dswrite_num_perK;
        constexpr int NIter_num = NIterPerWarp*mfma_per_wg;

        #pragma unroll
        for(int i=0;i<MIterPerWarp;i++)
        {
            dsread_perM[i] = 2;
            if(i==0)
            {
                dswrite_perM[0] = (dswrite_inst - MIterPerWarp + 2) > 0 ? dswrite_inst - MIterPerWarp + 2 : 0;
            }
            else if(i==MIterPerWarp-1)
            {
                dswrite_perM[MIterPerWarp-1] = 0;
            }
            else
            {
                dswrite_perM[i] = (i + 2 - dswrite_inst) > 0 ? 1 : 0;
            }
        }

        #pragma unroll
        for(int i=0;i<4;i++)
        {
            load_perM[i] = 2;
        }

        #pragma unroll
        for(int i=4;i<8;i++)
        {
            load_perM[i] = 1;
        }  

        #pragma unroll
        for(int i=0;i<MIterPerWarp;i++)
        {
            int biger_num = dsread_perM[i] > load_perM[i] ? (dsread_perM[i] > dswrite_perM[i] ? dsread_perM[i] : dswrite_perM[i]) : (load_perM[i] > dswrite_perM[i] ? load_perM[i] : dswrite_perM[i]);
            int total_num = dsread_perM[i] + load_perM[i] + dswrite_perM[i];
            int gap = (total_num+NIter_num-1)/NIter_num;

            index_t inst_order[MIterPerWarp*10];
            #pragma unroll
            for(int j=0;j<MIterPerWarp*10;j++)
            {
                inst_order[j] = 0;
            }

            int index=0;
            #pragma unroll
            for(int j=0;j<biger_num;j++)
            {
                if(dswrite_perM[i]>j)
                {
                    inst_order[index] = 1;
                    index++;
                }
                if(load_perM[i]>j)
                {
                    inst_order[index] = 2;
                    index++;
                }
                if(dsread_perM[i]>j)
                {
                    inst_order[index] = 3;
                    index++;
                }
            }

            #pragma unroll
            for(int j=0;j<NIter_num;j++)
            {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                #pragma unroll
                for(int m=0;m<gap;m++)
                {
                    if(m%2==0)
                    {
                        if(inst_order[j+m*NIter_num]==1)
                        {
                            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        }
                        if(inst_order[j+m*NIter_num]==2)
                        {
                            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        }
                        if(inst_order[j+m*NIter_num]==3)
                        {
                            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        }
                    }
                    else
                    {
                        if(inst_order[(m+1)*NIter_num-1-j]==1)
                        {
                            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                        }
                        if(inst_order[(m+1)*NIter_num-1-j]==2)
                        {
                            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        }
                        if(inst_order[(m+1)*NIter_num-1-j]==3)
                        {
                            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        }
                    }

                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        #endif

        if constexpr(kMPerBlock == 128 && kNPerBlock == 128 && kKPerBlock == 128)
        {
            constexpr index_t KPerLoad = Problem::VectorLoadSize / sizeof(ADataType); 
            constexpr index_t A_Buffer_Load_Inst_Num = kMPerBlock * kKPerBlock / BlockSize / KPerLoad;
            constexpr index_t A_LDS_Read_Inst_Num = MIterPerWarp * KIterPerWarp;
            constexpr index_t B_Buffer_Load_Inst_Num = NIterPerWarp * KIterPerWarp;

            static_for<0, A_LDS_Read_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            });
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            });
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            });
            static_for<0, B_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            });
            __builtin_amdgcn_sched_barrier(0);
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
        const index_t iMWarp = get_warp_id() / NWarp;

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

        auto a_lds_block_ping = make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_lds_block_desc);
        auto a_lds_block_pong = make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_lds_block_desc);

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
        auto a_warp_window_ping_tmp = make_tile_window(
            a_lds_block_ping,
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            {iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        auto a_warp_window_pong_tmp = make_tile_window(
            a_lds_block_pong,
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
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        auto a_warp_tensor_ping = load_tile(a_warp_windows_ping(mIter)(kIter));
    
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
    
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor_ping, b_warp_tensor_ping(nIter)(kIter));
    
                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});
            block_sync_lds();

            HotLoopScheduler();
            
            //Next K
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
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        auto a_warp_tensor_pong = load_tile(a_warp_windows_pong(mIter)(kIter));

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
    
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor_pong, b_warp_tensor_pong(nIter)(kIter));
    
                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});
            block_sync_lds();

            HotLoopScheduler();

            iCounter--;
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
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
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        auto a_warp_tensor_ping = load_tile(a_warp_windows_ping(mIter)(kIter));
    
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                        
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor_ping, b_warp_tensor_ping(nIter)(kIter));
    
                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });

            block_sync_lds();
            TailHotLoopScheduler();
            
            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        auto a_warp_tensor_pong = load_tile(a_warp_windows_pong(mIter)(kIter));
    
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                        
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor_pong, b_warp_tensor_pong(nIter)(kIter));
    
                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        auto a_warp_tensor_ping = load_tile(a_warp_windows_ping(mIter)(kIter));
    
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                        
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor_ping, b_warp_tensor_ping(nIter)(kIter));
    
                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }

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
