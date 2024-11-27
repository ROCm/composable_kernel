// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV3
{
    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        ignore = num_loop;
        return TailNumber::Full;
    }
};

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1
template <typename Problem, typename Policy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy>
struct GemmPipelineAgBgCrCompV3 : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base = BaseGemmPipelineAgBgCrCompV3<Problem>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t VectorSizeA = Problem::VectorSizeA;
    static constexpr index_t VectorSizeB = Problem::VectorSizeB;
    static constexpr index_t VectorSizeC = Problem::VectorSizeC;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    // Where is the right place for HasHotLoop and TailNum ???
    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    using Base::PrefetchStages;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave>
    {
        template <typename DstBlockTile, typename SrcTileWindow>
        CK_TILE_DEVICE void GlobalPrefetch(DstBlockTile& dst_block_tile,
                                           SrcTileWindow& dram_tile_window) const
        {
            load_tile(dst_block_tile, dram_tile_window);
            move_tile_window(dram_tile_window, {0, KPerBlock});
        }

        template <typename DstTileWindow, typename SrcBlockTile, typename ElementFunction>
        CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                         const SrcBlockTile& src_block_tile,
                                         const ElementFunction& element_func) const
        {
            const auto block_tile_tmp = tile_elementwise_in(element_func, src_block_tile);
            store_tile(lds_tile_window, block_tile_tmp);
        }

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(number<0>{});
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(number<1>{});
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(number<2>{});

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(number<0>{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(number<1>{});

            constexpr index_t A_LDS_Read_Width = KPerXDL;
            constexpr index_t B_LDS_Read_Width = KPerXDL;

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * VectorSizeA);
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * VectorSizeB);

            constexpr index_t A_LDS_Write_Inst_Num = MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Write_Inst_Num = NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * MPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            // A/B split schedule
            // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
            constexpr auto num_ds_read_inst_a = A_LDS_Read_Width * sizeof(ADataType) == 16
                                                    ? A_LDS_Read_Inst_Num
                                                    : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b = B_LDS_Read_Width * sizeof(BDataType) == 16
                                                    ? B_LDS_Read_Inst_Num
                                                    : B_LDS_Read_Inst_Num / 2;

            constexpr auto num_ds_write_inst_a = A_LDS_Write_Inst_Num;
            constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

            constexpr auto num_buffer_load_inst_a = A_Buffer_Load_Inst_Num;
            constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

            constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

            constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
            constexpr auto ds_read_a_issue_cycle =
                A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
            constexpr auto ds_read_b_issue_cycle =
                B_LDS_Read_Width * sizeof(BDataType) == 16 ? 8 : 4;
            constexpr auto ds_read_a_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
            constexpr auto ds_read_b_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

            constexpr auto num_dsread_a_mfma =
                (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_b_mfma =
                (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

            // stage 1
            // Separate this part?
            // constexpr auto num_mfma_per_ds_read = sizeof(ComputeDataType) / sizeof(ADataType) >
            //                                               sizeof(ComputeDataType) /
            //                                               sizeof(BDataType)
            //                                           ? sizeof(ComputeDataType) /
            //                                           sizeof(ADataType) : sizeof(ComputeDataType)
            //                                           / sizeof(BDataType);
            constexpr auto num_mfma_stage1 =
                num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
            constexpr auto num_mfma_per_issue =
                num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
            constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
            constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

            static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_a, 0); // MFMA
            });
            static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_b, 0); // MFMA
            });

            // stage 2
            static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_a - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_b - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            static_assert(MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                              NPerBlock ==
                                  BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                              KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                          "A/B block window appropriate sizes must be equal to MPerBlock/NPerblock"
                          " or KPerBlock!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            // A tile in LDS
            ADataType* p_a_lds              = static_cast<ADataType*>(p_smem);
            constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
            auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

            // TODO: LDS alignment should come from Policy!
            constexpr index_t a_lds_block_space_size_aligned =
                integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(),
                                    16) *
                16;

            // B tile in LDS
            BDataType* p_b_lds = static_cast<BDataType*>(
                static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
            constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
            auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

            // A DRAM tile window for load
            auto a_copy_dram_window =
                make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 a_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeADramTileDistribution<Problem>());

            // A LDS tile window for store
            auto a_copy_lds_window =
                make_tile_window(a_lds_block,
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 a_copy_dram_window.get_tile_distribution());
            // B DRAM tile window for load
            auto b_copy_dram_window =
                make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 b_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeBDramTileDistribution<Problem>());

            // B LDS tile window for store
            auto b_copy_lds_window =
                make_tile_window(b_lds_block,
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 b_copy_dram_window.get_tile_distribution());

            // A LDS tile for block GEMM
            auto a_lds_gemm_window = make_tile_window(
                a_lds_block, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});
            // B LDS tile for block GEMM
            auto b_lds_gemm_window = make_tile_window(
                b_lds_block, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            ABlockTile a_block_tile;
            BBlockTile b_block_tile;

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            GlobalPrefetch(a_block_tile, a_copy_dram_window);
            GlobalPrefetch(b_block_tile, b_copy_dram_window);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
            LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);

            GlobalPrefetch(a_block_tile, a_copy_dram_window);
            GlobalPrefetch(b_block_tile, b_copy_dram_window);

            block_sync_lds();
            block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);

            __builtin_amdgcn_sched_barrier(0);

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    block_sync_lds();

                    LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                    LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);

                    GlobalPrefetch(a_block_tile, a_copy_dram_window);
                    GlobalPrefetch(b_block_tile, b_copy_dram_window);

                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                    block_sync_lds();
                    block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);

                    i += 1;
                } while(i < (num_loop - 1));
            }
            // tail
            if constexpr(TailNum == TailNumber::Full)
            {
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            // Let's leak last MFMA block to epilogue region, cover the potential lds-shuffle
            // latency
            // __builtin_amdgcn_sched_barrier(0);
            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem);
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
