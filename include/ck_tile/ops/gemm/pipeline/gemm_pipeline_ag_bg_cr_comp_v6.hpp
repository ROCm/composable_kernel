// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV6
{
    static constexpr index_t PrefetchStages  = 3;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 2;
    static constexpr index_t HotloopUnroll   = 2;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % HotloopUnroll == 1)
        {
            return TailNumber::Odd;
        }
        else
        {
            return TailNumber::Even;
        }
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto TailHandler(const RunFunction& run_func,
                                                [[maybe_unused]] bool has_hot_loop,
                                                TailNumber tail_num)
    {
        if(tail_num == ck_tile::TailNumber::Three)
        {
            return run_func(
                has_hot_loop,
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
        }
        else
        {
            return run_func(
                has_hot_loop,
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
        }
    }
};

/**
 * @brief Compute optimized pipeline version 5 TODO
 *
 *
 * @note TODO
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV6DefaultPolicy>
struct GemmPipelineAgBgCrCompV6 : public BaseGemmPipelineAgBgCrCompV6<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV6<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm          = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    // TODO check KRepeat
    static constexpr index_t KRepeat = KPerBlock / GetSmemPackA();

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Policy::template IsTransposeC<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0);
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1);
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2);

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0);
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1);

            constexpr index_t A_LDS_Read_Width = KPerXDL;
            constexpr index_t B_LDS_Read_Width = KPerXDL;

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t A_LDS_Write_Inst_Num = MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Write_Inst_Num = NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_ds_read_inst_a =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? A_LDS_Read_Inst_Num
                                                                         : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? B_LDS_Read_Inst_Num
                                                                         : B_LDS_Read_Inst_Num / 2;

            constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
            constexpr auto ds_read_a_issue_cycle =
                A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
            constexpr auto ds_read_b_issue_cycle =
                B_LDS_Read_Width * sizeof(BDataType) == 16 ? 8 : 4;
            constexpr auto ds_read_a_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
            constexpr auto ds_read_b_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

            constexpr auto num_dsread_stage1_a = num_ds_read_inst_a / KRepeat * (KRepeat - 1);
            constexpr auto num_dsread_stage1_b = num_ds_read_inst_b / KRepeat * (KRepeat - 1);
            constexpr auto num_dsread_stage3_a = num_ds_read_inst_a / KRepeat;
            constexpr auto num_dsread_stage3_b = num_ds_read_inst_b / KRepeat;

            constexpr auto num_dsread_stage1_a_mfma =
                (num_dsread_stage1_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_stage1_b_mfma =
                (num_dsread_stage1_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;
            constexpr auto num_dsread_stage3_a_mfma =
                (num_dsread_stage3_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_stage3_b_mfma =
                (num_dsread_stage3_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

            constexpr auto num_mfma_stage2 = C_MFMA_Inst_Num -
                                             num_ds_read_inst_a / ds_read_a_mfma_rate -
                                             num_ds_read_inst_b / ds_read_b_mfma_rate;
            constexpr auto num_mfma_per_issue =
                num_mfma_stage2 / (A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num);
            constexpr auto num_dswrite_per_issue_a = A_LDS_Write_Inst_Num / A_Buffer_Load_Inst_Num;
            constexpr auto num_dswrite_per_issue_b = B_LDS_Write_Inst_Num / B_Buffer_Load_Inst_Num;

            // stage 1
            static_for<0, num_dsread_stage1_a_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage1_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage1_a - (num_dsread_stage1_a_mfma - 1) * ds_read_a_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, num_dsread_stage1_b_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage1_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage1_b - (num_dsread_stage1_b_mfma - 1) * ds_read_b_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            // stage 2
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
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
            static_for<0, B_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
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

            // stage 3
            static_for<0, num_dsread_stage3_a_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage3_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage3_a - (num_dsread_stage3_a_mfma - 1) * ds_read_a_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, num_dsread_stage3_b_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage3_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage3_b - (num_dsread_stage3_b_mfma - 1) * ds_read_b_mfma_rate,
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
                "Data Type conflict on A and B matrix input data type.");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "B block window has incorrect lengths for defined BLayout!");

            ////////////// LDS desc, window & register /////////////////
            using ALdsType =
                remove_cvref_t<decltype(PipelineImplBase::GetABLdsTensorViews(p_smem).at(I0))>;
            using BLdsType =
                remove_cvref_t<decltype(PipelineImplBase::GetABLdsTensorViews(p_smem).at(I1))>;
            auto&& ABLdsTensorViews = PipelineImplBase::GetABLdsTensorViews(p_smem);
            ALdsType& a_lds_block   = ABLdsTensorViews.at(I0);
            BLdsType& b_lds_block   = ABLdsTensorViews.at(I1);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using acopy_dram_type =
                remove_cvref_t<decltype(PipelineImplBase::GetAWindows(a_dram_block_window_tmp,
                                                                      a_lds_block,
                                                                      a_lds_load_tile_distr)
                                            .at(I0))>;
            using bcopy_dram_type =
                remove_cvref_t<decltype(PipelineImplBase::GetBWindows(b_dram_block_window_tmp,
                                                                      b_lds_block,
                                                                      b_lds_load_tile_distr)
                                            .at(I0))>;

            using a_copy_lds_window_type =
                remove_cvref_t<decltype(PipelineImplBase::GetAWindows(a_dram_block_window_tmp,
                                                                      a_lds_block,
                                                                      a_lds_load_tile_distr)
                                            .at(I1))>;
            using b_copy_lds_window_type =
                remove_cvref_t<decltype(PipelineImplBase::GetBWindows(b_dram_block_window_tmp,
                                                                      b_lds_block,
                                                                      b_lds_load_tile_distr)
                                            .at(I1))>;

            using a_lds_load_tile_distr_type =
                remove_cvref_t<decltype(PipelineImplBase::GetAWindows(a_dram_block_window_tmp,
                                                                      a_lds_block,
                                                                      a_lds_load_tile_distr)
                                            .at(I2))>;
            using b_lds_load_tile_distr_type =
                remove_cvref_t<decltype(PipelineImplBase::GetBWindows(b_dram_block_window_tmp,
                                                                      b_lds_block,
                                                                      b_lds_load_tile_distr)
                                            .at(I2))>;

            auto&& aWindows = PipelineImplBase::GetAWindows(
                a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto&& bWindows = PipelineImplBase::GetBWindows(
                b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            acopy_dram_type& a_copy_dram_window           = aWindows.at(I0);
            a_copy_lds_window_type& a_copy_lds_window     = aWindows.at(I1);
            a_lds_load_tile_distr_type& a_lds_gemm_window = aWindows.at(I2);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            bcopy_dram_type& b_copy_dram_window           = bWindows.at(I0);
            b_copy_lds_window_type& b_copy_lds_window     = bWindows.at(I1);
            b_lds_load_tile_distr_type& b_lds_gemm_window = bWindows.at(I2);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            ABlockTile a_block_tile[Base::GlobalBufferNum];
            BBlockTile b_block_tile[Base::GlobalBufferNum];

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            constexpr auto ALdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto BLdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            ALdsTile a_lds_tile;
            BLdsTile b_lds_tile;
            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // Global prefetch 1
            PipelineImplBase::GlobalPrefetch(
                a_block_tile[I0], a_copy_dram_window, a_dram_tile_window_step);
            PipelineImplBase::GlobalPrefetch(
                b_block_tile[I0], b_copy_dram_window, b_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // Local prefill 1
            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tile[I0]);
                PipelineImplBase::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                PipelineImplBase::LocalPrefill(a_copy_lds_window, a_block_tile[I0], a_element_func);
            }
            if constexpr(is_b_row_major)
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tile[I0]);
                PipelineImplBase::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
            }
            else
            {
                PipelineImplBase::LocalPrefill(b_copy_lds_window, b_block_tile[I0], b_element_func);
            }

            // Global prefetch 2
            PipelineImplBase::GlobalPrefetch(
                a_block_tile[I0], a_copy_dram_window, a_dram_tile_window_step);
            PipelineImplBase::GlobalPrefetch(
                b_block_tile[I0], b_copy_dram_window, b_dram_tile_window_step);

            // Global prefetch 3
            PipelineImplBase::GlobalPrefetch(
                a_block_tile[I1], a_copy_dram_window, a_dram_tile_window_step);
            PipelineImplBase::GlobalPrefetch(
                b_block_tile[I1], b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();

            // Local prefetch 1
            PipelineImplBase::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
            PipelineImplBase::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

            if(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    auto LoopFunc = [&](auto vmem_buf_idx) {
                        block_sync_lds();

                        // Local prefill 2
                        if constexpr(is_a_col_major)
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, a_block_tile[vmem_buf_idx]);
                            PipelineImplBase::LocalPrefill(
                                a_copy_lds_window, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            PipelineImplBase::LocalPrefill(
                                a_copy_lds_window, a_block_tile[vmem_buf_idx], a_element_func);
                        }
                        if constexpr(is_b_row_major)
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, b_block_tile[vmem_buf_idx]);
                            PipelineImplBase::LocalPrefill(
                                b_copy_lds_window, b_shuffle_tmp, b_element_func);
                        }
                        else
                        {
                            PipelineImplBase::LocalPrefill(
                                b_copy_lds_window, b_block_tile[vmem_buf_idx], b_element_func);
                        }

                        // Global prefetch 4
                        PipelineImplBase::GlobalPrefetch(a_block_tile[vmem_buf_idx],
                                                         a_copy_dram_window,
                                                         a_dram_tile_window_step);
                        PipelineImplBase::GlobalPrefetch(b_block_tile[vmem_buf_idx],
                                                         b_copy_dram_window,
                                                         b_dram_tile_window_step);

                        block_sync_lds();

                        block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                        // Local prefetch 2
                        PipelineImplBase::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                        PipelineImplBase::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

                        HotLoopScheduler();
                        __builtin_amdgcn_sched_barrier(0);
                    };

                    LoopFunc(I0);
                    LoopFunc(I1);

                    i += Base::HotloopUnroll;
                } while(i < (num_loop - Base::PrefetchStages));
            }

            auto ReadWriteCompFunc = [&](auto vmem_buf_idx) {
                block_sync_lds();

                // Local prefill 3
                if constexpr(is_a_col_major)
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, a_block_tile[vmem_buf_idx]);
                    PipelineImplBase::LocalPrefill(
                        a_copy_lds_window, a_shuffle_tmp, a_element_func);
                }
                else
                {
                    PipelineImplBase::LocalPrefill(
                        a_copy_lds_window, a_block_tile[vmem_buf_idx], a_element_func);
                }
                if constexpr(is_b_row_major)
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, b_block_tile[vmem_buf_idx]);
                    PipelineImplBase::LocalPrefill(
                        b_copy_lds_window, b_shuffle_tmp, b_element_func);
                }
                else
                {
                    PipelineImplBase::LocalPrefill(
                        b_copy_lds_window, b_block_tile[vmem_buf_idx], b_element_func);
                }

                block_sync_lds();
                block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                PipelineImplBase::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                PipelineImplBase::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

                HotLoopScheduler();
            };

            auto ReadCompFunc = [&]() {
                block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                // Local prefetch 4
                PipelineImplBase::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                PipelineImplBase::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

                block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                HotLoopScheduler();
            };

            if constexpr(TailNum == TailNumber::Odd)
            {
                ReadWriteCompFunc(I0);
                ReadWriteCompFunc(I1);
                ReadCompFunc();
            }
            else if constexpr(TailNum == TailNumber::Even)
            {
                ReadWriteCompFunc(I0);
                ReadCompFunc();
            }

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

    public:
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
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
