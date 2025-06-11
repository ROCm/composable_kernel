// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV5
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
};

/**
 * @brief Compute optimized pipeline version 5 TODO
 *
 *
 * @note TODO
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV5DefaultPolicy>
struct GemmPipelineAgBgCrCompV5 : public BaseGemmPipelineAgBgCrCompV5<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV5<Problem>;
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

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

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
    static constexpr index_t KRepeat = 1;

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
        using Base                              = PipelineImplBase;
        static constexpr index_t PrefetchStages = 3;
        static constexpr index_t HotloopUnroll  = 2;

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0{});
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1{});
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2{});

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

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
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            ////////////// LDS desc, window & register /////////////////
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            ABlockTile a_block_tile[2]; // TODO GlobalBufferNum
            BBlockTile b_block_tile[2]; // TODO GlobalBufferNum

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

            // a_lds_block.print();

            // Global prefetch 1
            Base::GlobalPrefetch(a_block_tile[I0{}], a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile[I0{}], b_copy_dram_window, b_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // Local prefill 1
            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tile[I0{}]);
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tile[I0{}], a_element_func);
            }
            if constexpr(is_b_row_major)
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tile[I0{}]);
                Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window, b_block_tile[I0{}], b_element_func);
            }

            // Global prefetch 2
            Base::GlobalPrefetch(a_block_tile[I0{}], a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile[I0{}], b_copy_dram_window, b_dram_tile_window_step);

            // Global prefetch 3
            Base::GlobalPrefetch(a_block_tile[I1{}], a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile[I1{}], b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();

            // Local prefetch 1
            Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
            Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

            // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            //    printf("First value of a_lds_tile: %f\n",
            //    static_cast<float>(a_lds_tile.get_thread_buffer().get(0)));
            //}

            __builtin_amdgcn_sched_barrier(0);

            if(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    { // Loop 1
                        block_sync_lds();

                        // Local prefill 2
                        if constexpr(is_a_col_major)
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, a_block_tile[I0{}]);
                            Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window, a_block_tile[I0{}], a_element_func);
                        }
                        if constexpr(is_b_row_major)
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, b_block_tile[I0{}]);
                            Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                b_copy_lds_window, b_block_tile[I0{}], b_element_func);
                        }

                        // Global prefetch 4
                        Base::GlobalPrefetch(
                            a_block_tile[I0{}], a_copy_dram_window, a_dram_tile_window_step);
                        Base::GlobalPrefetch(
                            b_block_tile[I0{}], b_copy_dram_window, b_dram_tile_window_step);

                        block_sync_lds();

                        block_gemm(c_block_tile, a_lds_tile, b_lds_tile);
                        Policy::template GetVectorSizeA<Problem>();

                        // Local prefetch 2
                        Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                        Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

                        HotLoopScheduler();
                        __builtin_amdgcn_sched_barrier(0);
                    }

                    { // Loop 2
                        block_sync_lds();

                        // Local prefill 2
                        if constexpr(is_a_col_major)
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, a_block_tile[I1{}]);
                            Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window, a_block_tile[I1{}], a_element_func);
                        }
                        if constexpr(is_b_row_major)
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, b_block_tile[I1{}]);
                            Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                b_copy_lds_window, b_block_tile[I1{}], b_element_func);
                        }

                        // Global prefetch 4
                        Base::GlobalPrefetch(
                            a_block_tile[I1{}], a_copy_dram_window, a_dram_tile_window_step);
                        Base::GlobalPrefetch(
                            b_block_tile[I1{}], b_copy_dram_window, b_dram_tile_window_step);

                        block_sync_lds();

                        block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                        // Local prefetch 2
                        Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                        Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

                        HotLoopScheduler();
                        __builtin_amdgcn_sched_barrier(0);
                    }

                    i += HotloopUnroll;
                } while(i < (num_loop - PrefetchStages));
            }

            // printf("TailNum: %d\n", static_cast<int>(TailNum));

            if constexpr(TailNum == TailNumber::Odd)
            {
                // printf("TailNum == Odd...\n");
                // ReadWriteCompFunc();
                {
                    block_sync_lds();

                    // Local prefill 3
                    if constexpr(is_a_col_major)
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile[I0{}]);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, a_block_tile[I0{}], a_element_func);
                    }
                    if constexpr(is_b_row_major)
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile[I0{}]);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, b_block_tile[I0{}], b_element_func);
                    }

                    block_sync_lds();
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    // Local prefetch 3
                    Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                    Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);
                    HotLoopScheduler();
                }
                // ReadWriteCompFunc();
                {
                    block_sync_lds();

                    // Local prefill 3
                    if constexpr(is_a_col_major)
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile[I1{}]);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, a_block_tile[I1{}], a_element_func);
                    }
                    if constexpr(is_b_row_major)
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile[I1{}]);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, b_block_tile[I1{}], b_element_func);
                    }

                    block_sync_lds();
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    // Local prefetch 3
                    Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                    Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);
                    HotLoopScheduler();
                }
                // ReadCompFunc();
                {
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    // Local prefetch 4
                    Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                    Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

                    // TODO verify
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    HotLoopScheduler();
                }
            }
            else if constexpr(TailNum == TailNumber::Even)
            {
                // printf("TailNum == Even...\n");
                // ReadWriteCompFunc();
                {
                    block_sync_lds();

                    // Local prefill 3
                    if constexpr(is_a_col_major)
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile[I0{}]);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, a_block_tile[I0{}], a_element_func);
                    }
                    if constexpr(is_b_row_major)
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile[I0{}]);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, b_block_tile[I0{}], b_element_func);
                    }

                    block_sync_lds();
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    // Local prefetch 3
                    Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                    Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);
                    HotLoopScheduler();
                }
                // ReadCompFunc();
                { // one
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    // Local prefetch 4
                    Base::LocalPrefetch(a_lds_tile, a_lds_gemm_window);
                    Base::LocalPrefetch(b_lds_tile, b_lds_gemm_window);

                    // TODO verify
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    HotLoopScheduler();
                }
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
