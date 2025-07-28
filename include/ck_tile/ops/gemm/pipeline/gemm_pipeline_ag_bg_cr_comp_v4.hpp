// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV4
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
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::Three;
        }
        else
        {
            return TailNumber::Two;
        }
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        // Handle all the valid cases.
        if(has_hot_loop)
        {
            if(tail_number == TailNumber::Three)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Three>{});
            }
            else if(tail_number == TailNumber::Two)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
        }
        else
        {
            if(tail_number == TailNumber::Three)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Three>{});
            }
            else if(tail_number == TailNumber::Two)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
        }
        // If execution reaches here, it's an invalid tail_number because it wasn't handled above.
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error("Invalid TailNumber: Only TailNumber::Full and smaller than "
                               "PrefetchStages are supported.");
#endif
    }
};

/**
 * @brief Compute optimized pipeline version 4
 *
 * This version introduces a dual LDS window mechanism using a ping-pong buffer approach
 * for more efficient data handling from global memory. Unlike compute version 3, this method
 * allows one LDS to fetch data from global memory while the other LDS executes warps for MFMA
 * matrix multiplication. This dual operation helps in keeping the Warp unit continuously busy,
 * thereby significantly reducing memory load times and enhancing overall performance.
 *
 * @note This version shows improved performance over Compute Version 3 with the same block tile.
 * It is particularly more efficient for large matrices where M, N, and K are greater than 8K,
 * even when Compute Version 3's block size is twice that of Compute Version 4.
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy>
struct GemmPipelineAgBgCrCompV4 : public BaseGemmPipelineAgBgCrCompV4<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV4<Problem>;
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

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrCompV3", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(),  GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

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
        using Base = PipelineImplBase;

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

            constexpr auto num_ds_read_inst     = num_ds_read_inst_a + num_ds_read_inst_b;
            constexpr auto num_ds_write_inst    = A_LDS_Write_Inst_Num + B_LDS_Write_Inst_Num;
            constexpr auto num_buffer_load_inst = A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num;
            constexpr auto num_issue            = num_buffer_load_inst;

            static_for<0, num_buffer_load_inst, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(
                    0x100, num_ds_read_inst / num_issue, 0);       // DS read : 2
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA: 1
                __builtin_amdgcn_sched_group_barrier(
                    0x200, num_ds_write_inst / num_issue, 0);      // DS write : 1
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read :1
                __builtin_amdgcn_sched_group_barrier(
                    0x008, C_MFMA_Inst_Num / num_issue - 3, 0); // MFMA : 5
            });
            __builtin_amdgcn_sched_barrier(0);
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
                                       void* __restrict__ p_smem_0,
                                       void* __restrict__ p_smem_1) const
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

            ////////////// global window & register /////////////////
            // A DRAM tile window for load
            auto a_copy_dram_window =
                make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 a_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeADramTileDistribution<Problem>());

            // B DRAM tile window for load
            auto b_copy_dram_window =
                make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 b_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeBDramTileDistribution<Problem>());

            // A register tile for global load
            constexpr auto ABlockTileDistr = a_copy_dram_window.get_tile_distribution();
            constexpr auto BBlockTileDistr = b_copy_dram_window.get_tile_distribution();
            using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr));
            using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr));
            ABlockTile a_global_load_tile;
            BBlockTile b_global_load_tile;

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // global prefetch 0
            // global read 0
            Base::GlobalPrefetch(a_global_load_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_global_load_tile, b_copy_dram_window, b_dram_tile_window_step);
            ////////////// LDS desc, window & register /////////////////
            auto&& [a_lds_block0, b_lds_block0] = Base::GetABLdsTensorViews(p_smem_0);
            auto&& [a_lds_block1, b_lds_block1] = Base::GetABLdsTensorViews(p_smem_1);

            constexpr auto a_lds_shape = []() {
                if constexpr(is_a_load_tr_v())
                    return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
                else
                    return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
            }();
            auto a_copy_lds_window0 = make_tile_window(a_lds_block0, a_lds_shape, {0, 0});
            auto a_copy_lds_window1 = make_tile_window(a_lds_block1, a_lds_shape, {0, 0});

            constexpr auto b_lds_shape = []() {
                if constexpr(is_b_load_tr_v())
                    return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
                else
                    return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
            }();
            auto b_copy_lds_window0 = make_tile_window(b_lds_block0, b_lds_shape, {0, 0});
            auto b_copy_lds_window1 = make_tile_window(b_lds_block1, b_lds_shape, {0, 0});

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_global_load_tile);
                Base::LocalPrefill(a_copy_lds_window0, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window0, a_global_load_tile, a_element_func);
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_global_load_tile);
                Base::LocalPrefill(b_copy_lds_window0, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window0, b_global_load_tile, b_element_func);
            }

            // global read 1
            Base::GlobalPrefetch(a_global_load_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_global_load_tile, b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();

            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto BLdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));
            ALdsTile a_block_tile0, a_block_tile1;
            BLdsTile b_block_tile0, b_block_tile1;

            constexpr auto a_lds_input_tile_distr = [&]() {
                if constexpr(is_a_load_tr_v())
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            decltype(BlockGemm::MakeABlockDistributionEncode()),
                            typename Problem::ADataType>::TransposedDstrEncode{});
                else
                    return ALdsTileDistr;
            }();
            constexpr auto b_lds_input_tile_distr = [&]() {
                if constexpr(is_b_load_tr_v())
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            decltype(BlockGemm::MakeBBlockDistributionEncode()),
                            typename Problem::BDataType>::TransposedDstrEncode{});
                else
                    return BLdsTileDistr;
            }();
            auto a_lds_ld_window0 =
                make_tile_window(a_lds_block0, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto a_lds_ld_window1 =
                make_tile_window(a_lds_block1, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto b_lds_ld_window0 =
                make_tile_window(b_lds_block0, b_lds_shape, {0, 0}, b_lds_input_tile_distr);
            auto b_lds_ld_window1 =
                make_tile_window(b_lds_block1, b_lds_shape, {0, 0}, b_lds_input_tile_distr);

            static_assert(!is_tile_window_linear_v<decltype(a_lds_ld_window0)> &&
                              !is_tile_window_linear_v<decltype(a_lds_ld_window1)> &&
                              !is_tile_window_linear_v<decltype(b_lds_ld_window0)> &&
                              !is_tile_window_linear_v<decltype(b_lds_ld_window1)>,
                          "LDS windows must not be linear");

            Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
            Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);

            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_global_load_tile);
                Base::LocalPrefill(a_copy_lds_window1, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window1, a_global_load_tile, a_element_func);
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_global_load_tile);
                Base::LocalPrefill(b_copy_lds_window1, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window1, b_global_load_tile, b_element_func);
            }

            Base::GlobalPrefetch(a_global_load_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_global_load_tile, b_copy_dram_window, b_dram_tile_window_step);

            if(HasHotLoop)
            {
                // minus 2 because we have ping-pong double buffer.
                index_t iCounter = __builtin_amdgcn_readfirstlane(num_loop - 2);
                do
                {
                    // ping
                    {
                        block_sync_lds();
                        Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);

                        if constexpr(is_a_col_major && !is_a_load_tr_v())
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, a_global_load_tile);
                            Base::LocalPrefill(a_copy_lds_window0, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window0, a_global_load_tile, a_element_func);
                        }
                        if constexpr(is_b_row_major && !is_b_load_tr_v())
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, b_global_load_tile);
                            Base::LocalPrefill(b_copy_lds_window0, b_shuffle_tmp, b_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                b_copy_lds_window0, b_global_load_tile, b_element_func);
                        }

                        Base::GlobalPrefetch(
                            a_global_load_tile, a_copy_dram_window, a_dram_tile_window_step);
                        Base::GlobalPrefetch(
                            b_global_load_tile, b_copy_dram_window, b_dram_tile_window_step);
                        // gemm
                        block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                        HotLoopScheduler();
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    // pong
                    {
                        block_sync_lds();
                        Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);

                        if constexpr(is_a_col_major && !is_a_load_tr_v())
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, a_global_load_tile);
                            Base::LocalPrefill(a_copy_lds_window1, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window1, a_global_load_tile, a_element_func);
                        }
                        if constexpr(is_b_row_major && !is_b_load_tr_v())
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, b_global_load_tile);
                            Base::LocalPrefill(b_copy_lds_window1, b_shuffle_tmp, b_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                b_copy_lds_window1, b_global_load_tile, b_element_func);
                        }
                        block_sync_lds();

                        Base::GlobalPrefetch(
                            a_global_load_tile, a_copy_dram_window, a_dram_tile_window_step);
                        Base::GlobalPrefetch(
                            b_global_load_tile, b_copy_dram_window, b_dram_tile_window_step);
                        // gemm
                        block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                        HotLoopScheduler();
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    iCounter -= 2;
                } while(iCounter > 1);
            }

            // tail 3
            if(TailNum == TailNumber::Three)
            {
                // 3
                {
                    block_sync_lds();
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_global_load_tile);
                        Base::LocalPrefill(a_copy_lds_window0, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window0, a_global_load_tile, a_element_func);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_global_load_tile);
                        Base::LocalPrefill(b_copy_lds_window0, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window0, b_global_load_tile, b_element_func);
                    }
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                }
                // 2
                {
                    block_sync_lds();
                    Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                }
                // 1
                {
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            else
            {
                // 2
                {
                    block_sync_lds();
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                    static_for<0, 8, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
                    });
                    __builtin_amdgcn_sched_barrier(0);
                }
                // 1
                {
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                    __builtin_amdgcn_sched_barrier(0);
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
                                   void* p_smem_0,
                                   void* p_smem_1) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem_0,
            p_smem_1);
    }

    public:
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem_0,
                                   void* __restrict__ p_smem_1) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem_0,
            p_smem_1);
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* __restrict__ p_smem_0,
                                   void* __restrict__ p_smem_1) const
    {
        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            constexpr bool hot_loop    = hot_loop_.value;
            constexpr auto tail_num    = tail_num_.value;
            constexpr auto PassThrough = [](const auto& x) { return x; };
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop, tail_num>(
                a_dram_block_window_tmp,
                PassThrough,
                b_dram_block_window_tmp,
                PassThrough,
                num_loop,
                p_smem_0,
                p_smem_1);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }
};
} // namespace ck_tile
