// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <typename Problem, typename Policy = GemmABQuantPipelineAgBgCrDefaultPolicy>
struct ABQuantGemmPipelineAgBgCrCompV4 : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmABQuantPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using AQDataType      = remove_cvref_t<typename Problem::AQDataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
    using AQuantGroupSize = remove_cvref_t<typename Problem::AQuantGroupSize>;
    using BQuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;
    // BDataType gets converted from PkInt4 during loading
    using OverrideBDataType =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;

    static_assert(BQuantGroupSize::kM == 1, "only N/K blocks for BQuant kernel!");
    static_assert(AQuantGroupSize::kN == 1, "only M/K blocks for AQuant kernel!");
    static_assert(AQuantGroupSize::kM == 1, "no block M for AQuant kernel supported yet!");
    static_assert(AQuantGroupSize::kK == BQuantGroupSize::kK,
                  "AQuantGroupSize::kK should be equal to BQuantGroupSize::kK");

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t AQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<AQDataType>>::PackedSize;

    static constexpr index_t BQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BQDataType>>::PackedSize;

    using ALayout  = remove_cvref_t<typename Problem::ALayout>;
    using AQLayout = remove_cvref_t<typename Problem::AQLayout>;
    using BLayout  = remove_cvref_t<typename Problem::BLayout>;
    using BQLayout = remove_cvref_t<typename Problem::BQLayout>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    static constexpr index_t BlockSize   = Problem::kBlockSize;
    static constexpr index_t MPerBlock   = BlockGemmShape::kM;
    static constexpr index_t NPerBlock   = BlockGemmShape::kN;
    static constexpr index_t KPerBlock   = BlockGemmShape::kK;
    static constexpr index_t KPerBlockAQ = BlockGemmShape::kK / AQuantGroupSize::kK;
    static constexpr index_t NPerBlockBQ = BlockGemmShape::kN / BQuantGroupSize::kN;
    static constexpr index_t KPerBlockBQ = BlockGemmShape::kK / BQuantGroupSize::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }
    static constexpr index_t GetVectorSizeAQ()
    {
        return Policy::template GetVectorSizeAQ<Problem>();
    }
    static constexpr index_t GetVectorSizeBQ()
    {
        return Policy::template GetVectorSizeBQ<Problem>();
    }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr bool PreshuffleQuant  = Problem::Traits::PreshuffleQuant;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    using Base::PrefetchStages;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "abquant_pipeline_AgBgCrCompV4",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', BlockGemm::WarpGemm::kM, BlockGemm::WarpGemm::kN, BlockGemm::WarpGemm::kK),
                      concat('x', kPadM, kPadN, kPadK), AQuantGroupSize::GetName(), BQuantGroupSize::GetName());
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static std::string Print()
    {
        constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
        constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
        constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

        constexpr index_t WaveSize = 64;
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

        constexpr index_t A_LDS_Read_Width = GetSmemPackA();
        constexpr index_t B_LDS_Read_Width = GetSmemPackB();

        constexpr index_t A_LDS_Write_Width = GetSmemPackA();
        constexpr index_t B_LDS_Write_Width = GetSmemPackB();

        constexpr index_t A_Buffer_Load_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
        constexpr index_t B_Buffer_Load_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());
        constexpr index_t AQ_Buffer_Load_Inst_Num =
            MPerBlock * KPerBlockAQ / (BlockSize * GetVectorSizeAQ());
        constexpr index_t BQ_Buffer_Load_Inst_Num =
            NPerBlockBQ * KPerBlockBQ / (BlockSize * GetVectorSizeBQ());

        constexpr index_t A_LDS_Write_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
        constexpr index_t B_LDS_Write_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

        constexpr index_t A_LDS_Read_Inst_Num =
            WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
        constexpr index_t B_LDS_Read_Inst_Num =
            WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

        constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                            (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

        auto str = std::stringstream{};

        str << "A/B vector size: " << GetVectorSizeA() << ", " << GetVectorSizeB() << ", "
            << "AQ vector size: " << GetVectorSizeAQ() << "\n"
            << "BQ vector size: " << GetVectorSizeBQ() << "\n"
            << "A/B LDS read/write width: " << A_LDS_Read_Width << ", " << B_LDS_Read_Width << "\n"
            << "A/B buffer load inst: " << A_Buffer_Load_Inst_Num << ", " << B_Buffer_Load_Inst_Num
            << ", " << "AQ buffer load inst: " << AQ_Buffer_Load_Inst_Num << "\n"
            << ", " << "BQ buffer load inst: " << BQ_Buffer_Load_Inst_Num << "\n"
            << "A/B LDS write inst: " << A_LDS_Write_Inst_Num << ", " << B_LDS_Write_Inst_Num
            << "\n"
            << "A/B LDS read inst: " << A_LDS_Read_Inst_Num << ", " << B_LDS_Read_Inst_Num << "\n"
            << "C MFMA inst: " << C_MFMA_Inst_Num << "\n"
            << "AQuantGroupSize: " << AQuantGroupSize::GetName() << "\n"
            << "BQuantGroupSize: " << BQuantGroupSize::GetName() << "\n"
            << "KPack: " << BlockGemm::Traits::KPack << "\n"
            << "PrefetchStages: " << PrefetchStages << "\n";
        return str.str();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <typename ADramWindow, typename ABlockTile_>
        CK_TILE_DEVICE static void LoadAndConvertATile(ABlockTile_& a_block_tile,
                                                       const ADramWindow& a_dram_window)
        {
            using DestDataType            = typename ABlockTile_::DataType;
            using SrcDataType             = typename ADramWindow::Base::TileWindowBase::DataType;
            constexpr index_t UnaryOpSize = 8;
            load_int4_tile<SrcDataType, DestDataType, UnaryOpSize>(a_block_tile, a_dram_window);
        }

        template <typename BDramWindow, typename BBlockTile_>
        CK_TILE_DEVICE static void LoadAndConvertBTile(BBlockTile_& b_block_tile,
                                                       const BDramWindow& b_dram_window)
        {
            using DestDataType            = typename BBlockTile_::DataType;
            using SrcDataType             = typename BDramWindow::Base::TileWindowBase::DataType;
            constexpr index_t UnaryOpSize = 8;
            load_int4_tile<SrcDataType, DestDataType, UnaryOpSize>(b_block_tile, b_dram_window);
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AQDramBlockWindowTmp,
                  typename BQDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                       const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                       index_t m,
                                       index_t n,
                                       index_t num_loop,
                                       void* p_smem_0,
                                       void* p_smem_1) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<AQDataType,
                                   remove_cvref_t<typename AQDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BQDataType,
                                   remove_cvref_t<typename BQDramBlockWindowTmp::DataType>>,
                "A/B/AQ/BQ Dram block window should have the same data type as appropriate "
                "([A|B|AQ|BQ]DataType) defined in Problem definition!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_aq_col_major =
                std::is_same_v<AQLayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
            constexpr bool is_bq_row_major =
                std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>;

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
            static_assert(
                PreshuffleQuant ||
                    (is_bq_row_major
                         ? (KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                            NPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                         : (NPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                            KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I1{}])),
                "Bq block window has incorrect lengths for defined BqLayout!");

            // Double-Buffering (loop_count=2) for full load/compute overlap.
            const index_t loop_count = 2;

            using ADramTileWindowStep  = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep  = typename BDramBlockWindowTmp::BottomTensorIndex;
            using AQDramTileWindowStep = typename AQDramBlockWindowTmp::BottomTensorIndex;
            using BQDramTileWindowStep = typename BQDramBlockWindowTmp::BottomTensorIndex;

            // Note: BDataType PkInt4 gets converted during loading, before going to LDS
            auto&& [a_lds_block_ping, b_lds_block_ping] =
                Base::template GetABLdsTensorViews<ADataType, OverrideBDataType>(p_smem_0);
            auto&& [a_lds_block_pong, b_lds_block_pong] =
                Base::template GetABLdsTensorViews<ADataType, OverrideBDataType>(p_smem_1);

            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            auto&& [a_copy_dram_window, a_copy_lds_window_ping, a_lds_gemm_window_ping] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block_ping, a_lds_load_tile_distr);
            auto&& [b_copy_dram_window, b_copy_lds_window_ping, b_lds_gemm_window_ping] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block_ping, b_lds_load_tile_distr);

            auto&& [a_copy_dram_window_, a_copy_lds_window_pong, a_lds_gemm_window_pong] =
                Base::GetBWindows(a_dram_block_window_tmp, a_lds_block_pong, a_lds_load_tile_distr);
            auto&& [b_copy_dram_window_, b_copy_lds_window_pong, b_lds_gemm_window_pong] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block_pong, b_lds_load_tile_distr);

            auto aq_copy_dram_window = Base::GetAQDramLoadWindow(aq_dram_block_window_tmp);
            auto bq_copy_dram_window = Base::GetBQDramLoadWindow(bq_dram_block_window_tmp);

            using ABlockTileDistr  = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr  = decltype(b_copy_dram_window.get_tile_distribution());
            using AQBlockTileDistr = decltype(aq_copy_dram_window.get_tile_distribution());
            using BQBlockTileDistr = decltype(bq_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));
            using AQBlockTile =
                decltype(make_static_distributed_tensor<AQDataType>(AQBlockTileDistr{}));
            using BQBlockTile =
                decltype(make_static_distributed_tensor<BQDataType>(BQBlockTileDistr{}));

            auto block_gemm = BlockGemm();

            ABlockTile a_block_tile[2];
            BBlockTile b_block_tile[2];
            AQBlockTile aq_block_tile[2];
            BQBlockTile bq_block_tile[2];
            int currIdx = 0;

            auto c_block_tile = block_gemm.MakeCBlockTile();
            // A B Data
            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            // AQ BQ Data
            const AQDramTileWindowStep aq_dram_tile_window_step =
                PreshuffleQuant
                    ? make_array(ck_tile::integer_least_multiple(m, MPerBlock) /
                                     BlockGemm::WarpGemm::kM,
                                 0)
                    : (is_aq_col_major ? make_array(KPerBlockAQ, 0) : make_array(0, KPerBlockAQ));
            const BQDramTileWindowStep bq_dram_tile_window_step =
                (PreshuffleQuant) ? make_array(ck_tile::integer_least_multiple(n, NPerBlock) /
                                                   BlockGemmShape::WarpTile::at(number<1>{}),
                                               0)
                : is_bq_row_major ? make_array(KPerBlockBQ, 0)
                                  : make_array(0, KPerBlockBQ);

            // 1.1 Load A From Dram to Reg
            LoadAndConvertATile(a_block_tile[currIdx], a_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            LoadAndConvertBTile(b_block_tile[currIdx], b_copy_dram_window);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
            Base::GlobalPrefetch(
                aq_block_tile[currIdx], aq_copy_dram_window, aq_dram_tile_window_step);
            Base::GlobalPrefetch(
                bq_block_tile[currIdx], bq_copy_dram_window, bq_dram_tile_window_step);

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
            // 1.2 Store A Reg To LDS
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tile[currIdx]);
                Base::LocalPrefill(a_copy_lds_window_ping, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window_ping, a_block_tile[currIdx], a_element_func);
            }

            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tile[currIdx]);
                Base::LocalPrefill(b_copy_lds_window_ping, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window_ping, b_block_tile[currIdx], b_element_func);
            }

            // 2.1 Load A From Dram to Reg (2i+1)
            LoadAndConvertATile(a_block_tile[(currIdx + 1) % 2], a_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

            LoadAndConvertBTile(b_block_tile[(currIdx + 1) % 2], b_copy_dram_window);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            Base::GlobalPrefetch(
                aq_block_tile[(currIdx + 1) % 2], aq_copy_dram_window, aq_dram_tile_window_step);
            Base::GlobalPrefetch(
                bq_block_tile[(currIdx + 1) % 2], bq_copy_dram_window, bq_dram_tile_window_step);

            block_sync_lds();
            // 1.3 LDS TO Gemm Reg (2i)
            block_gemm.LocalPrefetch(
                a_lds_gemm_window_ping, b_lds_gemm_window_ping, is_a_load_tr_v, is_b_load_tr_v);

            if constexpr(HasHotLoop)
            {
                index_t iCounter = (num_loop - 1) / loop_count;
                do
                {
                    // 1.4 RUN (2i)
                    block_gemm(c_block_tile,
                               aq_block_tile[currIdx],
                               bq_block_tile[currIdx],
                               a_lds_gemm_window_ping,
                               b_lds_gemm_window_ping);

                    // 2.2 Store A Reg To LDS (2i+1)
                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile[(currIdx + 1) % 2]);
                        Base::LocalPrefill(a_copy_lds_window_pong, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window_pong,
                                           a_block_tile[(currIdx + 1) % 2],
                                           a_element_func);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        // Note: BDataType PkInt4 gets converted during loading earlier
                        auto b_shuffle_tmp = make_static_distributed_tensor<OverrideBDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile[(currIdx + 1) % 2]);
                        Base::LocalPrefill(b_copy_lds_window_pong, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window_pong,
                                           b_block_tile[(currIdx + 1) % 2],
                                           b_element_func);
                    }
                    block_sync_lds(); // don't remove

                    // 2.3 LDS TO Gemm Reg (2i+1)
                    block_gemm.LocalPrefetch(a_lds_gemm_window_pong,
                                             b_lds_gemm_window_pong,
                                             is_a_load_tr_v,
                                             is_b_load_tr_v);

                    // 2.4 RUN (2i+1)
                    block_gemm(c_block_tile,
                               aq_block_tile[(currIdx + 1) % 2],
                               bq_block_tile[(currIdx + 1) % 2],
                               a_lds_gemm_window_pong,
                               b_lds_gemm_window_pong);

                    // 3.1. Load A From Dram to Reg (2i+2)
                    LoadAndConvertATile(a_block_tile[currIdx], a_copy_dram_window);
                    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                    LoadAndConvertBTile(b_block_tile[currIdx], b_copy_dram_window);
                    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
                    Base::GlobalPrefetch(
                        aq_block_tile[currIdx], aq_copy_dram_window, aq_dram_tile_window_step);
                    Base::GlobalPrefetch(
                        bq_block_tile[currIdx], bq_copy_dram_window, bq_dram_tile_window_step);

                    // 3.2 Store A Reg To LDS (2i+2)
                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile[currIdx]);
                        Base::LocalPrefill(a_copy_lds_window_ping, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(
                            a_copy_lds_window_ping, a_block_tile[currIdx], a_element_func);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        // Note: BDataType PkInt4 gets converted during loading earlier
                        auto b_shuffle_tmp = make_static_distributed_tensor<OverrideBDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile[currIdx]);
                        Base::LocalPrefill(b_copy_lds_window_ping, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(
                            b_copy_lds_window_ping, b_block_tile[currIdx], b_element_func);
                    }

                    // block_sync_lds(); // don't remove

                    // 1.1 Load A From Dram to Reg (2i+3)
                    LoadAndConvertATile(a_block_tile[(currIdx + 1) % 2], a_copy_dram_window);
                    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                    LoadAndConvertBTile(b_block_tile[(currIdx + 1) % 2], b_copy_dram_window);
                    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
                    Base::GlobalPrefetch(aq_block_tile[(currIdx + 1) % 2],
                                         aq_copy_dram_window,
                                         aq_dram_tile_window_step);
                    Base::GlobalPrefetch(bq_block_tile[(currIdx + 1) % 2],
                                         bq_copy_dram_window,
                                         bq_dram_tile_window_step);
                    block_sync_lds(); // don't remove
                    // 3.3 LDS TO Gemm Reg (2i+2)
                    block_gemm.LocalPrefetch(a_lds_gemm_window_ping,
                                             b_lds_gemm_window_ping,
                                             is_a_load_tr_v,
                                             is_b_load_tr_v);
                    iCounter--;
                } while(iCounter > 0);
            }
            // tail
            if constexpr((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd))
            {
                // 1.4 RUN (2i)
                block_gemm(c_block_tile,
                           aq_block_tile[currIdx],
                           bq_block_tile[currIdx],
                           a_lds_gemm_window_ping,
                           b_lds_gemm_window_ping);
            }
            else
            {
                // 1.4 RUN (2i)
                block_gemm(c_block_tile,
                           aq_block_tile[currIdx],
                           bq_block_tile[currIdx],
                           a_lds_gemm_window_ping,
                           b_lds_gemm_window_ping);
                // 2.2 Store A Reg To LDS (2i+1)
                if constexpr(is_a_col_major)
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, a_block_tile[(currIdx + 1) % 2]);
                    Base::LocalPrefill(a_copy_lds_window_pong, a_shuffle_tmp, a_element_func);
                }
                else
                {
                    Base::LocalPrefill(
                        a_copy_lds_window_pong, a_block_tile[(currIdx + 1) % 2], a_element_func);
                }
                if constexpr(is_b_row_major)
                {
                    // Note: BDataType gets converted during loading from PkInt4
                    auto b_shuffle_tmp = make_static_distributed_tensor<OverrideBDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, b_block_tile[(currIdx + 1) % 2]);
                    Base::LocalPrefill(b_copy_lds_window_pong, b_shuffle_tmp, b_element_func);
                }
                else
                {
                    Base::LocalPrefill(
                        b_copy_lds_window_pong, b_block_tile[(currIdx + 1) % 2], b_element_func);
                }
                block_sync_lds(); // don't remove
                // 2.3 LDS TO Gemm Reg (2i+1)
                block_gemm.LocalPrefetch(
                    a_lds_gemm_window_pong, b_lds_gemm_window_pong, is_a_load_tr_v, is_b_load_tr_v);

                // 2.4 RUN (2i+1)
                block_gemm(c_block_tile,
                           aq_block_tile[currIdx + 1],
                           bq_block_tile[currIdx + 1],
                           a_lds_gemm_window_pong,
                           b_lds_gemm_window_pong);
            }
            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AQDramBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem_0,
                                   void* p_smem_1,
                                   index_t m = 0,
                                   index_t n = 0) const
    {

        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            aq_dram_block_window_tmp,
            bq_dram_block_window_tmp,
            m,
            n,
            num_loop,
            p_smem_0,
            p_smem_1);
    }
};

} // namespace ck_tile
