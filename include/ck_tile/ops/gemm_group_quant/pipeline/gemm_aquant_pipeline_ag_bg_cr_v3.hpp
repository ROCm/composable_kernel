// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <typename Problem>
struct BaseAQuantGemmPipelineAgBgCrCompV3 : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        if(has_hot_loop)
        {
            if(tail_number == ck_tile::TailNumber::Full)
            {
                return run_func(
                    ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_number == ck_tile::TailNumber::Odd)
            {
                return run_func(
                    ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_number == ck_tile::TailNumber::Even)
            {
                return run_func(
                    ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
            }
            else
            {
                throw std::runtime_error("Unsupported tail number for this operation !!!");
            }
        }
        else
        {
            if(tail_number == ck_tile::TailNumber::Full)
            {
                return run_func(
                    ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_number == ck_tile::TailNumber::Odd)
            {
                return run_func(
                    ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_number == ck_tile::TailNumber::Even)
            {
                return run_func(
                    ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
            }
            else
            {
                throw std::runtime_error("Unsupported tail number for this operation !!!");
            }
        }
    }
};

template <typename Problem, typename Policy = GemmAQuantPipelineAgBgCrDefaultPolicy>
struct AQuantGemmPipelineAgBgCrCompV3 : public BaseAQuantGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmAQuantPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using AQDataType     = remove_cvref_t<typename Problem::AQDataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t AQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<AQDataType>>::PackedSize;

    using ALayout  = remove_cvref_t<typename Problem::ALayout>;
    using AQLayout = remove_cvref_t<typename Problem::AQLayout>;
    using BLayout  = remove_cvref_t<typename Problem::BLayout>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    static constexpr index_t BlockSize      = Problem::kBlockSize;
    static constexpr index_t MPerBlock      = BlockGemmShape::kM;
    static constexpr index_t NPerBlock      = BlockGemmShape::kN;
    static constexpr index_t KPerBlock      = BlockGemmShape::kK;
    static constexpr index_t QuantGroupSize = Problem::kQuantGroupSize;
    static constexpr index_t KPerBlockAQ    = BlockGemmShape::kK / QuantGroupSize;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }
    static constexpr index_t GetVectorSizeAQ()
    {
        return Policy::template GetVectorSizeAQ<Problem>();
    }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    using Base::PrefetchStages;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "aquant_pipeline_AgBgCrCompV3", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', BlockGemm::WarpGemm::kM, BlockGemm::WarpGemm::kN, BlockGemm::WarpGemm::kK),
                      concat('x', kPadM, kPadN, kPadK), "QuantGroupSize", QuantGroupSize);
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
            << "A/B LDS read/write width: " << A_LDS_Read_Width << ", " << B_LDS_Read_Width << "\n"
            << "A/B buffer load inst: " << A_Buffer_Load_Inst_Num << ", " << B_Buffer_Load_Inst_Num
            << ", " << "AQ buffer load inst: " << AQ_Buffer_Load_Inst_Num << "\n"
            << "A/B LDS write inst: " << A_LDS_Write_Inst_Num << ", " << B_LDS_Write_Inst_Num
            << "\n"
            << "A/B LDS read inst: " << A_LDS_Read_Inst_Num << ", " << B_LDS_Read_Inst_Num << "\n"
            << "C MFMA inst: " << C_MFMA_Inst_Num << "\n"
            << "QuantGroupSize: " << QuantGroupSize << "\n"
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

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AQDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<AQDataType,
                                   remove_cvref_t<typename AQDramBlockWindowTmp::DataType>>,
                "A/B/AQ Dram block window should have the same data type as appropriate "
                "([A|B|AQ]DataType) defined in Problem definition!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_aq_col_major =
                std::is_same_v<AQLayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(!is_aq_col_major, "Aq must be row major (col major not supported yet)");
            static_assert(MPerBlock == AQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                              KPerBlockAQ == AQDramBlockWindowTmp{}.get_window_lengths()[I1{}],
                          "Aq block window has incorrect lengths for defined AqLayout!");

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

            using ADramTileWindowStep  = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep  = typename BDramBlockWindowTmp::BottomTensorIndex;
            using AQDramTileWindowStep = typename AQDramBlockWindowTmp::BottomTensorIndex;

            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);
            auto aq_copy_dram_window = Base::GetAQDramLoadWindow(aq_dram_block_window_tmp);

            using ABlockTileDistr  = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr  = decltype(b_copy_dram_window.get_tile_distribution());
            using AQBlockTileDistr = decltype(aq_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));
            using AQBlockTile =
                decltype(make_static_distributed_tensor<AQDataType>(AQBlockTileDistr{}));

            auto block_gemm = BlockGemm();

            ABlockTile a_block_tile;
            BBlockTile b_block_tile;
            AQBlockTile aq_block_tile[2];
            int currIdx = 0;

            auto c_block_tile = block_gemm.MakeCBlockTile();

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr AQDramTileWindowStep aq_dram_tile_window_step =
                is_aq_col_major ? make_array(KPerBlockAQ, 0) : make_array(0, KPerBlockAQ);

            // DRAM prefetch (global read 0)
            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);
            Base::GlobalPrefetch(
                aq_block_tile[currIdx], aq_copy_dram_window, aq_dram_tile_window_step);

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffled2DStaticTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tile);
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
            }

            if constexpr(is_b_row_major)
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffled2DStaticTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tile);
                Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
            }

            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();

            block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);

            __builtin_amdgcn_sched_barrier(0);

            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    block_sync_lds();

                    if constexpr(is_a_col_major)
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                    }
                    if constexpr(is_b_row_major)
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
                    }

                    Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
                    Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);
                    Base::GlobalPrefetch(aq_block_tile[(currIdx + 1) % 2],
                                         aq_copy_dram_window,
                                         aq_dram_tile_window_step);

                    block_gemm(
                        c_block_tile, aq_block_tile[currIdx], a_lds_gemm_window, b_lds_gemm_window);

                    currIdx = (currIdx + 1) % 2;

                    block_sync_lds();

                    block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                    __builtin_amdgcn_sched_barrier(0);

                    i += 1;
                } while(i < (num_loop - 1));
            }
            // tail
            if constexpr((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd))
            {
                block_gemm(
                    c_block_tile, aq_block_tile[currIdx], a_lds_gemm_window, b_lds_gemm_window);
            }
            else
            {
                Base::GlobalPrefetch(aq_block_tile[(currIdx + 1) % 2],
                                     aq_copy_dram_window,
                                     aq_dram_tile_window_step);
                block_gemm(
                    c_block_tile, aq_block_tile[currIdx], a_lds_gemm_window, b_lds_gemm_window);
                block_sync_lds();

                currIdx = (currIdx + 1) % 2;

                if constexpr(is_a_col_major)
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, a_block_tile);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                }
                if constexpr(is_b_row_major)
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, b_block_tile);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
                }
                block_sync_lds();
                block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                block_gemm(
                    c_block_tile, aq_block_tile[currIdx], a_lds_gemm_window, b_lds_gemm_window);
            }
            return c_block_tile;
        }
    };
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            aq_dram_block_window_tmp,
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
