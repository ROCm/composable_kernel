// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_skip_b_lds_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = UniversalGemmPipelineAgBgCrSkipBLdsPolicy>
struct GemmPipelineAgBgCrMemSkipBLds : public BaseGemmPipelineAgBgCrMem<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrMem<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

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

    // Where is the right place for HasHotLoop and TailNum ???
    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrMemSkipBLds",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    using Base::PrefetchStages;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
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

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles
            auto a_lds_block = Base::GetALdsTensorView(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto b_lds_load_tile_distr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto a_windows =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto& a_copy_dram_window = a_windows.at(I0{});
            auto& a_copy_lds_window  = a_windows.at(I1{});
            auto& a_lds_gemm_window  = a_windows.at(I2{});

            // B DRAM tile window for load
            auto b_copy_dram_window =
                Base::GetBDramWindowSkipLds(b_dram_block_window_tmp, b_lds_load_tile_distr);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            tuple_array<ABlockTile, PrefetchStages> a_block_tiles;
            tuple_array<BBlockTile, PrefetchStages> b_block_tiles;

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            Base::GlobalPrefetch(
                a_block_tiles.get(I0{}), a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(
                b_block_tiles.get(I0{}), b_copy_dram_window, b_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tiles.get(I0{}));
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tiles.get(I0{}), a_element_func);
            }

            // TODO add b rowmajor support
            static_assert(is_b_row_major == false, "BRowMajor not supported yet!");
            tile_elementwise_inout(b_element_func, b_block_tiles.get(I0{}));

            // Global prefetch [1, PrefetchStages]
            static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}),
                                     a_copy_dram_window,
                                     a_dram_tile_window_step);
                Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                     b_copy_dram_window,
                                     b_dram_tile_window_step);
            });

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds();
                        block_gemm.LocalPrefetch(a_lds_gemm_window);
                        block_gemm(c_block_tile,
                                   a_lds_gemm_window,
                                   b_block_tiles.get(number<prefetch_idx>{}));

                        block_sync_lds();

                        if constexpr(is_a_col_major)
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(
                                a_shuffle_tmp,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                            Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                                a_element_func);
                        }
                        tile_elementwise_inout(
                            b_element_func,
                            b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                        // TODO add b rowmajor support

                        Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}),
                                             a_copy_dram_window,
                                             a_dram_tile_window_step);
                        Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                             b_copy_dram_window,
                                             b_dram_tile_window_step);
                    });

                    i += PrefetchStages;
                } while(i < (num_loop - PrefetchStages));
            }

            auto HotLoopTail = [&](auto tail_num) {
                static_for<1, tail_num, 1>{}([&](auto prefetch_idx) {
                    block_sync_lds();

                    block_gemm.LocalPrefetch(a_lds_gemm_window);
                    block_gemm(c_block_tile, a_lds_gemm_window, b_block_tiles.get(number<0>{}));

                    block_sync_lds();

                    if constexpr(is_a_col_major)
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tiles.get(number<prefetch_idx>{}));
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window,
                                           a_block_tiles.get(number<prefetch_idx>{}),
                                           a_element_func);
                    }

                    // TODO add b rowmajor support
                    tile_elementwise_inout(b_element_func,
                                           b_block_tiles.get(number<prefetch_idx>{}));
                });

                block_sync_lds();
                block_gemm.LocalPrefetch(a_lds_gemm_window);
                block_gemm(
                    c_block_tile, a_lds_gemm_window, b_block_tiles.get(number<tail_num - 1>{}));
            };

            if constexpr(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm.LocalPrefetch(a_lds_gemm_window);
                block_gemm(c_block_tile, a_lds_gemm_window, b_block_tiles.get(number<0>{}));
            }
            else if constexpr(TailNum == TailNumber::Two)
            {
                HotLoopTail(number<2>{});
            }
            else if constexpr(TailNum == TailNumber::Three)
            {
                HotLoopTail(number<3>{});
            }
            else if constexpr(TailNum == TailNumber::Four)
            {
                HotLoopTail(number<4>{});
            }
            else if constexpr(TailNum == TailNumber::Five)
            {
                HotLoopTail(number<5>{});
            }
            else if constexpr(TailNum == TailNumber::Six)
            {
                HotLoopTail(number<6>{});
            }
            else if constexpr(TailNum == TailNumber::Seven)
            {
                HotLoopTail(number<7>{});
            }
            else if constexpr(TailNum == TailNumber::Full)
            {
                HotLoopTail(number<PrefetchStages>{});
            }

            return c_block_tile;
        }
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Interwave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

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

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles
            auto a_lds_block = Base::GetALdsTensorView(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto b_lds_load_tile_distr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto a_windows =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto& a_copy_dram_window = a_windows.at(I0{});
            auto& a_copy_lds_window  = a_windows.at(I1{});
            auto& a_lds_gemm_window  = a_windows.at(I2{});

            // B DRAM tile window for load
            auto b_copy_dram_window =
                Base::GetBDramWindowSkipLds(b_dram_block_window_tmp, b_lds_load_tile_distr);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            tuple_array<ABlockTile, PrefetchStages> a_block_tiles;
            tuple_array<BBlockTile, PrefetchStages> b_block_tiles;

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            Base::GlobalPrefetch(
                a_block_tiles.get(I0{}), a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(
                b_block_tiles.get(I0{}), b_copy_dram_window, b_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tiles.get(I0{}));
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tiles.get(I0{}), a_element_func);
            }
            // TODO add b rowmajor support
            static_assert(is_b_row_major == false, "BRowMajor not supported yet!");
            tile_elementwise_inout(b_element_func, b_block_tiles.get(I0{}));

            // Global prefetch [1, PrefetchStages]
            static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}),
                                     a_copy_dram_window,
                                     a_dram_tile_window_step);
                Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                     b_copy_dram_window,
                                     b_dram_tile_window_step);
            });

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds();
                        block_gemm(c_block_tile,
                                   a_lds_gemm_window,
                                   b_block_tiles.get(number<prefetch_idx>{}));
                        // no second block_sync_lds because it's interwave

                        if constexpr(is_a_col_major)
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(
                                a_shuffle_tmp,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                            Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                                a_element_func);
                        }
                        // TODO add b rowmajor support
                        tile_elementwise_inout(
                            b_element_func,
                            b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));

                        Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}),
                                             a_copy_dram_window,
                                             a_dram_tile_window_step);
                        Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                             b_copy_dram_window,
                                             b_dram_tile_window_step);
                    });

                    i += PrefetchStages;
                } while(i < (num_loop - PrefetchStages));
            }

            auto HotLoopTail = [&](auto tail_num) {
                static_for<1, tail_num, 1>{}([&](auto prefetch_idx) {
                    block_sync_lds();
                    block_gemm(c_block_tile,
                               a_lds_gemm_window,
                               b_block_tiles.get(number<prefetch_idx - 1>{}));
                    // no second block_sync_lds because it's interwave

                    if constexpr(is_a_col_major)
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tiles.get(number<prefetch_idx>{}));
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window,
                                           a_block_tiles.get(number<prefetch_idx>{}),
                                           a_element_func);
                    }
                    // TODO add b rowmajor support
                    tile_elementwise_inout(b_element_func,
                                           b_block_tiles.get(number<prefetch_idx>{}));
                });

                block_sync_lds();
                block_gemm(
                    c_block_tile, a_lds_gemm_window, b_block_tiles.get(number<tail_num - 1>{}));
            };

            if constexpr(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm(c_block_tile, a_lds_gemm_window, b_block_tiles.get(number<0>{}));
            }
            else if constexpr(TailNum == TailNumber::Two)
            {
                HotLoopTail(number<2>{});
            }
            else if constexpr(TailNum == TailNumber::Three)
            {
                HotLoopTail(number<3>{});
            }
            else if constexpr(TailNum == TailNumber::Four)
            {
                HotLoopTail(number<4>{});
            }
            else if constexpr(TailNum == TailNumber::Five)
            {
                HotLoopTail(number<5>{});
            }
            else if constexpr(TailNum == TailNumber::Six)
            {
                HotLoopTail(number<6>{});
            }
            else if constexpr(TailNum == TailNumber::Seven)
            {
                HotLoopTail(number<7>{});
            }
            else if constexpr(TailNum == TailNumber::Full)
            {
                HotLoopTail(number<PrefetchStages>{});
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
