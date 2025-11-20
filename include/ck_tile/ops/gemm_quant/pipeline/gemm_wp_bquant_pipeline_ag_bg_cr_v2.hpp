// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename Problem, typename PipelinePolicy = GemmWPQuantPipelineAgBgCrPolicy>
struct WPQuantBPipelineAgBgCrV2 : public WeightPreshufflePipelineAGmemBGmemCRegV2<Problem>
{
    using Base            = WeightPreshufflePipelineAGmemBGmemCRegV2<Problem>;
    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
    using QuantGroupSize  = remove_cvref_t<typename Problem::QuantGroupSize>;

    using ALayout  = remove_cvref_t<typename Problem::ALayout>;
    using BLayout  = remove_cvref_t<typename Problem::BLayout>;
    using BQLayout = remove_cvref_t<typename Problem::BQLayout>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using BlockWeightPreshuffle = remove_cvref_t<
        decltype(PipelinePolicy::template GetBlockWeightPreshuffleBQuant<Problem>())>;

    static constexpr auto config =
        BlockWeightPreshuffle::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    using Base::kKPerBlock;
    using Base::kMPerBlock;
    using Base::kNPerBlock;

    using Base::KIterPerWarp;
    using Base::MIterPerWarp;
    using Base::NIterPerWarp;

    using Base::BlockSize;

    using Base::kPadK;
    using Base::kPadM;
    using Base::kPadN;

    using Base::I0;
    using Base::I1;
    using Base::I2;

    using Base::MWarp;
    using Base::NWarp;

    using Base::KPerBlockPerIter;
    using Base::MPerBlockPerIter;

    using Base::flatKPerWarp;
    using Base::flatNPerWarp;

    using Base::m_preload;

    static constexpr index_t KPerBlockBQ =
        integer_divide_ceil(BlockGemmShape::kK, QuantGroupSize::kK);
    static constexpr index_t QScalesPerBlockRow =
        integer_divide_ceil(kKPerBlock, QuantGroupSize::kK);

    static constexpr index_t GetVectorSizeBQ()
    {
        return PipelinePolicy::template GetVectorSizeBQ<Problem>();
    }
    static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0);
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1);
        return concat('_', "bquant_pipeline_AgBgCrV2_preshuffleB", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock),
                      BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', Base::GetVectorSizeA(), Base::GetVectorSizeB(), GetVectorSizeBQ()),
                      concat('x', kPadM, kPadN, kPadK), QuantGroupSize::GetName());
        // clang-format on
    }

    static constexpr bool PreshuffleB = Problem::PreshuffleB;
    static constexpr auto TailNum     = Problem::TailNum;

    template <TailNumber TailNum,
              typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename BQDramBlockWindowTmp,
              typename AElementFunction,
              index_t UnaryOpSize_ = 8>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BFlatBlockWindowTmp::DataType>> &&
                std::is_same_v<BQDataType, remove_cvref_t<typename BQDramBlockWindowTmp::DataType>>,
            "A/B/BQ Dram block window should have the same data type as appropriate "
            "([A|B|BQ]DataType) defined in Problem definition!");

        constexpr bool is_a_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        static_assert(!is_a_col_major, "A must be row major (col major not supported yet)");

        constexpr bool is_bq_col_major = std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>;
        static_assert(is_bq_col_major, "Bq must be col major (row major not supported yet)");

        constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        static_assert(!is_b_row_major, "B must be col major (row major not supported yet)");

        const index_t iMWarp = get_warp_id() / NWarp;

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

        using BTypeToUse =
            std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
        using BTileType = decltype(make_static_distributed_tensor<BTypeToUse>(b_flat_distribution));

        // pingpong buffer for B
        statically_indexed_array<
            statically_indexed_array<decltype(b_flat_dram_window), KIterPerWarp>,
            NIterPerWarp>
            b_flat_dram_windows;

        statically_indexed_array<statically_indexed_array<BTileType, KIterPerWarp>, NIterPerWarp>
            b_warp_tensor_ping;

        statically_indexed_array<statically_indexed_array<BTileType, KIterPerWarp>, NIterPerWarp>
            b_warp_tensor_pong;

        // BQ DRAM window for load
        auto bq_copy_dram_window =
            make_tile_window(bq_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<KPerBlockBQ>{}, number<kNPerBlock>{}),
                             bq_dram_block_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeBQDramTileDistribution<Problem>());

        // Prefetch A0
        auto a_block_tile = load_tile(a_copy_dram_window);
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        // prefetch B
        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                 {nIter * flatNPerWarp, kIter * flatKPerWarp});

                load_int4_tile<BDataType, ADataType, UnaryOpSize_>(
                    b_warp_tensor_ping(nIter)(kIter), b_flat_dram_windows(nIter)(kIter));
            });
        });
        // move B window to next flat K
        move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

        // Strictly not needed given type deduction, but helps with readability
        using BQBlockTileDistr = decltype(bq_copy_dram_window.get_tile_distribution());
        using BQBlockTile =
            decltype(make_static_distributed_tensor<BQDataType>(BQBlockTileDistr{}));

        // Load tile 0 for BQ data directly into registers for block tile
        BQBlockTile bq_block_tile, bq_block_tile_2;
        bq_block_tile = load_tile(bq_copy_dram_window);
        // move BQ to tile 1
        move_tile_window(bq_copy_dram_window, {KPerBlockBQ, 0});

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
        statically_indexed_array<decltype(load_tile(a_warp_windows_ping(number<0>{})(number<0>{}))),
                                 m_preload>
            a_warp_tensor;

        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MIterPerWarp;
            constexpr auto kIter = loadIter / MIterPerWarp;
            a_warp_tensor(loadIter) =
                load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
        });
        __builtin_amdgcn_sched_barrier(0);

        // MAIN LOOP
        index_t iCounter = (num_loop - 1) / 2;
        while(iCounter > 0)
        {
            // prefetch B(2i+1)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * flatNPerWarp, kIter * flatKPerWarp});
                    load_int4_tile<BDataType, ADataType, UnaryOpSize_>(
                        b_warp_tensor_pong(nIter)(kIter), b_flat_dram_windows(nIter)(kIter));
                });
            });
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            bq_block_tile_2 = load_tile(bq_copy_dram_window);
            move_tile_window(bq_copy_dram_window, {KPerBlockBQ, 0});

            // Prefill A(2i+1)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_pong, a_block_tile_tmp);

            // Prefetch A(2i+2)
            a_block_tile = load_tile(a_copy_dram_window);
            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // GEMM 2i
            block_weight_preshuffle(c_block_tile,
                                    a_warp_tensor,
                                    b_warp_tensor_ping,
                                    bq_block_tile,
                                    a_warp_windows_ping);

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });
            Base::HotLoopScheduler();

            // Next K

            // prefetch B(2i+2)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * flatNPerWarp, kIter * flatKPerWarp});
                    load_int4_tile<BDataType, ADataType, UnaryOpSize_>(
                        b_warp_tensor_ping(nIter)(kIter), b_flat_dram_windows(nIter)(kIter));
                });
            });
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            bq_block_tile = load_tile(bq_copy_dram_window);
            move_tile_window(bq_copy_dram_window, {KPerBlockBQ, 0});

            // Prefill A(2i+2)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_ping, a_block_tile_tmp);

            // Prefetch A(2i+3)
            a_block_tile = load_tile(a_copy_dram_window);
            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // GEMM 2i+1
            block_weight_preshuffle(c_block_tile,
                                    a_warp_tensor,
                                    b_warp_tensor_pong,
                                    bq_block_tile_2,
                                    a_warp_windows_pong);

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor(loadIter) =
                    load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
            });
            Base::HotLoopScheduler();

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
                                     {nIter * flatNPerWarp, kIter * flatKPerWarp});

                    load_int4_tile<BDataType, ADataType, UnaryOpSize_>(
                        b_warp_tensor_pong(nIter)(kIter), b_flat_dram_windows(nIter)(kIter));
                });
            });
            bq_block_tile_2 = load_tile(bq_copy_dram_window);

            // Prefill A(loopK)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_pong, a_block_tile_tmp);

            // GEMM loopK-1
            block_weight_preshuffle(c_block_tile,
                                    a_warp_tensor,
                                    b_warp_tensor_ping,
                                    bq_block_tile,
                                    a_warp_windows_ping);

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });

            Base::Last2ndHotLoopScheduler();

            // GEMM loopK
            block_weight_preshuffle(c_block_tile,
                                    a_warp_tensor,
                                    b_warp_tensor_pong,
                                    bq_block_tile_2,
                                    a_warp_windows_pong);
            Base::LastHotLoopScheduler();
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            // GEMM loopK
            block_weight_preshuffle(c_block_tile,
                                    a_warp_tensor,
                                    b_warp_tensor_ping,
                                    bq_block_tile,
                                    a_warp_windows_ping);
            Base::LastHotLoopScheduler();
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {

        return operator()<TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_flat_dram_block_window_tmp,
            bq_dram_block_window_tmp,
            num_loop,
            p_smem_ping,
            p_smem_pong);
    }

    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   TailNumber tail_number,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        const auto RunPipeline = [&](auto bool_val, auto tail_num_) {
            (void)bool_val; // Suppress unused parameter warning
            constexpr auto tail_num = tail_num_.value;
            return operator()<tail_num>(
                a_dram_block_window_tmp,
                [](const ADataType& a) { return a; },
                b_flat_dram_block_window_tmp,
                bq_dram_block_window_tmp,
                num_loop,
                p_smem_ping,
                p_smem_pong);
        };
        return Base::TailHandler(RunPipeline, true, tail_number);
    }
};
} // namespace ck_tile
