// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/moe_gemm/pipeline/moe_gemm_pipeline_agmem_bgmem_creg_flatmm_policy.hpp"
#include <cwchar>

namespace ck_tile {

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct MoeGemmPipelineAgBgCrImpl : public FlatmmPipelineAGmemBGmemCRegV1
{
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

    using GateActivation = remove_cvref_t<typename Problem::Traits::GateActivation>;

    using BlockFlatmm = remove_cvref_t<decltype(PipelinePolicy::template GetBlockFlatmm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr bool IsInputGemm = Problem::Traits::IsInputGemm;
    static constexpr bool IsGateOnly = Problem::Traits::IsGateOnly;
    static constexpr bool IsFusedQuant = Problem::Traits::IsFusedQuant;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    // static constexpr index_t GetVectorSizeA() { return PipelinePolicy::template GetVectorSizeA<Problem>(); }
    // static constexpr index_t GetVectorSizeB() { return PipelinePolicy::template GetVectorSizeB<Problem>(); }
    // static constexpr index_t GetVectorSizeC() { return PipelinePolicy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetVectorSizeA() { return Problem::VectorSizeA; }
    static constexpr index_t GetVectorSizeB() { return Problem::VectorSizeB; }
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }
    static constexpr index_t GetSmemPackA() { return PipelinePolicy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return PipelinePolicy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    // static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto Scheduler  = Problem::Scheduler;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr auto GetADramTileDistribution() {
        return PipelinePolicy::template MakeADramTileDistribution<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return PipelinePolicy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE constexpr static auto GetACoord()
    {
        constexpr auto a_dist = PipelinePolicy::template MakeADramTileDistribution<Problem>();
		return a_dist.calculate_index();
    }

    // get thread coordinate of A in the threadblock
    CK_TILE_HOST_DEVICE constexpr static auto GetAMRepeat()
    {
        constexpr auto a_dist = PipelinePolicy::template MakeADramTileDistribution<Problem>();

		using ADstrEncode = typename decltype(a_dist)::DstrEncode;
		constexpr ck_tile::index_t MRepeat = ADstrEncode::hs_lengthss_[number<0>{}][number<0>{}];
        return MRepeat;
    }

    template <typename ADramBlockWindow, typename BFlatBlockWindowTmp, typename AElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(ADramBlockWindow& a_dram_block_window,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        index_t N,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindow::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindow{}.get_window_lengths()[number<0>{}],
                      "wrong!");
        static_assert(kKPerBlock == ADramBlockWindow{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto config = BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr index_t KFlatPerBlockPerIter = flatKPerWarp;
        constexpr index_t NFlatPerBlockPerIter = flatNPerWarp;

        constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
        constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;


        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        // A LDS tile window for store
        auto a_copy_lds_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        auto block_flatmm = BlockFlatmm();

        // B flat DRAM window for load
        auto b_flat_distribution =
            PipelinePolicy::template MakeBFlatDramTileDistribution<Problem>();
        auto b_gate_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

		move_tile(b_flat_dram_block_window_tmp, {N, 0});
        auto b_up_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        // Acc register tile
        using c_block_tile_type = decltype(block_flatmm(a_lds_gemm_window, b_gate_flat_dram_window));
        auto c_gate_block_tile = c_block_tile_type{};
        auto c_up_block_tile = c_block_tile_type{}

        // prefetch
        // global read 0
		a_block_tile = load_tile(a_dram_block_window);

        statically_indexed_array<
            statically_indexed_array<decltype(b_gate_flat_dram_window), KIterPerWarp>,
            NIterPerWarp>
            b_flat_dram_windows;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_gate_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_up_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_2;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_flat_dram_windows(nIter)(kIter) = b_gate_flat_dram_window;

                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                 {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                b_warp_tensor(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
            });
        });

        {
            // move to 1
            move_tile_window(a_dram_block_window, {0, kKPerBlock});

            // move to next flat K
            move_tile_window(b_gate_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            store_tile(a_copy_lds_window, tile_elementwise_in(a_element_func, a_block_tile));
            block_sync_lds();
        }

        index_t iCounter = num_loop - 1;
        while(iCounter > 0)
        {
            // global read i + 1
            a_block_tile = load_tile(a_dram_block_window);

            // GEMM i
            block_flatmm(c_gate_block_tile, a_warp_windows, b_warp_tensor);

            block_sync_lds();

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_up_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_2(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // move to next flat K
            move_tile_window(b_up_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            // GEMM i
            block_flatmm(c_up_block_tile, a_warp_windows, b_warp_tensor_2);

            block_sync_lds();

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_gate_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // move to next flat K
            move_tile_window(b_gate_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            // LDS write i + 1
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);

            // HotLoopScheduler();
            block_sync_lds();

            iCounter--;
        }

        // tail
        {
            // GEMM i
            block_flatmm(c_gate_block_tile, a_warp_windows, b_warp_tensor);

            block_sync_lds();

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_up_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_2(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // HotLoopScheduler();
            block_sync_lds();

            // GEMM num_loop - 1
            block_flatmm(c_up_block_tile, a_warp_windows, b_warp_tensor_2);
        }

        sweep_tile(c_gate_block_tile,
            [&](auto idx0, auto idx1) {
                fp32x2_t v_{c_gate_block_tile.at(number<0>{})(idx0), c_gate_block_tile.at(number<0>{})(idx1)};
                typename Problem::GateActivation{}(v_, v_);
                c_gate_block_tile.at(number<0>{})(idx0) = v_.x;
                c_gate_block_tile.at(number<0>{})(idx1) = v_.y;
            },
            sequence<1, 2>{});

        auto c_block_tile =
            tile_elementwise_in([&](const auto& a_, const auto& b_) { return a_ * b_; },
                                c_gate_block_tile,
                                c_up_block_tile);

        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(ADramBlockWindow& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t N,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_flat_dram_block_window_tmp,
            N,
            num_loop,
            p_smem);
    }

};
} // namespace ck_tile

