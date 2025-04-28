// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/moe_gemm/pipeline/moe_gemm_pipeline_agmem_bgmem_creg_flatmm_policy.hpp"
#include <cwchar>

namespace ck_tile {

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct MoeGemmPipelineAgBgCrImpl
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
        auto b_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        // Acc register tile
        auto c_block_tile = decltype(block_flatmm(a_lds_gemm_window, b_flat_dram_window)){};

        // prefetch
        // global read 0
        auto a_block_tile = a_dram_block_window.load();

        {
            // move to 1
            move_tile_window(a_dram_block_window, {0, kKPerBlock});

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    PipelinePolicy::template MakeShuffledARegBlockDistribution<Problem>());
                shuffle_tile(a_shuffle_tmp, a_block_tile);
                const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_shuffle_tmp);
                store_tile(a_copy_lds_window, a_block_tile_tmp);
            }
            else
            {
                store_tile(a_copy_lds_window, tile_elementwise_in(a_element_func, a_block_tile));
            }
        }

        index_t iCounter = num_loop - 1;
        while(iCounter > 0)
        {
            // global read i + 1
            a_dram_block_window.load(a_block_tile);

            block_sync_lds();

            // GEMM i
            block_flatmm(c_block_tile, a_lds_gemm_window, b_flat_dram_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(a_dram_block_window, {0, kKPerBlock});

            // LDS write i + 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);

            // move to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            iCounter--;
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 1
            block_flatmm(c_block_tile, a_lds_gemm_window, b_flat_dram_window);
        }

        sweep_tile(c_block_tile,
            [&](auto idx0, auto idx1) {
                fp32x2_t v_{c_block_tile(idx0), c_block_tile(idx1)};
                GateActivation{}(v_, v_);
                c_block_tile(idx0) = v_.x;
                c_block_tile(idx1) = v_.y;
            },
            sequence<1, 2>{});

        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(ADramBlockWindow& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_flat_dram_block_window_tmp,
            num_loop,
            p_smem);
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

        auto b_gate_flat_dram_window =
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        b_flat_dram_block_window_tmp.move({N, 0})
        auto b_up_flat_dram_window =
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        using c_block_tile_type = decltype(block_flatmm(a_lds_gemm_window, b_gate_flat_dram_window));
        auto c_block_tiles[2] = {c_block_tile_type{}, c_block_tile_type{}};

        // prefetch
        // global read 0
        auto a_block_tile = a_dram_block_window.load();

        {
            // move to 1
            move_tile_window(a_dram_block_window, {0, kKPerBlock});

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tiles[0]);
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tiles[1]);

            // LDS write 0
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    PipelinePolicy::template MakeShuffledARegBlockDistribution<Problem>());
                shuffle_tile(a_shuffle_tmp, a_block_tile);
                const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_shuffle_tmp);
                store_tile(a_copy_lds_window, a_block_tile_tmp);
            }
            else
            {
                store_tile(a_copy_lds_window, tile_elementwise_in(a_element_func, a_block_tile));
            }
        }

        index_t iCounter = num_loop - 1;
        while(iCounter > 0)
        {
            // global read i + 1
            a_dram_block_window.load(a_block_tile);

            block_sync_lds();

            // GEMM i
            block_flatmm(c_block_tiles[0], a_lds_gemm_window, b_gate_flat_dram_window);

            //TODO: simply add b_gate flatmm
            block_flatmm(c_block_tiles[1], a_lds_gemm_window, b_up_flat_dram_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(a_dram_block_window, {0, kKPerBlock});

            // LDS write i + 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);

            // move to next flat K
            move_tile_window(b_gate_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});
            move_tile_window(b_up_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            iCounter--;
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 1
            block_flatmm(c_block_tiles[0], a_lds_gemm_window, b_gate_flat_dram_window);
            block_flatmm(c_block_tiles[1], a_lds_gemm_window, b_up_flat_dram_window);
        }

        sweep_tile(c_block_tiles[0],
            [&](auto idx0, auto idx1) {
                fp32x2_t v_{c_block_tiles[0].at(number<0>{})(idx0), c_block_tiles[0].at(number<0>{})(idx1)};
                typename Problem::GateActivation{}(v_, v_);
                c_block_tiles[0].at(number<0>{})(idx0) = v_.x;
                c_block_tiles[0].at(number<0>{})(idx1) = v_.y;
            },
            sequence<1, 2>{});

        auto c_block_tile =
            tile_elementwise_in([&](const auto& a_, const auto& b_) { return a_ * b_; },
                                c_block_tiles[0],
                                c_block_tiles[1]);

        return c_block_tiles[0];
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

