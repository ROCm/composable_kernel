// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

template <typename Problem, typename Policy>
struct GridGemm
{
    using ADataType        = typename Problem::ADataType;
    using BDataType        = typename Problem::BDataType;
    using CDataType        = typename Problem::CDataType;
    using WeightType       = typename Problem::WeightType;
    using IndexType        = typename Problem::IndexType;
    using AccDataType      = typename Problem::AccDataType;
    using ComputeDataType  = float;
    using CElementFunction = typename Problem::CElementFunction;

    static constexpr auto kMPerBlock = Policy::kMPerBlock;
    static constexpr auto kNPerBlock = Policy::kNPerBlock;
    static constexpr auto kKPerBlock = Policy::kKPerBlock;
    static constexpr auto topk = Policy::kTopKPerBlock;
    static constexpr auto kBlockSize = Policy::kBlockSize;

    template <typename AGridTensorView, typename BGridTensorView, typename DebugGridTensorView, typename ValueGridTensorView, typename IndexGridTensorView>
    CK_TILE_DEVICE void operator()(const AGridTensorView& a_grid,
                                   const BGridTensorView& b_grid,
                                   DebugGridTensorView& debug_grid,
                                   ValueGridTensorView& value_grid,
                                   IndexGridTensorView& index_grid,
                                   const CElementFunction& c_element_func) const
    {
        const auto M = a_grid.get_tensor_descriptor().get_length(number<0>{});
        const auto N = b_grid.get_tensor_descriptor().get_length(number<0>{});
        const auto K = a_grid.get_tensor_descriptor().get_length(number<1>{});

        // divide problem
        const auto id_block = get_block_id();

        const auto num_tile_m = integer_divide_ceil(M, kMPerBlock);
        const auto num_tile_n = integer_divide_ceil(N, kNPerBlock);

        const auto block2tile = Policy::template MakeBlock2TileMap<Problem>(num_tile_m, num_tile_n);

        const auto id_tile = block2tile(id_block);

        const auto iM =
            __builtin_amdgcn_readfirstlane(id_tile.template get(number<0>{}) * kMPerBlock);
        const auto iN =
            __builtin_amdgcn_readfirstlane(id_tile.template get(number<1>{}) * kNPerBlock);

        // A block window
        auto a_block_window = make_tile_window(
            a_grid, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {iM, 0});

        // B block window
        auto b_block_window = make_tile_window(
            b_grid, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {iN, 0});

        constexpr auto block_gemm_pipeline = Policy::template GetBlockGemmPipeline<Problem>();

        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];

        // // store C
        // auto c_window = make_tile_window(
        //     c_grid, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, iN});

        // store value and index
        // constexpr index_t kBlockSize = Problem::kBlockSize;
        // constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        // constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(WeightType);
        constexpr index_t K0 = topk / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        auto value_window = make_tile_window(
            value_grid, make_tuple(number<kMPerBlock>{}, number<topk>{}), {iM, iN},
            make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                    tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                    tuple<sequence<1>, sequence<1, 2>>,
                                    tuple<sequence<1>, sequence<2, 0>>,
                                    sequence<1, 2>,
                                    sequence<0, 1>>{}));
        auto index_window = make_tile_window(
            index_grid, make_tuple(number<kMPerBlock>{}, number<topk>{}), {iM, iN},
            make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                    tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                    tuple<sequence<1>, sequence<1, 2>>,
                                    tuple<sequence<1>, sequence<2, 0>>,
                                    sequence<1, 2>,
                                    sequence<0, 1>>{}));

        using ValueBlockTileDistr = decltype(value_window.get_tile_distribution());
        using IndexBlockTileDistr = decltype(index_window.get_tile_distribution());

        using ValueBlockTile = decltype(make_static_distributed_tensor<WeightType>(ValueBlockTileDistr{}));
        using IndexBlockTile = decltype(make_static_distributed_tensor<IndexType>(IndexBlockTileDistr{}));

        ValueBlockTile value_block_tile;
        IndexBlockTile index_block_tile;

        // Initialize value_block_tile and index_block_tile
        tile_elementwise_inout([](auto& value) { value = 0; }, value_block_tile);
        tile_elementwise_inout([](auto& index) { index = 0; }, index_block_tile);

        // constexpr index_t debugK1 = 16 / sizeof(WeightType);
        // constexpr index_t debugK0 = kNPerBlock / debugK1;
        // constexpr index_t debugM2 = get_warp_size() / debugK0;
        // // coalesce reading for each blocks
        // constexpr index_t debugM1 = kBlockSize / get_warp_size();
        // constexpr index_t debugM0 = kMPerBlock / (debugM2 * debugM1);

        auto debug_window = make_tile_window(
            debug_grid, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, iN});

        // auto debug_window = make_tile_window(
        //     debug_grid, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, iN},
        //     make_static_tile_distribution(
        //     tile_distribution_encoding<sequence<1>,
        //                             tuple<sequence<debugM0, debugM1, debugM2>, sequence<debugK0, debugK1>>,
        //                             tuple<sequence<1>, sequence<1, 2>>,
        //                             tuple<sequence<1>, sequence<2, 0>>,
        //                             sequence<1, 2>,
        //                             sequence<0, 1>>{}));
        
        // using DebugBlockTileDistr = decltype(debug_window.get_tile_distribution());
        // using DebugBlockTile = decltype(make_static_distributed_tensor<WeightType>(DebugBlockTileDistr{}));
        // DebugBlockTile debug_block_tile;
        // tile_elementwise_inout([](auto& debug) { debug = 0; }, debug_block_tile);

        // block_gemm_pipeline(a_block_window, b_block_window, debug_block_tile, value_block_tile, index_block_tile, K / kKPerBlock, p_smem_char);
        const auto debug_block_tile = block_gemm_pipeline(a_block_window, b_block_window, K / kKPerBlock, p_smem_char);
        // block_gemm_pipeline(a_block_window, b_block_window, debug_block_tile, K / kKPerBlock, p_smem_char);

        // cast DataType and apply CElementFunction
        const auto debug_cast_block_tile = tile_elementwise_in(
            [&](const auto& debug) { return c_element_func(type_convert<WeightType>(debug)); },
            debug_block_tile);

        const auto value_cast_block_tile = tile_elementwise_in(
            [&](const auto& value) { return c_element_func(type_convert<WeightType>(value)); },
            value_block_tile);

        const auto index_cast_block_tile = tile_elementwise_in(
            [&](const auto& index) { return c_element_func(type_convert<IndexType>(index)); },
            index_block_tile);


        store_tile(debug_window, debug_cast_block_tile);
        store_tile(value_window, value_cast_block_tile);
        store_tile(index_window, index_cast_block_tile);
    }
};

} // namespace ck_tile
