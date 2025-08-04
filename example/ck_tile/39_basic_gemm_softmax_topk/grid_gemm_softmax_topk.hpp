// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

template <typename Problem, typename Policy>
struct GridGemmSoftmaxTopk
{
    using ADataType        = typename Problem::ADataType;
    using BDataType        = typename Problem::BDataType;
    using WeightType       = typename Problem::WeightType;
    using IndexType        = typename Problem::IndexType;
    using AccDataType      = typename Problem::AccDataType;
    // using CElementFunction = typename Problem::CElementFunction;

    static constexpr auto kMPerBlock = Policy::kMPerBlock;
    static constexpr auto kNPerBlock = Policy::kNPerBlock;
    static constexpr auto kKPerBlock = Policy::kKPerBlock;
    static constexpr auto topk = Policy::kTopKPerBlock;

    template <typename AGridTensorView, typename BGridTensorView, typename ValueGridTensorView, typename IndexGridTensorView>
    CK_TILE_DEVICE void operator()(const AGridTensorView& a_grid,
                                   const BGridTensorView& b_grid,
                                   ValueGridTensorView& value_grid,
                                   IndexGridTensorView& index_grid) const
                                //    const CElementFunction& c_element_func) const
    {
        const auto M = a_grid.get_tensor_descriptor().get_length(number<0>{});
        const auto N = b_grid.get_tensor_descriptor().get_length(number<0>{});
        const auto K = a_grid.get_tensor_descriptor().get_length(number<1>{});
        // const auto topk = value_grid.get_tensor_descriptor().get_length(number<1>{});

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

        constexpr auto block_gemm_softmax_topk_pipeline = Policy::template GetBlockGemmPipeline<Problem>();

        __shared__ char p_smem_char[block_gemm_softmax_topk_pipeline.GetStaticLdsSize()];

        // store value and index
        auto value_window = make_tile_window(
            value_grid, make_tuple(number<kMPerBlock>{}, number<topk>{}), {iM, iN},
            Policy::template MakeOutputDistribution<Problem>());
        auto index_window = make_tile_window(
            index_grid, make_tuple(number<kMPerBlock>{}, number<topk>{}), {iM, iN},
            Policy::template MakeOutputDistribution<Problem>());

        block_gemm_softmax_topk_pipeline(a_block_window, b_block_window, value_window, index_window, K / kKPerBlock, p_smem_char);

        // // cast to WeightType and apply CElementFunction
        // value_block_tile = tile_elementwise_in(
        //     [&](const auto& value) { return c_element_func(type_convert<WeightType>(value)); },
        //     value_block_tile);

        // store_tile(value_window, value_block_tile);
        // store_tile(index_window, index_block_tile);
    }
};

} // namespace ck_tile
