// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

template <typename Problem, typename Policy>
struct GridGemm
{
    using ADataType        = typename Problem::ADataType;
    using BDataType        = typename Problem::BDataType;
    using CDataType        = typename Problem::CDataType;
    using AccDataType      = typename Problem::AccDataType;
    using CElementFunction = typename Problem::CElementFunction;

    static constexpr auto kMPerBlock = Policy::kMPerBlock;
    static constexpr auto kNPerBlock = Policy::kNPerBlock;
    static constexpr auto kKPerBlock = Policy::kKPerBlock;

    template <typename AGridTensorView, typename BGridTensorView, typename CGridTensorView>
    CK_TILE_DEVICE void operator()(const AGridTensorView& a_grid,
                                   const BGridTensorView& b_grid,
                                   CGridTensorView& c_grid,
                                   const CElementFunction& c_element_func) const
    {
        const auto M = a_grid.get_tensor_descriptor().get_length(number<0>{});
        const auto N = c_grid.get_tensor_descriptor().get_length(number<1>{});
        const auto K = a_grid.get_tensor_descriptor().get_length(number<1>{});

        // divide problem
        const auto id_block = get_block_id();

        const auto num_tile_m = integer_divide_ceil(M, kMPerBlock);
        const auto num_tile_n = integer_divide_ceil(N, kNPerBlock);

        const auto block2tile = Policy::template MakeBlock2TileMap<Problem>(num_tile_m, num_tile_n);

        const auto id_tile = block2tile(id_block);

        const auto iM = __builtin_amdgcn_readfirstlane(id_tile.template at<0>() * kMPerBlock);
        const auto iN = __builtin_amdgcn_readfirstlane(id_tile.template at<1>() * kNPerBlock);

        // A block window
        auto a_block_window = make_tile_window(
            a_grid, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {iM, 0});

        // B block window
        auto b_block_window = make_tile_window(
            b_grid, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {iN, 0});

        constexpr auto block_gemm_pipeline = Policy::template GetBlockGemmPipeline<Problem>();

        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];

        const auto acc_block_tile =
            block_gemm_pipeline(a_block_window, b_block_window, K / kKPerBlock, p_smem_char);

        // cast to CDataType and apply CElementFunction
        const auto c_block_tile = tile_elementwise_in(
            [&](const auto& acc) { return c_element_func(type_convert<CDataType>(acc)); },
            acc_block_tile);

        // store C
        auto c_window = make_tile_window(
            c_grid, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, iN});

        store_tile(c_window, c_block_tile);
    }
};

} // namespace ck_tile
