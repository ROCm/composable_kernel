// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GridGemmSoftmaxTopk
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
    
    // for topk computing
    struct ArgmaxPacket
    {
        WeightType arg;
        IndexType value;
    };

    template <typename AGridTensorView, typename BGridTensorView, typename ValueGridTensorView, typename IndexGridTensorView>
    CK_TILE_DEVICE void operator()(const AGridTensorView& a_grid,
                                   const BGridTensorView& b_grid,
                                   ValueGridTensorView& value_grid,
                                   IndexGridTensorView& index_grid,
                                   const CElementFunction& c_element_func) const
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

        constexpr auto block_gemm_pipeline = Policy::template GetBlockGemmPipeline<Problem>();

        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];

        // store value and index
        auto value_window = make_tile_window(
            value_grid, make_tuple(number<kMPerBlock>{}, number<topk>{}), {iM, iN},
            ck_tile::BlockGemmPipelineAGmemBGmemCRegDefaultPolicy::template MakeOutputDistribution<Problem>());
        auto index_window = make_tile_window(
            index_grid, make_tuple(number<kMPerBlock>{}, number<topk>{}), {iM, iN},
            ck_tile::BlockGemmPipelineAGmemBGmemCRegDefaultPolicy::template MakeOutputDistribution<Problem>());

        const auto acc_block_tile =
            block_gemm_pipeline(a_block_window, b_block_window, K / kKPerBlock, p_smem_char);

        // cast to CDataType and apply CElementFunction
        const auto c_block_tile = tile_elementwise_in(
            [&](const auto& acc) { return c_element_func(type_convert<CDataType>(acc)); },
            acc_block_tile);

        using ValueBlockTileDistr = decltype(value_window.get_tile_distribution());
        using IndexBlockTileDistr = decltype(index_window.get_tile_distribution());

        using ValueBlockTile = decltype(make_static_distributed_tensor<WeightType>(ValueBlockTileDistr{}));
        using IndexBlockTile = decltype(make_static_distributed_tensor<IndexType>(IndexBlockTileDistr{}));

        ValueBlockTile value_block_tile;
        IndexBlockTile index_block_tile;

        // apply softmax for c_block_tile
        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // m_local = rowmax(c_block_tile)
        auto m_local = block_tile_reduce<ComputeDataType>(
            c_block_tile, sequence<1>{}, f_max, std::numeric_limits<ComputeDataType>::lowest());
        
        block_tile_reduce_sync(m_local, f_max);

        // Pcompute{j} = sum(exp(x - m_local))
        auto p_compute =
            make_static_distributed_tensor<ComputeDataType>(c_block_tile.get_tile_distribution());

        constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();

        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                p_compute(i_j_idx) = exp(c_block_tile[i_j_idx] - m_local[i_idx]);
            });
        });

        // rowsum for p_compute{i, j}
        auto rowsum_p = block_tile_reduce<ComputeDataType>(
            p_compute, sequence<1>{}, f_sum, ComputeDataType{0});

        block_tile_reduce_sync(rowsum_p, f_sum);

        // softmax = p_compute{i, j} / rowsum_p
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                p_compute(i_j_idx) = p_compute[i_j_idx] / rowsum_p[i_idx];
            });
        });

        // apply topk for softmax output
        auto x_tmp = p_compute;
        // constexpr auto dst_dist = BlockGemmPipelineAGmemBGmemCRegDefaultPolicy::MakeOutputDistribution();

        // argmax for topk
        const auto f_argmax = [](ArgmaxPacket e0, ArgmaxPacket e1) {
            return e0.arg > e1.arg ? e0 : e1;
        };

        for(index_t i_k = 0; i_k < topk; i_k++)
        {
            constexpr auto span_2d = decltype(p_compute)::get_distributed_spans();
            auto packet            = [&]() {
                auto tmp = make_static_distributed_tensor<ArgmaxPacket>(p_compute.get_tile_distribution());

                sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        ArgmaxPacket t;
                        t.arg        = x_tmp(i_j_idx); // !!! we reference p_compute here
                        t.value      = tile_idx.at(number<1>{});
                        tmp(i_j_idx) = t;
                    });
                });
                return tmp;
            }();

            auto argmax_init = ArgmaxPacket{-numeric<WeightType>::infinity(), 0};
            auto r = block_tile_reduce<ArgmaxPacket>(packet, sequence<1>{}, f_argmax, argmax_init);
            block_tile_reduce_xor_sync(r, f_argmax);

            // auto value_block_tile = make_static_distributed_tensor<WeightType>(dst_dist);
            // auto index_block_tile = make_static_distributed_tensor<IndexType>(dst_dist);

            // Initialize value_block_tile and index_block_tile
            tile_elementwise_inout([](auto& value) { value = 0; }, value_block_tile);
            tile_elementwise_inout([](auto& index) { index = 0; }, index_block_tile);

            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ArgmaxPacket tmp       = r(i_j_idx);
                    value_block_tile(i_j_idx)             = tmp.arg;
                    index_block_tile(i_j_idx)             = tmp.value;
                });
            });

            // update value
            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});

                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    x_tmp(i_j_idx) = (col_id == r(i_j_idx).value) ? -numeric<WeightType>::infinity()
                                                                    : x_tmp(i_j_idx);
                });
            });
        }

        // store value and index
        store_tile(value_window, value_block_tile);
        store_tile(index_window, index_block_tile);
    }
};

} // namespace ck_tile
