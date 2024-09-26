// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"

namespace ck_tile {

/*
simple 2d softmax implementation, along row (dim=1)
requirement:
    1). each row is within a warp
    2). data type must be a dword
*/
template <typename Problem_, typename Policy_ = void>
struct BlockSoftmax2D
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using DataType = typename Problem::DataType;

    template <typename DistributedTensor, index_t dim = 1>
    CK_TILE_DEVICE void
    operator()(const DistributedTensor& x, DistributedTensor& y, number<dim> = {})
    {
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // compute row max
        auto row_max =
            block_tile_reduce<DataType>(x, sequence<dim>{}, f_max, -numeric<DataType>::infinity());

        block_tile_reduce_xor_sync(row_max, f_max);

        // compute elementwise softmax
        constexpr auto span_2d = DistributedTensor::get_distributed_spans();

        sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                y(i_j_idx)             = exp(x[i_j_idx] - row_max(i_idx));
            });
        });

        // compute row sum
        auto row_sum = block_tile_reduce<DataType>(y, sequence<dim>{}, f_sum, DataType{0});
        block_tile_reduce_xor_sync(row_sum, f_sum);

        // reciprocal
        auto r = make_static_distributed_tensor<DataType>(row_sum.get_tile_distribution());
        constexpr auto span_1d = decltype(r)::get_distributed_spans();
        sweep_tile_span(span_1d[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            r(i_idx)             = DataType{1} / row_sum(i_idx);
        });

        // scale
        sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                y(i_j_idx)             = y(i_j_idx) * r(i_idx);
            });
        });
    }

    template <typename DistributedTensor, index_t dim = 1>
    CK_TILE_DEVICE decltype(auto) operator()(const DistributedTensor& x, number<dim> = {})
    {
        auto y = DistributedTensor{}; // distributed tensor
        operator()(x, y, number<dim>{});
        return y;
    }
};

} // namespace ck_tile
