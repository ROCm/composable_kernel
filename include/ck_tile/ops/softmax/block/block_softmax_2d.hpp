// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"

#define _BLOCK_SOFTMAX_USE_UNPACK2 0

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
#if _BLOCK_SOFTMAX_USE_UNPACK2
        const auto f_max3 = [](auto e0, auto e1, auto e2) {
            float rtn;
            asm volatile("v_max3_f32 %0, %1, %2, %3" : "=v"(rtn) : "v"(e0), "v"(e1), "v"(e2));
            return rtn;
        };
        const auto f_sum3 = [](auto e0, auto e1, auto e2) { return e0 + e1 + e2; };
#endif

        // compute row max
        auto reduce_row_max = BlockReduce2D{x, -numeric<DataType>::infinity()};
#if _BLOCK_SOFTMAX_USE_UNPACK2
        auto row_max = reduce_row_max(f_max3, f_max, sequence<1, 2>{});
#else
        auto row_max = reduce_row_max(f_max);
#endif
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
        auto reduce_row_sum = BlockReduce2D<decltype(y)>{y, DataType{0}};
#if _BLOCK_SOFTMAX_USE_UNPACK2
        auto row_sum = reduce_row_sum(f_sum3, f_sum, sequence<1, 2>{});
#else
        auto row_sum = reduce_row_sum(f_sum);
#endif
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
