// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType, typename ComputeDataType, typename YDataType, typename ReduceOp>
CK_TILE_HOST void
reference_reduce(const HostTensor<XDataType>& x_m_n, HostTensor<YDataType>& y_m, ReduceOp reduce_op)
{
    auto f = [&](auto m) {
        const int N = x_m_n.mDesc.get_lengths()[1];

        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        for(int n = 0; n < N; ++n)
        {
            const ComputeDataType v_a = type_convert<ComputeDataType>(x_m_n(m, n));

            v_acc = reduce_op(v_acc, v_a);
        }

        y_m(m) = ck_tile::type_convert<YDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f, y_m.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}

// Generic reference reduce for arbitrary dimensions
template <typename XDataType,
          typename ComputeDataType,
          typename YDataType,
          typename ReduceOp,
          typename KeptDim,
          typename ReduceDims>
CK_TILE_HOST void reference_reduce(const HostTensor<XDataType>& x_tensor,
                                   HostTensor<YDataType>& y_tensor,
                                   ReduceOp reduce_op,
                                   KeptDim kept_dim,
                                   ReduceDims reduce_dims)
{
    const auto& x_lengths = x_tensor.mDesc.get_lengths();
    const auto kept_len   = x_lengths[kept_dim.at(0)];

    // Calculate total reduce elements
    index_t total_reduce_elements = 1;
    static_for<0, reduce_dims.size(), 1>{}(
        [&](auto i) { total_reduce_elements *= x_lengths[reduce_dims.at(i)]; });

    auto f = [&](auto kept_idx) {
        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        for(index_t reduce_idx = 0; reduce_idx < total_reduce_elements; ++reduce_idx)
        {
            // Convert linear index to multi-dimensional indices
            std::vector<index_t> indices(x_lengths.size(), 0);
            indices[kept_dim.at(0)] = kept_idx;

            index_t temp = reduce_idx;
            static_for<0, reduce_dims.size(), 1>{}([&](auto i) {
                constexpr auto dim = reduce_dims.at(reduce_dims.size() - 1 - i);
                const auto len     = x_lengths[dim];
                indices[dim]       = temp % len;
                temp /= len;
            });

            // Flat tensor access
            index_t flat_idx    = 0;
            const auto& strides = x_tensor.mDesc.get_strides();
            for(size_t d = 0; d < indices.size(); ++d)
            {
                flat_idx += indices[d] * strides[d];
            }
            const auto v_a = type_convert<ComputeDataType>(x_tensor.mData[flat_idx]);

            v_acc = reduce_op(v_acc, v_a);
        }

        y_tensor(kept_idx) = type_convert<YDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f, kept_len)(std::thread::hardware_concurrency());
}
} // namespace ck_tile
