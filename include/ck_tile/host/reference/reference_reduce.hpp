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
template <
    typename XDataType,
    typename ComputeDataType,
    typename YDataType,
    typename ReduceOp,
    typename KeptDim, // Expected type: ck_tile::sequence<...> containing dimension indices to keep
    typename ReduceDims> // Expected type: ck_tile::sequence<...> containing dimension indices to
                         // reduce
CK_TILE_HOST void reference_reduce(const HostTensor<XDataType>& x_tensor,
                                   HostTensor<YDataType>& y_tensor,
                                   ReduceOp reduce_op,
                                   KeptDim kept_dim,
                                   ReduceDims reduce_dims)
{
    const auto& x_lengths = x_tensor.mDesc.get_lengths();

    // Calculate total kept elements (product of all kept dimension lengths)
    index_t total_kept_elements = 1;
    static_for<0, kept_dim.size(), 1>{}(
        [&](auto i) { total_kept_elements *= x_lengths[kept_dim.at(i)]; });

    // Calculate total reduce elements (product of all reduce dimension lengths)
    index_t total_reduce_elements = 1;
    static_for<0, reduce_dims.size(), 1>{}(
        [&](auto i) { total_reduce_elements *= x_lengths[reduce_dims.at(i)]; });

    auto f = [&](auto linear_kept_idx) {
        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        // Convert linear kept index to multi-dimensional kept indices
        std::vector<index_t> kept_indices(kept_dim.size());
        index_t temp_kept = linear_kept_idx;
        static_for<0, kept_dim.size(), 1>{}([&](auto i) {
            constexpr auto dim_idx = kept_dim.size() - 1 - i;
            constexpr auto dim     = kept_dim.at(dim_idx);
            const auto len         = x_lengths[dim];
            kept_indices[dim_idx]  = temp_kept % len;
            temp_kept /= len;
        });

        for(index_t reduce_idx = 0; reduce_idx < total_reduce_elements; ++reduce_idx)
        {
            // Convert linear reduce index to multi-dimensional reduce indices
            std::vector<index_t> reduce_indices(reduce_dims.size());
            index_t temp_reduce = reduce_idx;
            static_for<0, reduce_dims.size(), 1>{}([&](auto i) {
                constexpr auto dim_idx  = reduce_dims.size() - 1 - i;
                constexpr auto dim      = reduce_dims.at(dim_idx);
                const auto len          = x_lengths[dim];
                reduce_indices[dim_idx] = temp_reduce % len;
                temp_reduce /= len;
            });

            // Build full input tensor indices by combining kept and reduce indices
            std::vector<std::size_t> full_indices(x_lengths.size(), 0);
            static_for<0, kept_dim.size(), 1>{}(
                [&](auto i) { full_indices[kept_dim.at(i)] = kept_indices[i]; });
            static_for<0, reduce_dims.size(), 1>{}(
                [&](auto i) { full_indices[reduce_dims.at(i)] = reduce_indices[i]; });

            // Access input tensor element
            const auto v_a = type_convert<ComputeDataType>(x_tensor(full_indices));

            v_acc = reduce_op(v_acc, v_a);
        }

        // Calculate output tensor index using kept indices
        // The output tensor has the same structure as the kept dimensions
        std::vector<std::size_t> y_indices(kept_dim.size());
        static_for<0, kept_dim.size(), 1>{}([&](auto i) { y_indices[i] = kept_indices[i]; });

        y_tensor(y_indices) = type_convert<YDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f, total_kept_elements)(std::thread::hardware_concurrency());
}
} // namespace ck_tile
