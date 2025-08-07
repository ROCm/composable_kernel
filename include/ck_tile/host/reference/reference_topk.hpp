// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>
#include <numeric>
#include <functional>
#include <utility>
#include <algorithm>

namespace ck_tile {

/*
    similiar to torch.topk()
    x (Tensor) – the input tensor.
    k (int) – the k in “top-k”
    dim (int, optional) – the dimension to sort along
    largest (bool, optional) – largest or smallest elements
    sorted (bool, optional) – elements in sorted order or not

    output:
    y_values
    y_indices

    https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TopKImpl.h
*/
template <typename DataType, typename IndexType = index_t>
CK_TILE_HOST void reference_topk(const HostTensor<DataType>& x,
                                 HostTensor<DataType>& y_values,
                                 HostTensor<IndexType>& y_indices,
                                 index_t k,
                                 index_t dim  = -1,
                                 bool largest = true,
                                 bool sorted  = true)
{
    // rank must be the same
    index_t rank = x.get_num_of_dimension();
    assert(static_cast<std::size_t>(rank) == y_values.get_num_of_dimension());
    assert(static_cast<size_t>(rank) == y_indices.get_num_of_dimension());
    assert(dim == -1 || dim < rank);

    index_t topk_dim     = dim == -1 ? (rank - 1) : dim;
    index_t topk_src_len = x.get_length(topk_dim);
    auto x_len           = x.get_lengths();

    assert(k <= topk_src_len);
    assert(static_cast<size_t>(k) == y_values.get_length(topk_dim) &&
           static_cast<size_t>(k) == y_indices.get_length(topk_dim));

    index_t n_parallel = x.get_element_size() / topk_src_len;

    // clang-format off
    auto f = [&](auto i_element) {
        std::vector<size_t> topk_coord = [&](){
            std::vector<size_t> t_(rank, 0);
            size_t r = i_element;
            for(index_t i = rank - 1; i >= 0; i--) {
                if(i == topk_dim)          continue; // topk dim should be zero
                t_[i] = r % x_len[i];      r = r / x_len[i];
            }
            return t_;
        }();

        using elem_t = std::pair<DataType, IndexType>;
        std::vector<elem_t> q = [&](){
            std::vector<elem_t> t_(topk_src_len);
            for(index_t i = 0; i < topk_src_len; i++) {
                auto c_ = topk_coord;  c_[topk_dim] = i;
                t_[i].first = x(c_);   t_[i].second = i;
            }
            return t_;
        }();

        // run topk
        if(largest) {
            std::nth_element(q.begin(), q.begin() + k - 1, q.end(),
            [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first > rhs.first; });
            if(sorted) {
                std::sort(q.begin(), q.begin() + k - 1,
                [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first > rhs.first; });
            }
        } else {
            std::nth_element(q.begin(), q.begin() + k - 1, q.end(),
            [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first < rhs.first; });
            if(sorted) {
                std::sort(q.begin(), q.begin() + k - 1,
                [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first < rhs.first; });
            }
        }

        // write out
        for(index_t i = 0; i < k; i++) {
            auto c_ = topk_coord;  c_[topk_dim] = i;
            y_values(c_) = q[i].first;  y_indices(c_) = q[i].second;
        }
    };
    // clang-format on

    make_ParallelTensorFunctor(f, n_parallel)(std::thread::hardware_concurrency());
}

// TODO: if using this method, the return tensor would be dense(no stride)
template <typename DataType, typename IndexType = index_t>
CK_TILE_HOST auto reference_topk(const HostTensor<DataType>& x,
                                 index_t k,
                                 index_t dim  = -1,
                                 bool largest = true,
                                 bool sorted  = true)
{
    auto lens          = x.get_lengths();
    index_t target_dim = (dim == -1) ? (lens.size() - 1) : dim;
    assert(target_dim < lens.size());
    assert(k <= lens[target_dim]);
    lens[target_dim] = k;
    HostTensor<DataType> y_values(lens);
    HostTensor<IndexType> y_indices(lens);

    reference_topk<DataType, IndexType>(x, y_values, y_indices, k, dim, largest, sorted);

    return ck_tile::make_tuple(y_values, y_indices);
}

/*
    similiar to vllm grouped_topk() in fused_moe.py
    x (Tensor) – the input tensor.
    topk (int) – the k in “top-k”
    num_expert_group (int) – the number of expert groups
    topk_group (int) – the k for expert groups

    dim (int, optional) – the dimension to sort along
    largest (bool, optional) – largest or smallest elements
    sorted (bool, optional) – elements in sorted order or not

    output:
    y_values
    y_indices

    https://github.com/ROCm/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L1657
*/
template <typename DataType, typename IndexType = index_t>
CK_TILE_HOST void reference_grouped_topk(const HostTensor<DataType>& x,
                                         HostTensor<DataType>& y_values,
                                         HostTensor<IndexType>& y_indices,
                                         index_t topk,
                                         index_t num_expert_group = 4,
                                         index_t topk_group = 2,
                                         index_t dim  = -1,
                                         bool largest = true,
                                         bool sorted  = true)
{
    auto lens          = x.get_lengths();
    index_t num_token = lens[0];
    index_t num_expert = lens[1];
    index_t expert_per_group = num_expert / num_expert_group;

    index_t target_dim = (dim == -1) ? (lens.size() - 1) : dim;
    assert(target_dim < lens.size());
    assert(k <= lens[target_dim]);
    lens[target_dim] = topk;

    HostTensor<DataType> group_scores({num_token, num_expert_group}, {num_expert_group, 1});
    HostTensor<DataType> group_mask({num_token, num_expert_group}, {num_expert_group, 1});

    HostTensor<DataType> score_mask({num_token, num_expert}, {num_expert, 1});
    HostTensor<DataType> masked_scores({num_token, num_expert}, {num_expert, 1});

    HostTensor<DataType> group_values({num_token, topk_group}, {topk_group, 1});
    HostTensor<IndexType> group_indices({num_token, topk_group}, {topk_group, 1});

    // calculate group score
    auto f1 = [&](auto m) {
        for(int n_group = 0; n_group < num_expert_group; ++n_group) {
            // max value for expert group
            DataType group_max = std::numeric_limits<DataType>::lowest();
            for(int n = n_group * expert_per_group; n < (n_group + 1) * expert_per_group; ++n)
            {
                const DataType group_value = x(m, n);
                group_max = group_max < group_value ? group_value : group_max;
            }
            group_scores(m, n_group) = group_max;
        }
    };
    make_ParallelTensorFunctor(f1, num_token)(std::thread::hardware_concurrency());

    // select group values and group_indices
    reference_topk<DataType, IndexType>(group_scores, group_values, group_indices, topk_group, dim, largest, sorted);

    // mask score
    auto f2 = [&](auto m) {
        // initialize score mask as -inf
        for(int n = 0; n < num_expert; ++n) {
            score_mask(m, n) = std::numeric_limits<DataType>::lowest();
        }
        // set mask value = 0 for topk groups
        for(int k_group = 0; k_group < topk_group; ++k_group) {
            int k_group_idx = group_indices(m, k_group);
            for(int n = k_group_idx * expert_per_group; n < (k_group_idx + 1) * expert_per_group; ++n)
            {
                score_mask(m, n) = 0;
            }
        }
        // add mask for scores
        for(int n = 0; n < num_expert; ++n) {
            masked_scores(m, n) = x(m, n) + score_mask(m, n);
        }
    };
    make_ParallelTensorFunctor(f2, num_token)(std::thread::hardware_concurrency());

    // select topk values from masked scores
    reference_topk<DataType, IndexType>(masked_scores, y_values, y_indices, topk, dim, largest, sorted);

}
} // namespace ck_tile
