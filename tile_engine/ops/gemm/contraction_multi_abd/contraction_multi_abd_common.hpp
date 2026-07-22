// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <sstream>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "common/utils.hpp"

inline std::vector<ck_tile::index_t> parse_dimensions(const std::string& dims_str)
{
    std::vector<ck_tile::index_t> dims;
    std::stringstream ss(dims_str);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        dims.push_back(std::stoi(token));
    }
    if(dims.empty())
    {
        throw std::invalid_argument("Dimensions cannot be empty");
    }
    return dims;
}

inline ck_tile::index_t calculate_total(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto d : dims)
        total *= d;
    return total;
}

template <std::size_t N>
inline std::array<ck_tile::index_t, N> to_fixed_dims(const std::vector<ck_tile::index_t>& dims)
{
    if(dims.size() != N)
    {
        throw std::invalid_argument("Expected " + std::to_string(N) + " dimensions, got " +
                                    std::to_string(dims.size()));
    }

    std::array<ck_tile::index_t, N> result{};
    std::copy(dims.begin(), dims.end(), result.begin());
    return result;
}

inline std::vector<ck_tile::index_t>
concatenate_dims(const std::vector<std::vector<ck_tile::index_t>>& components)
{
    std::vector<ck_tile::index_t> result;
    for(const auto& c : components)
        result.insert(result.end(), c.begin(), c.end());
    return result;
}

inline std::vector<ck_tile::index_t>
compute_row_major_strides(const std::vector<ck_tile::index_t>& dims)
{
    std::vector<ck_tile::index_t> strides(dims.size(), 1);
    for(int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

// Compute strides for a contraction tensor T[G..., outer..., inner...] honoring the
// 2D matrix layout of the [outer, inner] block:
//   RowMajor    : storage order [G..., outer..., inner...] (inner block innermost)
//   ColumnMajor : storage order [G..., inner..., outer...] (outer block innermost)
// For A: outer = M, inner = K. For B: outer = N, inner = K. For E: outer = M, inner = N.
template <typename Layout>
inline std::vector<ck_tile::index_t>
compute_strides_for_layout(const std::vector<ck_tile::index_t>& g_dims,
                           const std::vector<ck_tile::index_t>& outer_dims,
                           const std::vector<ck_tile::index_t>& inner_dims)
{
    if constexpr(std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>)
    {
        std::vector<ck_tile::index_t> all_dims;
        all_dims.insert(all_dims.end(), g_dims.begin(), g_dims.end());
        all_dims.insert(all_dims.end(), outer_dims.begin(), outer_dims.end());
        all_dims.insert(all_dims.end(), inner_dims.begin(), inner_dims.end());
        return compute_row_major_strides(all_dims);
    }
    else
    {
        std::vector<ck_tile::index_t> storage_dims;
        storage_dims.insert(storage_dims.end(), g_dims.begin(), g_dims.end());
        storage_dims.insert(storage_dims.end(), inner_dims.begin(), inner_dims.end());
        storage_dims.insert(storage_dims.end(), outer_dims.begin(), outer_dims.end());
        const auto storage_strides = compute_row_major_strides(storage_dims);

        const auto num_g     = g_dims.size();
        const auto num_inner = inner_dims.size();
        const auto num_outer = outer_dims.size();

        std::vector<ck_tile::index_t> logical_strides(num_g + num_outer + num_inner);
        for(std::size_t i = 0; i < num_g; ++i)
            logical_strides[i] = storage_strides[i];
        for(std::size_t i = 0; i < num_outer; ++i)
            logical_strides[num_g + i] = storage_strides[num_g + num_inner + i];
        for(std::size_t i = 0; i < num_inner; ++i)
            logical_strides[num_g + num_outer + i] = storage_strides[num_g + i];
        return logical_strides;
    }
}
