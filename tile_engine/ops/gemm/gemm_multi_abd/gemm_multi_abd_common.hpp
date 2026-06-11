// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <array>
#include <vector>
#include <utility>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"

// Helper to create an std::array of HostTensors from a vector of strides.
// All tensors share the same DataType and Layout.
template <typename DataType, typename Layout, std::size_t N, std::size_t... Is>
auto make_host_tensor_array_impl(ck_tile::index_t rows,
                                 ck_tile::index_t cols,
                                 const std::vector<int>& strides,
                                 std::index_sequence<Is...>)
{
    return std::array<ck_tile::HostTensor<DataType>, N>{
        ck_tile::HostTensor<DataType>(ck_tile::host_tensor_descriptor(
            rows, cols, static_cast<ck_tile::index_t>(strides[Is]), is_row_major(Layout{})))...};
}

template <typename DataType, typename Layout, std::size_t N>
auto make_host_tensor_array(ck_tile::index_t rows,
                            ck_tile::index_t cols,
                            const std::vector<int>& strides)
{
    return make_host_tensor_array_impl<DataType, Layout, N>(
        rows, cols, strides, std::make_index_sequence<N>{});
}
