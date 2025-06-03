// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <typename TLayout>
constexpr auto
f_host_tensor_descriptor(std::size_t row, std::size_t col, std::size_t stride, TLayout layout)
{
    using namespace ck_tile::literals;

    if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
    {
        return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
    }
    else
    {
        return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
    }
}
template <typename TLayout>
constexpr auto
f_get_default_stride(std::size_t row, std::size_t col, std::size_t stride, TLayout layout)
{
    if(stride == 0)
    {
        if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return col;
        }
        else
        {
            return row;
        }
    }
    else
        return stride;
}

auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(A0DataType) < sizeof(B0DataType), A0DataType, B0DataType>;

    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));

    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);

    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);

    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}
