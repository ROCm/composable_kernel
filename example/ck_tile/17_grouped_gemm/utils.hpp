// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
