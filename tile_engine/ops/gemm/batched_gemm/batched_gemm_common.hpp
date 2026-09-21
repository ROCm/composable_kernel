// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::int8_t>
{
    static constexpr const char* name = "int8";
};

template <>
struct DataTypeTraits<ck_tile::int32_t>
{
    static constexpr const char* name = "int32";
};

template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename Layout>
inline auto make_batched_host_tensor_descriptor(std::size_t batch_count,
                                                std::size_t row,
                                                std::size_t col,
                                                std::size_t stride,
                                                std::size_t batch_stride,
                                                Layout layout)
{
    using namespace ck_tile::literals;

    if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
    {
        return ck_tile::HostTensorDescriptor({batch_count, row, col}, {batch_stride, stride, 1_uz});
    }
    else
    {
        return ck_tile::HostTensorDescriptor({batch_count, row, col}, {batch_stride, 1_uz, stride});
    }
}
