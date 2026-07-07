// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/core/numeric/integer.hpp"

// DataTypeTraits for all supported types
template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
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

// Helper function to determine if a layout is row-major
template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// Structure to hold kernel traits for dispatcher
struct AQuantKernelTraits
{
    std::string pipeline;  // mem, compv3
    std::string scheduler; // intrawave, interwave
    std::string epilogue;  // default
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool a_preshuffle_quant;

    AQuantKernelTraits()
        : pipeline("mem"),
          scheduler("interwave"),
          epilogue("default"),
          pad_m(false),
          pad_n(false),
          pad_k(true),
          a_preshuffle_quant(false)
    {
    }
};
