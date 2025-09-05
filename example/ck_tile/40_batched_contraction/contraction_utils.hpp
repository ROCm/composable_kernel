// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

// Type configuration for batched contraction - similar to GemmTypeConfig
template <typename ADataType, typename BDataType = ADataType, typename EDataType = ADataType>
struct BatchedContractionTypeConfig;

template <>
struct BatchedContractionTypeConfig<float>
{
    using ADataType   = float;
    using BDataType   = float;
    using AccDataType = float;
    using EDataType   = float;
};

template <>
struct BatchedContractionTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using EDataType   = ck_tile::half_t;
};

template <>
struct BatchedContractionTypeConfig<ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using EDataType   = ck_tile::bf16_t;
};

// Data type traits for printing
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

// Layout helper (for future use)
template <typename Layout>
constexpr bool is_row_major()
{
    return std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>;
}

// Argument creation function - similar to gemm_utils.hpp
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256", "M dimension")
        .insert("n", "256", "N dimension")
        .insert("k", "128", "K dimension")
        .insert("batch", "4", "Batch count")
        .insert("prec", "fp32", "data type. fp32/fp16/bf16")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "50", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("init", "1", "0:uniform[-5,5], 1:monotonic, 2:uniform[1,1], other=zero")
        .insert("flush_cache", "0", "flush cache before running the kernel")
        .insert("log", "0", "log level for debugging");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
