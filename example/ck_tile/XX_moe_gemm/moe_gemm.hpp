// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/moe_gemm.hpp"

template <typename DataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = float;
};

using Types = GemmTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using CDataType   = Types::CDataType;

using moe_gemm_kargs = ck_tile::MoeGemmHostArgs;

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("experts", "8", "Num of experts - 8 by default")
        .insert("NumTokens", "128", "M dimensions - 128 by default.")
        .insert("TopK", "3", "Top K - 2 by default.")
        // .insert("TopK", "2", "Top K - 2 by default.")
        // .insert("N", "8192", "N dimensions - 4096 by default.")
        // .insert("K", "6144", "K dimensions - 4096 by default.")
        .insert("N", "4096", "N dimensions - 4096 by default.")
        .insert("K", "4096", "K dimensions - 4096 by default.")
        .insert("stride_A", "", "Tensor A strides - it is empty by default.")
        .insert("stride_B", "", "Tensor B strides - it is empty by default.")
        .insert("stride_C", "", "Tensor C strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "C", "B tensor data layout - Col by default.")
        .insert("c_layout", "R", "C tensor data layout - Row by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("repeat", "10", "number of iterations to benchmark the kernel.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

std::size_t get_workspace_size(const moe_gemm_kargs& gemm_desc);

float moe_gemm(const moe_gemm_kargs& gemm_desc, const ck_tile::stream_config& s);
