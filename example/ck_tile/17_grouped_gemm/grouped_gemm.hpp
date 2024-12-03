// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"

template <typename DataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = float;
};

using Types = GemmBasicTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using CDataType   = Types::CDataType;

struct gemm_grouped_basic_parser_args
{
    std::vector<ck_tile::index_t> Ms;
    std::vector<ck_tile::index_t> Ns;
    std::vector<ck_tile::index_t> Ks;

    std::vector<ck_tile::index_t> stride_As;
    std::vector<ck_tile::index_t> stride_Bs;
    std::vector<ck_tile::index_t> stride_Cs;

    ck_tile::index_t group_count;
    ck_tile::index_t batch_size;
    ck_tile::index_t n_warmup;
    ck_tile::index_t n_repeat;
    bool verbose;
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "R", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("v", "2", "0. No validation, 1. Validation on CPU")
        .insert("warmup", "10", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("group_count", "16", "group count");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

float gemm_calc(std::vector<const void*>& a_m_k_dev_buf,
                std::vector<const void*>& b_k_n_dev_buf,
                std::vector<void*>& c_m_n_dev_buf,
                const gemm_grouped_basic_parser_args& args,
                const ck_tile::stream_config& s);
