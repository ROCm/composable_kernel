// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

template <typename DataType>
struct BatchedGemmTypeConfig;

template <>
struct BatchedGemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

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

using Types = BatchedGemmTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using CDataType   = Types::CDataType;

struct batched_gemm_args
{
    const void* p_a;
    const void* p_b;
    void* p_c;
    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    ck_tile::index_t stride_C;
    ck_tile::index_t batch_stride_A;
    ck_tile::index_t batch_stride_B;
    ck_tile::index_t batch_stride_C;
    ck_tile::index_t batch_count;
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256", "m dimension")
        .insert("n", "128", "n dimension")
        .insert("k", "128", "k dimension")
        .insert("stride_a", "128", "Tensor A stride")
        .insert("stride_b", "128", "Tensor B stride")
        .insert("stride_c", "128", "Tensor C stride")
        .insert("batch_stride_a", "32768", "Batch A stride")
        .insert("batch_stride_b", "16384", "Batch B stride")
        .insert("batch_stride_c", "32768", "Batch C stride")
        .insert("batch_count", "16", "Batch count")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// host API
float gemm_calc(batched_gemm_args args, const ck_tile::stream_config& s);
