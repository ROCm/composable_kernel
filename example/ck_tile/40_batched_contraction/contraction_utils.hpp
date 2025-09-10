// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV3
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV3
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave

template <typename DataType>
struct BatchedContractionTypeConfig
{
    using ADataType   = DataType;
    using BDataType   = DataType;
    using AccDataType = float;
    using EDataType   = DataType;
};

using ContractionTypes = BatchedContractionTypeConfig<ck_tile::half_t>;

using ADataType   = ContractionTypes::ADataType;
using BDataType   = ContractionTypes::BDataType;
using AccDataType = ContractionTypes::AccDataType;
using EDataType   = ContractionTypes::EDataType;

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "512", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert(
            "g_dims", "4", "G dimensions separated by comma (e.g., '4,2' for 2D, '2,3,4' for 3D)")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_e", "0", "Tensor E stride")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Col by default")
        .insert("e_layout", "R", "E tensor data layout - Row by default")
        .insert("v", "1", "0. No validation, 1. Validation on CPU")
        .insert("prec", "fp16", "data type. fp32/fp16/bf16")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "10", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("log", "1", "log level for debugging");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// Helper function to parse G dimensions from string
std::vector<ck_tile::index_t> parse_g_dimensions(const std::string& g_dims_str)
{
    std::vector<ck_tile::index_t> g_dims;
    std::stringstream ss(g_dims_str);
    std::string token;

    while(std::getline(ss, token, ','))
    {
        g_dims.push_back(std::stoi(token));
    }

    if(g_dims.empty())
    {
        g_dims.push_back(1); // Default to single batch if empty
    }

    return g_dims;
}

// Helper function to calculate total G elements
ck_tile::index_t calculate_total_g(const std::vector<ck_tile::index_t>& g_dims)
{
    ck_tile::index_t total = 1;
    for(auto dim : g_dims)
    {
        total *= dim;
    }
    return total;
}

// Helper function to calculate G strides
std::vector<ck_tile::index_t> calculate_g_strides(const std::vector<ck_tile::index_t>& g_dims,
                                                  ck_tile::index_t base_stride)
{
    std::vector<ck_tile::index_t> strides(g_dims.size());

    if(g_dims.size() == 0)
        return strides;

    // Calculate strides in row-major order
    strides.back() = base_stride; // Last dimension stride

    for(int i = static_cast<int>(g_dims.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * g_dims[i + 1];
    }

    return strides;
}
