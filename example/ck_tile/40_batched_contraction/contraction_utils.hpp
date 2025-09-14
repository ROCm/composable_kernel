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
    arg_parser.insert("m_dims", "512", "M dimensions separated by comma (e.g., '16,32' for 2D M)")
        .insert("n_dims", "1024", "N dimensions separated by comma (e.g., '32,32' for 2D N)")
        .insert("k_dims", "2048", "K dimensions separated by comma (e.g., '64,32' for 2D K)")
        .insert(
            "g_dims", "8", "G dimensions separated by comma (e.g., '4,2' for 2D, '2,3,4' for 3D)")
        .insert("stride_a", "0", "Custom A tensor leading dimension stride (0 = auto)")
        .insert("stride_b", "0", "Custom B tensor leading dimension stride (0 = auto)")
        .insert("stride_e", "0", "Custom E tensor leading dimension stride (0 = auto)")
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

// Helper function to parse G, M, N, K dimensions from string
std::vector<ck_tile::index_t> parse_dimensions(const std::string& dims_str)
{
    std::vector<ck_tile::index_t> dims;
    std::stringstream ss(dims_str);
    std::string token;

    while(std::getline(ss, token, ','))
    {
        dims.push_back(std::stoi(token));
    }

    if(dims.empty())
    {
        throw std::invalid_argument("Dimensions cannot be empty");
    }

    return dims;
}

// Helper function to Calculate total elements from multi-dimensional vector
ck_tile::index_t calculate_total_elements(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto dim : dims)
    {
        total *= dim;
    }
    return total;
}

// Helper function to Build tensor dimensions vector from dimensions for example
// [G0,G1,..,M0,M1,..,K0,K1,..] >> [s0,s1,..]
std::vector<ck_tile::index_t>
get_tensor_dims(const std::vector<std::vector<ck_tile::index_t>>& dim_components)
{
    std::vector<ck_tile::index_t> result;

    // Concatenate all dimension components: [G_dims] + ([M_dims] or [N_dims]) + ([K_dims] or
    // [N_dims])
    for(const auto& component : dim_components)
    {
        result.insert(result.end(), component.begin(), component.end());
    }

    return result;
}

// Helper function to Calculate tensor strides from all dimensions
std::vector<ck_tile::index_t> get_tensor_strides(const std::vector<ck_tile::index_t>& dims)
{
    std::vector<ck_tile::index_t> strides(dims.size());

    if(dims.empty())
        return strides;

    // Row-major strides: rightmost dimension has stride 1
    strides.back() = 1;

    // Calculate strides from right to left
    for(int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    return strides;
}

// Helper function for printing dimensions
void print_dims(const std::string& name,
                const std::vector<ck_tile::index_t>& dims,
                ck_tile::index_t total)
{
    std::cout << name << ": [";
    for(size_t i = 0; i < dims.size(); ++i)
    {
        std::cout << dims[i];
        if(i < dims.size() - 1)
            std::cout << ",";
    }
    std::cout << "] ";
    if(total != 0)
        std::cout << "(total=" << total << ")";
    std::cout << std::endl;
}
