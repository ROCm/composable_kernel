// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

struct AddDs
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
    {
        const float x0_f =
            ck_tile::type_convert<float>(c) + (ck_tile::type_convert<float>(ds) + ...);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

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
    using DDataType   = DataType;
};

using ContractionTypes = BatchedContractionTypeConfig<ck_tile::half_t>;

using ADataType   = ContractionTypes::ADataType;
using BDataType   = ContractionTypes::BDataType;
using AccDataType = ContractionTypes::AccDataType;
using EDataType   = ContractionTypes::EDataType;
using DDataType   = ContractionTypes::DDataType;

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m_dims", "4,256", "M dimensions separated by comma (e.g., '16,32' for 2D M)")
        .insert("n_dims", "16,128", "N dimensions separated by comma (e.g., '32,32' for 2D N)")
        .insert("k_dims", "64", "K dimensions separated by comma (e.g., '64,32' for 2D K)")
        .insert(
            "g_dims", "1,2", "G dimensions separated by comma (e.g., '4,2' for 2D, '2,3,4' for 3D)")
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

/**
 * @brief Flattens a list of tensor dimension components into a single dimension vector.
 *
 * This function takes a list of dimension vectors (e.g., representing different components
 * such as G, M, N, or K dimensions) and concatenates them into a single vector.
 *
 * Example:
 * Input: {{G0, G1}, {M0, M1}, {K0}}
 * Output: {G0, G1, M0, M1, K0}
 *
 * @param dim_components A vector of vectors, where each inner vector represents a set of tensor
 * dimensions.
 * @return A single vector containing all dimensions concatenated in order.
 */
std::vector<ck_tile::index_t>
concatenate_dim_components(const std::vector<std::vector<ck_tile::index_t>>& dim_components)
{
    std::vector<ck_tile::index_t> result;

    // Concatenate all dimension components into a single vector
    for(const auto& component : dim_components)
    {
        result.insert(result.end(), component.begin(), component.end());
    }

    return result;
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
