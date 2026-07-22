// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

// 2 A tensors (fp16, fp16), 1 B tensor (fp16), 1 D tensor (fp16), E = fp16
using A0DataType = ck_tile::half_t;
using A1DataType = ck_tile::half_t;
using B0DataType = ck_tile::half_t;
using D0DataType = ck_tile::half_t;
using EDataType  = ck_tile::half_t;

using AsDataType = ck_tile::tuple<A0DataType, A1DataType>;
using BsDataType = ck_tile::tuple<B0DataType>;
using DsDataType = ck_tile::tuple<D0DataType>;

using AccDataType = float;

static constexpr ck_tile::index_t NumATensor = AsDataType::size();
static constexpr ck_tile::index_t NumBTensor = BsDataType::size();
static constexpr ck_tile::index_t NumDTensor = DsDataType::size();

#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV3
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV3
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave

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

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m_dims", "4,256", "M dimensions separated by comma")
        .insert("n_dims", "16,128", "N dimensions separated by comma")
        .insert("k_dims", "64", "K dimensions separated by comma")
        .insert("g_dims", "1,2", "G dimensions separated by comma")
        .insert("v", "1", "0. No validation, 1. Validation on CPU")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "10", "number of iterations to benchmark the kernel")
        .insert("log", "1", "log level for debugging");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

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

ck_tile::index_t calculate_total_elements(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto dim : dims)
        total *= dim;
    return total;
}

template <std::size_t N>
std::array<ck_tile::index_t, N> to_fixed_dims(const std::vector<ck_tile::index_t>& dims)
{
    if(dims.size() != N)
    {
        throw std::invalid_argument("Expected " + std::to_string(N) + " dimensions, got " +
                                    std::to_string(dims.size()));
    }

    std::array<ck_tile::index_t, N> result{};
    std::copy(dims.begin(), dims.end(), result.begin());
    return result;
}

std::vector<ck_tile::index_t>
concatenate_dim_components(const std::vector<std::vector<ck_tile::index_t>>& dim_components)
{
    std::vector<ck_tile::index_t> result;
    for(const auto& component : dim_components)
        result.insert(result.end(), component.begin(), component.end());
    return result;
}

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
    std::cout << "]";
    if(total != 0)
        std::cout << " (total=" << total << ")";
    std::cout << std::endl;
}

inline std::vector<ck_tile::index_t>
compute_row_major_strides(const std::vector<ck_tile::index_t>& dims)
{
    std::vector<ck_tile::index_t> strides(dims.size(), 1);
    for(int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

// Compute strides for a contraction tensor T[G..., outer..., inner...] honoring the
// 2D matrix layout of the [outer, inner] block:
//   RowMajor    : storage order [G..., outer..., inner...] (inner block innermost)
//   ColumnMajor : storage order [G..., inner..., outer...] (outer block innermost)
// For A: outer = M, inner = K. For B: outer = N, inner = K. For E: outer = M, inner = N.
template <typename Layout>
inline std::vector<ck_tile::index_t>
compute_strides_for_layout(const std::vector<ck_tile::index_t>& g_dims,
                           const std::vector<ck_tile::index_t>& outer_dims,
                           const std::vector<ck_tile::index_t>& inner_dims)
{
    if constexpr(std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>)
    {
        std::vector<ck_tile::index_t> all_dims;
        all_dims.insert(all_dims.end(), g_dims.begin(), g_dims.end());
        all_dims.insert(all_dims.end(), outer_dims.begin(), outer_dims.end());
        all_dims.insert(all_dims.end(), inner_dims.begin(), inner_dims.end());
        return compute_row_major_strides(all_dims);
    }
    else
    {
        std::vector<ck_tile::index_t> storage_dims;
        storage_dims.insert(storage_dims.end(), g_dims.begin(), g_dims.end());
        storage_dims.insert(storage_dims.end(), inner_dims.begin(), inner_dims.end());
        storage_dims.insert(storage_dims.end(), outer_dims.begin(), outer_dims.end());
        const auto storage_strides = compute_row_major_strides(storage_dims);

        const auto num_g     = g_dims.size();
        const auto num_inner = inner_dims.size();
        const auto num_outer = outer_dims.size();

        std::vector<ck_tile::index_t> logical_strides(num_g + num_outer + num_inner);
        for(std::size_t i = 0; i < num_g; ++i)
            logical_strides[i] = storage_strides[i];
        for(std::size_t i = 0; i < num_outer; ++i)
            logical_strides[num_g + i] = storage_strides[num_g + num_inner + i];
        for(std::size_t i = 0; i < num_inner; ++i)
            logical_strides[num_g + num_outer + i] = storage_strides[num_g + i];
        return logical_strides;
    }
}
