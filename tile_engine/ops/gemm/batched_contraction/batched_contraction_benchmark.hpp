// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "common/utils.hpp"
#include "gemm/gemm_common.hpp"
#include "gemm/gemm_benchmark.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts

// Helper to parse comma-separated dimension strings
inline std::vector<ck_tile::index_t> parse_dims_string(const std::string& dims_str)
{
    std::vector<ck_tile::index_t> dims;
    if(dims_str.empty())
        return dims;
    std::stringstream ss(dims_str);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        dims.push_back(std::stoi(token));
    }
    return dims;
}

// Helper to calculate total elements from dimension vector
inline ck_tile::index_t calculate_total(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto d : dims)
    {
        total *= d;
    }
    return total;
}

// Helper to concatenate dimension vectors
inline std::vector<ck_tile::index_t>
concatenate_dims(const std::vector<std::vector<ck_tile::index_t>>& dim_components)
{
    std::vector<ck_tile::index_t> result;
    for(const auto& component : dim_components)
    {
        result.insert(result.end(), component.begin(), component.end());
    }
    return result;
}

// Helper to format a dimension vector as a string
inline std::string dims_to_string(const std::vector<ck_tile::index_t>& dims)
{
    std::string result;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i > 0)
            result += ",";
        result += std::to_string(dims[i]);
    }
    return result;
}

struct BatchedContractionProblem
{
    int split_k_;
    std::vector<ck_tile::index_t> g_dims_;
    std::vector<ck_tile::index_t> m_dims_;
    std::vector<ck_tile::index_t> n_dims_;
    std::vector<ck_tile::index_t> k_dims_;
    int num_d_tensors_;

    std::string dtype_a_, dtype_b_, dtype_d_, dtype_acc_, dtype_e_;
    std::string layout_a_, layout_b_, layout_e_;

    // Derived totals
    ck_tile::index_t G_total() const { return calculate_total(g_dims_); }
    ck_tile::index_t M_total() const { return calculate_total(m_dims_); }
    ck_tile::index_t N_total() const { return calculate_total(n_dims_); }
    ck_tile::index_t K_total() const { return calculate_total(k_dims_); }

    friend std::ostream& operator<<(std::ostream& os, const BatchedContractionProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\":" << problem.split_k_ << ",\n"
           << "   \"g_dims\":\"" << dims_to_string(problem.g_dims_) << "\",\n"
           << "   \"m_dims\":\"" << dims_to_string(problem.m_dims_) << "\",\n"
           << "   \"n_dims\":\"" << dims_to_string(problem.n_dims_) << "\",\n"
           << "   \"k_dims\":\"" << dims_to_string(problem.k_dims_) << "\",\n"
           << "   \"G_total\":" << problem.G_total() << ",\n"
           << "   \"M_total\":" << problem.M_total() << ",\n"
           << "   \"N_total\":" << problem.N_total() << ",\n"
           << "   \"K_total\":" << problem.K_total() << ",\n"
           << "   \"num_d_tensors\":" << problem.num_d_tensors_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_d\":\"" << problem.dtype_d_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_e\":\"" << problem.dtype_e_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_e\":\"" << problem.layout_e_ << "\"\n"
           << "}";
        return os;
    }
};

/// @brief Function to get the kernel output with reference implementation on CPU
template <typename ADataType,
          typename BDataType,
          typename DBaseDataType,
          typename AccDataType,
          typename EDataType,
          typename CDEElementWise,
          ck_tile::index_t NumDTensor>
void batched_contraction_host_reference(
    int verify,
    const ck_tile::HostTensor<ADataType>& a_tensor,
    const ck_tile::HostTensor<BDataType>& b_tensor,
    const std::array<ck_tile::HostTensor<DBaseDataType>, NumDTensor>& ds_tensors,
    ck_tile::HostTensor<EDataType>& e_host_result,
    ck_tile::index_t G_total,
    ck_tile::index_t M_total,
    ck_tile::index_t N_total,
    ck_tile::index_t K_total,
    const CDEElementWise& cde_elementwise,
    const std::vector<ck_tile::index_t>& G_dims,
    const std::vector<ck_tile::index_t>& M_dims,
    const std::vector<ck_tile::index_t>& N_dims,
    const std::vector<ck_tile::index_t>& K_dims)
{
    if(verify > 0)
    {
        ck_tile::compute_reference_batched_contraction<ADataType,
                                                       BDataType,
                                                       DBaseDataType,
                                                       EDataType,
                                                       AccDataType,
                                                       CDEElementWise,
                                                       NumDTensor>(a_tensor,
                                                                   b_tensor,
                                                                   ds_tensors,
                                                                   e_host_result,
                                                                   G_total,
                                                                   M_total,
                                                                   N_total,
                                                                   K_total,
                                                                   cde_elementwise,
                                                                   G_dims,
                                                                   M_dims,
                                                                   N_dims,
                                                                   K_dims);
    }
}
#pragma clang diagnostic pop
