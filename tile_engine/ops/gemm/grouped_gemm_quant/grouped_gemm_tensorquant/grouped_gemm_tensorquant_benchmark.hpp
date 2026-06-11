// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm_tensorquant_common.hpp"
#include "../../gemm_benchmark.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts

struct GroupedTensorQuantGemmProblem
{
    int group_count_;
    int kbatch_;
    std::vector<int> Ms_, Ns_, Ks_;
    std::vector<int> stride_As_, stride_Bs_, stride_Cs_;

    std::string dtype_a_, dtype_b_, dtype_acc_, dtype_c_;
    std::string layout_a_, layout_b_, layout_c_;

    friend std::ostream& operator<<(std::ostream& os, const GroupedTensorQuantGemmProblem& problem)
    {
        os << "{\n"
           << "   \"group_count\":" << problem.group_count_ << ",\n"
           << "   \"kbatch\":" << problem.kbatch_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_c\":\"" << problem.dtype_c_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_c\":\"" << problem.layout_c_ << "\",\n"
           << "   \"quant_type\":\"TensorQuant\"\n"
           << "}";
        return os;
    }
};

/// @brief Function to compare grouped TensorQuant gemm results across all groups
bool compare_grouped(std::string instanceName,
                     const GroupedTensorQuantGemmProblem& problem,
                     std::vector<ck_tile::HostTensor<CDataType>>& c_dev_results,
                     std::vector<ck_tile::HostTensor<CDataType>>& c_host_results)
{
    bool pass = true;
    for(int i = 0; i < problem.group_count_; ++i)
    {
        pass &= compare<GroupedTensorQuantGemmProblem>(instanceName + "[" + std::to_string(i) + "]",
                                                       problem.Ks_[i],
                                                       problem.kbatch_,
                                                       c_dev_results[i],
                                                       c_host_results[i]);
    }
    return pass;
}

/// @brief Compute host reference for TensorQuant grouped GEMM using GPU reference then apply scales
/// C_ref[m][n] = aq[0,0] * bq[0,0] * (sum_k A[m][k]*B[k][n])
void gemm_host_reference_grouped(int verify,
                                 const GroupedTensorQuantGemmProblem& problem,
                                 std::vector<ck_tile::HostTensor<ADataType>>& a_tensors,
                                 std::vector<ck_tile::HostTensor<BDataType>>& b_tensors,
                                 std::vector<ck_tile::HostTensor<CDataType>>& c_host_results,
                                 std::vector<ck_tile::HostTensor<AccDataType>>& aq_tensors,
                                 std::vector<ck_tile::HostTensor<AccDataType>>& bq_tensors)
{
    const int group_count = problem.group_count_;

    if(verify > 0)
    {
        for(int i = 0; i < group_count; ++i)
        {
            c_host_results[i].SetZero();
            ck_tile::reference_gemm_tensor_quant<ADataType,
                                                 AccDataType,
                                                 BDataType,
                                                 AccDataType,
                                                 AccDataType,
                                                 CDataType>(
                a_tensors[i], aq_tensors[i], b_tensors[i], bq_tensors[i], c_host_results[i]);
        }
    }
}
#pragma clang diagnostic pop
