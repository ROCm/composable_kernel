// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm/gemm_benchmark.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts
struct GemmMultiDProblem : GemmProblem
{
    int stride_d0_, stride_d1_;
    std::string dtype_d0_, dtype_d1_;
    std::string layout_d0_, layout_d1_;

    friend std::ostream& operator<<(std::ostream& os, const GemmMultiDProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\":" << problem.split_k_ << ",\n"
           << "   \"m\":" << problem.m_ << ",\n"
           << "   \"n\":" << problem.n_ << ",\n"
           << "   \"k\":" << problem.k_ << ",\n"
           << "   \"stride_a\":" << problem.stride_a_ << ",\n"
           << "   \"stride_b\":" << problem.stride_b_ << ",\n"
           << "   \"stride_d0\":" << problem.stride_d0_ << ",\n"
           << "   \"stride_d1\":" << problem.stride_d1_ << ",\n"
           << "   \"stride_c\":" << problem.stride_c_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_d0\":\"" << problem.dtype_d0_ << "\",\n"
           << "   \"dtype_d1\":\"" << problem.dtype_d1_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_c\":\"" << problem.dtype_c_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_d0\":\"" << problem.layout_d0_ << "\",\n"
           << "   \"layout_d1\":\"" << problem.layout_d1_ << "\",\n"
           << "   \"layout_c\":\"" << problem.layout_c_ << "\"" << "\n"
           << "}";
        return os;
    }
};

/// @brief Function to get the kernel output with reference implementation on CPU/GPU
void gemm_multi_d_host_reference(int verify,
                                 ck_tile::HostTensor<ADataType>& a_m_k,
                                 ck_tile::HostTensor<BDataType>& b_k_n,
                                 ck_tile::HostTensor<D0DataType>& d0_m_n,
                                 ck_tile::HostTensor<D1DataType>& d1_m_n,
                                 ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    if(verify > 0)
    {
        // Currently supporting on CPU verification for Gemm Multi D
        // e_m_n_host_result.SetZero();
        ck_tile::reference_gemm_multiple_d<ADataType,
                                           BDataType,
                                           DsDataType,
                                           AccDataType,
                                           CDataType,
                                           ElementWiseFn>(
            a_m_k, b_k_n, {d0_m_n, d1_m_n}, c_m_n_host_result);
    }
}
#pragma clang diagnostic pop
