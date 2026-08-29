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
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "gemm/gemm_benchmark.hpp"

#include "gemm_multi_abd_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts
struct GemmMultiABDProblem : GemmProblem
{
    // GemmProblem provides: split_k_, m_, n_, k_
    // stride_a_, stride_b_, stride_c_ are unused (multi-tensor ops use vectors below)
    std::vector<int> stride_as_;
    std::vector<int> stride_bs_;
    std::vector<int> stride_ds_;
    int stride_e_;
    std::vector<std::string> dtype_as_;
    std::vector<std::string> dtype_bs_;
    std::vector<std::string> dtype_ds_;
    std::string dtype_e_;
    std::vector<std::string> layout_as_;
    std::vector<std::string> layout_bs_;
    std::vector<std::string> layout_ds_;
    std::string layout_e_;
    std::string a_elementwise_;
    std::string b_elementwise_;
    std::string cde_elementwise_;

    friend std::ostream& operator<<(std::ostream& os, const GemmMultiABDProblem& p)
    {
        os << "{\n"
           << "   \"split_k\":" << p.split_k_ << ",\n"
           << "   \"m\":" << p.m_ << ",\n"
           << "   \"n\":" << p.n_ << ",\n"
           << "   \"k\":" << p.k_ << ",\n";
        for(std::size_t i = 0; i < p.stride_as_.size(); i++)
            os << "   \"stride_a" << i << "\":" << p.stride_as_[i] << ",\n";
        for(std::size_t i = 0; i < p.stride_bs_.size(); i++)
            os << "   \"stride_b" << i << "\":" << p.stride_bs_[i] << ",\n";
        for(std::size_t i = 0; i < p.stride_ds_.size(); i++)
            os << "   \"stride_d" << i << "\":" << p.stride_ds_[i] << ",\n";
        os << "   \"stride_e\":" << p.stride_e_ << ",\n";
        for(std::size_t i = 0; i < p.dtype_as_.size(); i++)
            os << "   \"dtype_a" << i << "\":\"" << p.dtype_as_[i] << "\",\n";
        for(std::size_t i = 0; i < p.dtype_bs_.size(); i++)
            os << "   \"dtype_b" << i << "\":\"" << p.dtype_bs_[i] << "\",\n";
        for(std::size_t i = 0; i < p.dtype_ds_.size(); i++)
            os << "   \"dtype_d" << i << "\":\"" << p.dtype_ds_[i] << "\",\n";
        os << "   \"dtype_acc\":\"" << p.dtype_acc_ << "\",\n"
           << "   \"dtype_e\":\"" << p.dtype_e_ << "\",\n";
        for(std::size_t i = 0; i < p.layout_as_.size(); i++)
            os << "   \"layout_a" << i << "\":\"" << p.layout_as_[i] << "\",\n";
        for(std::size_t i = 0; i < p.layout_bs_.size(); i++)
            os << "   \"layout_b" << i << "\":\"" << p.layout_bs_[i] << "\",\n";
        for(std::size_t i = 0; i < p.layout_ds_.size(); i++)
            os << "   \"layout_d" << i << "\":\"" << p.layout_ds_[i] << "\",\n";
        os << "   \"layout_e\":\"" << p.layout_e_ << "\",\n"
           << "   \"a_elementwise\":\"" << p.a_elementwise_ << "\",\n"
           << "   \"b_elementwise\":\"" << p.b_elementwise_ << "\",\n"
           << "   \"cde_elementwise\":\"" << p.cde_elementwise_ << "\"\n"
           << "}";
        return os;
    }
};

/// @brief Host reference computation for multi ABD
template <std::size_t NumA, std::size_t NumB, std::size_t NumD>
void gemm_multi_abd_host_reference(
    int verify,
    const std::array<ck_tile::HostTensor<ADataType>, NumA>& as_tensors,
    const std::array<ck_tile::HostTensor<BDataType>, NumB>& bs_tensors,
    const std::array<ck_tile::HostTensor<DBaseDataType>, NumD>& ds_tensors,
    ck_tile::HostTensor<EDataType>& e_m_n_host_result)
{
    if(verify > 0)
    {
        ck_tile::index_t M = as_tensors[0].get_length(0);
        ck_tile::index_t K = as_tensors[0].get_length(1);
        ck_tile::index_t N = bs_tensors[0].get_length(1);

        ck_tile::HostTensor<ADataType> a_m_k({M, K});
        ck_tile::HostTensor<BDataType> b_k_n({K, N});

        ck_tile::reference_gemm_multiple_abd<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             AElementWiseFn,
                                             BElementWiseFn,
                                             CDEElementWiseFn,
                                             ADataType,
                                             BDataType,
                                             DBaseDataType>(
            as_tensors, bs_tensors, ds_tensors, a_m_k, b_k_n, e_m_n_host_result);
    }
}

#pragma clang diagnostic pop
