// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm/gemm_benchmark.hpp"
#include "gemm_rowcolquant_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

// Data types and Layouts are defined by the generated kernel headers:
// ADataType, BDataType, AQDataType, BQDataType, AccDataType, CDataType
// ALayout, BLayout, CLayout, AQLayout, BQLayout

struct RowColQuantGemmProblem : GemmProblem
{
    int stride_aq_, stride_bq_;

    std::string dtype_aq_, dtype_bq_;

    friend std::ostream& operator<<(std::ostream& os, const RowColQuantGemmProblem& problem)
    {
        os << "{\n"
           << "  \"split_k\":" << problem.split_k_ << ",\n"
           << "  \"m\":" << problem.m_ << ",\n"
           << "  \"n\":" << problem.n_ << ",\n"
           << "  \"k\":" << problem.k_ << ",\n"
           << "  \"stride_a\":" << problem.stride_a_ << ",\n"
           << "  \"stride_b\":" << problem.stride_b_ << ",\n"
           << "  \"stride_c\":" << problem.stride_c_ << ",\n"
           << "  \"stride_aq\":" << problem.stride_aq_ << ",\n"
           << "  \"stride_bq\":" << problem.stride_bq_ << ",\n"
           << "  \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "  \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "  \"dtype_aq\":\"" << problem.dtype_aq_ << "\",\n"
           << "  \"dtype_bq\":\"" << problem.dtype_bq_ << "\",\n"
           << "  \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "  \"dtype_c\":\"" << problem.dtype_c_ << "\",\n"
           << "  \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "  \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "  \"layout_c\":\"" << problem.layout_c_ << "\"\n"
           << "}";
        return os;
    }
};

/// @brief CPU reference implementation for RowColQuant GEMM
void rowcolquant_gemm_host_reference(int verify,
                                     ck_tile::HostTensor<ADataType>& a_m_k,
                                     ck_tile::HostTensor<AQDataType>& aq_m_1,
                                     ck_tile::HostTensor<BDataType>& b_k_n,
                                     ck_tile::HostTensor<BQDataType>& bq_1_n,
                                     ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    if(verify == 1)
    {
        c_m_n_host_result.SetZero();
        ck_tile::reference_gemm_rowcol_quant<ADataType,
                                             AQDataType,
                                             BDataType,
                                             BQDataType,
                                             AccDataType,
                                             CDataType>(
            a_m_k, aq_m_1, b_k_n, bq_1_n, c_m_n_host_result);
    }
}
#pragma clang diagnostic pop
