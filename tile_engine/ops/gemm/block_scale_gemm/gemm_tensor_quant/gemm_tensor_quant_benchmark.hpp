// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "gemm/gemm_benchmark.hpp"
#include "gemm/gemm_common.hpp"

using GemmTensorQuantProblem = GemmProblem;

inline void gemm_tensor_quant_host_reference(int verify,
                                             ck_tile::HostTensor<ADataType>& a_m_k,
                                             ck_tile::HostTensor<AQDataType>& aq_tensor,
                                             ck_tile::HostTensor<BDataType>& b_k_n,
                                             ck_tile::HostTensor<BQDataType>& bq_tensor,
                                             ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    if(verify == 0)
    {
        return;
    }

    c_m_n_host_result.SetZero();

    ck_tile::reference_gemm_tensor_quant<ADataType,
                                         AQDataType,
                                         BDataType,
                                         BQDataType,
                                         AccDataType,
                                         CDataType>(
        a_m_k, aq_tensor, b_k_n, bq_tensor, c_m_n_host_result);
}
