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

/// @brief Function to get the kernel output with reference implementation on CPU/GPU
void gemm_host_reference(int verify,
                         ck_tile::HostTensor<ADataType>& a_m_k,
                         ck_tile::HostTensor<BDataType>& b_k_n,
                         ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                         ck_tile::DeviceMem& a_m_k_dev_buf,
                         ck_tile::DeviceMem& b_k_n_dev_buf,
                         ck_tile::index_t M,
                         ck_tile::index_t N,
                         ck_tile::index_t K,
                         ck_tile::index_t stride_A,
                         ck_tile::index_t stride_B,
                         ck_tile::index_t stride_C)
{
    if(verify == 1)
    {
        c_m_n_host_result.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_result);
    }
    else if(verify == 2)
    {
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Restore input for B for gpu reference
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_host_result.get_element_space_size_in_bytes());
        c_m_n_host_result.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buf_ref.GetDeviceBuffer());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_host_result.data());
    }
}
#pragma clang diagnostic pop
