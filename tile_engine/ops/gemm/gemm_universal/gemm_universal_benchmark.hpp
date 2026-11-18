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

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
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

/// @brief Checks that the float -> CDataType conversion used by the host reference
/// (-verify 1) and by the GPU reference (-verify 2) rounds to nearest even when C is bf16.
/// CK's compile-time default bf16 rounding is truncation; RNE comes from the LLVM builtin
/// path, which depends on the build flags (CK_GFX12_SUPPORT on the host side). An exact
/// compare against a reference that rounds differently would report false mismatches.
/// The check runs a 1 x N x 256 GEMM of all-ones A and integer B whose column sums are
/// rounding ties or near-ties in bf16, so it adds no device code beyond -verify 2.
/// Returns true when both conversions are RNE, or when C is not bf16 (nothing to check).
inline bool check_bf16_c_rounding_is_rne()
{
    if constexpr(!std::is_same_v<CDataType, ck_tile::bf16_t>)
    {
        std::cout << "bf16 RNE check: skipped, C is not bf16" << std::endl;
        return true;
    }
    else
    {
        // Column sums. 259 -> 260, 263 -> 264, -259 -> -260, 515 -> 516, 1029 -> 1032 under
        // RNE (truncation gives 258, 262, -258, 512, 1024); 257 -> 256 (tie to even).
        const std::vector<int> sums = {259, 263, -259, 515, 1029, 257};
        const ck_tile::index_t M    = 1;
        const ck_tile::index_t N    = static_cast<ck_tile::index_t>(sums.size());
        const ck_tile::index_t K    = 256;
        const ck_tile::index_t stride_a =
            ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
        const ck_tile::index_t stride_b =
            ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
        const ck_tile::index_t stride_c =
            ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

        ck_tile::HostTensor<ADataType> a(
            ck_tile::host_tensor_descriptor(M, K, stride_a, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b(
            ck_tile::host_tensor_descriptor(K, N, stride_b, is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_host(
            ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(CLayout{})));
        ck_tile::HostTensor<CDataType> c_gpu(
            ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(CLayout{})));

        a.SetZero();
        b.SetZero();
        for(ck_tile::index_t k = 0; k < K; ++k)
        {
            a(0, k) = ck_tile::type_convert<ADataType>(1.f);
        }
        // Column n holds the sum as entries of +-8 plus one remainder entry; integers up to
        // 8 are exact in every input type of this benchmark (fp16, bf16, fp8, bf8).
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            const int sign = sums[n] < 0 ? -1 : 1;
            int left       = sums[n] * sign;
            for(ck_tile::index_t k = 0; k < K && left > 0; ++k)
            {
                const int v = left < 8 ? left : 8;
                b(k, n)     = ck_tile::type_convert<BDataType>(static_cast<float>(sign * v));
                left -= v;
            }
        }

        c_host.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(a, b, c_host);

        ck_tile::DeviceMem a_buf(a.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_buf(b.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_buf(c_gpu.get_element_space_size_in_bytes());
        a_buf.ToDevice(a.data());
        b_buf.ToDevice(b.data());
        c_buf.SetZero();
        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                                             static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                                             static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
                                             M,
                                             N,
                                             K,
                                             stride_a,
                                             stride_b,
                                             stride_c);
        c_buf.FromDevice(c_gpu.data());

        bool pass = true;
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            // Software RNE of the exact fp32 sum.
            const uint32_t f        = ck_tile::bit_cast<uint32_t>(static_cast<float>(sums[n]));
            const uint16_t expected = static_cast<uint16_t>((f + 0x7fffu + ((f >> 16) & 1u)) >> 16);
            const uint16_t host     = ck_tile::bit_cast<uint16_t>(c_host(0, n));
            const uint16_t gpu      = ck_tile::bit_cast<uint16_t>(c_gpu(0, n));
            if(host != expected || gpu != expected)
            {
                std::cout << "bf16 RNE check: sum " << sums[n] << " expected 0x" << std::hex
                          << expected << " host 0x" << host << " device 0x" << gpu << std::dec
                          << std::endl;
                pass = false;
            }
        }
        std::cout << "bf16 RNE check (host and device reference conversion): "
                  << (pass ? "pass" : "FAIL") << std::endl;
        return pass;
    }
}
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
