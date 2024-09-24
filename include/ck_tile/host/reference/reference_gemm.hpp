// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm(const HostTensor<ADataType>& a_m_k,
                                 const HostTensor<BDataType>& b_n_k,
                                 HostTensor<CDataType>& c_m_n,
                                 const AElementOp& a_element_op     = {},
                                 const BElementOp& b_element_op     = {},
                                 const ACCElementOp& acc_element_op = {})
{
    const int N = b_n_k.mDesc.get_lengths()[0];
    const int K = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                      ? a_m_k.mDesc.get_lengths()[1]
                      : a_m_k.mDesc.get_lengths()[0];
    const int M = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                      ? a_m_k.mDesc.get_lengths()[0]
                      : a_m_k.mDesc.get_lengths()[1];

    auto f = [&](auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                                    ? a_element_op(a_m_k(m, k))
                                    : a_element_op(a_m_k(k, m));
                BDataType v_b = b_element_op(b_n_k(n, k));

                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
        }
    };

    make_ParallelTensorFunctor(f, M)(std::thread::hardware_concurrency());
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
__global__ void naive_gemm_kernel(ADataType* A,
                                  BDataType* B,
                                  CDataType* C,
                                  ck_tile::index_t M,
                                  ck_tile::index_t N,
                                  ck_tile::index_t K,
                                  ck_tile::index_t strideA,
                                  ck_tile::index_t strideB,
                                  ck_tile::index_t strideC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N; // Compute row index
    int col = idx % N; // Compute column index

    if(row < M && col < N)
    {
        AccDataType acc = 0.0;

        for(int k = 0; k < K; ++k)
        {
            acc += static_cast<AccDataType>(A[row * strideA + k]) *
                   static_cast<AccDataType>(B[col * strideB + k]);
        }

        C[row * strideC + col] = acc; // Store as AccDataType
    }
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_gemm_gpu(DeviceMem& a_device,
                        DeviceMem& b_device,
                        DeviceMem& c_device,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t stride_a,
                        index_t stride_b,
                        index_t stride_c)
{

    ADataType* d_A;
    BDataType* d_B;
    CDataType* d_C;

    hipError_t errA = hipMalloc(&d_A, M * K * sizeof(ADataType));
    hipError_t errB = hipMalloc(&d_B, N * K * sizeof(BDataType));
    hipError_t errC = hipMalloc(&d_C, M * N * sizeof(CDataType));
    if(errA != hipSuccess)
    {
        std::cerr << "Error allocating device memory for A: " << hipGetErrorString(errA)
                  << std::endl;
        return; // Early exit on error
    }

    if(errB != hipSuccess)
    {
        std::cerr << "Error allocating device memory for B: " << hipGetErrorString(errB)
                  << std::endl;
        return; // Early exit on error
    }

    if(errC != hipSuccess)
    {
        std::cerr << "Error allocating device memory for C: " << hipGetErrorString(errC)
                  << std::endl;
        return; // Early exit on error
    }

    errA = hipMemcpy(
        d_A, a_device.GetDeviceBuffer(), M * K * sizeof(ADataType), hipMemcpyHostToDevice);
    if(errA != hipSuccess)
    {
        std::cerr << "Error copying A to device: " << hipGetErrorString(errA) << std::endl;
    }

    errB = hipMemcpy(
        d_B, b_device.GetDeviceBuffer(), N * K * sizeof(BDataType), hipMemcpyHostToDevice);
    if(errB != hipSuccess)
    {
        std::cerr << "Error copying B to device: " << hipGetErrorString(errB) << std::endl;
    }

    int totalElements      = M * N;
    int numThreadsPerBlock = 256; // Common choice for threads per block
    int numBlocks          = (totalElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    naive_gemm_kernel<ADataType, BDataType, AccDataType, CDataType>
        <<<numBlocks, numThreadsPerBlock>>>(d_A, d_B, d_C, M, N, K, stride_a, stride_b, stride_c);
    errC = hipMemcpy(
        c_device.GetDeviceBuffer(), d_C, M * N * sizeof(CDataType), hipMemcpyDeviceToHost);
    if(errC != hipSuccess)
    {
        std::cerr << "Error copying C to device: " << hipGetErrorString(errC) << std::endl;
    }

    errA = hipFree(d_A);
    if(errA != hipSuccess)
    {
        std::cerr << "Error free the A memory: " << hipGetErrorString(errA) << std::endl;
    }

    errB = hipFree(d_B);
    if(errB != hipSuccess)
    {
        std::cerr << "Error free the B memory: " << hipGetErrorString(errB) << std::endl;
    }

    errC = hipFree(d_C);
    if(errC != hipSuccess)
    {
        std::cerr << "Error free the C memory: " << hipGetErrorString(errC) << std::endl;
    }

    return;
}
} // namespace ck_tile
