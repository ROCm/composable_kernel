// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "gemm_bias_add.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using F16 = ck::half_t;
using FP8 = ck::f8_t;
using F32 = float;

using A0DataType       = F16;
using B0DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F16;
using DsDataType       = ck::Tuple<D0DataType>;
using EDataType        = F16;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0Layout = Row;
using B0Layout = Row;
using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
// using Add         = ck::tensor_operation::element_wise::Add;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<A0DataType,
                                                                        B0DataType,
                                                                        EDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

    // GEMM shape
    ck::index_t M = 16;
    ck::index_t N = 16;
    ck::index_t K = 64;

    ck::index_t StrideA = K;
    ck::index_t StrideB = N;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 11)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<A0DataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, A0Layout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<D0DataType> d0_m_n(f_host_tensor_descriptor(M, N, StrideD, D0Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{-0.5, 0.5});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{0});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{0});
    }

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    d0_device_buf.ToDevice(d0_m_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    GemmBiasAddArgs gemm_args{a0_device_buf.GetDeviceBuffer(),
                              b0_device_buf.GetDeviceBuffer(),
                              d0_device_buf.GetDeviceBuffer(),
                              e_device_buf.GetDeviceBuffer(),
                              M,
                              N,
                              K};

    float ave_time = gemm_bias_add_fp16(gemm_args, StreamConfig{nullptr, time_kernel, 20, 50});
    // float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 20, 50});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {

        // RunUnfusedTest(a0_m_k.mData, b0_k_n.mData, d0_m_n.mData, e_m_n_host_result.mData, K, M,
        // N);
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a0_m_k, b0_k_n, e_m_n_host_result, AElementOp{}, BElementOp{}, CElementOp{});

        ref_invoker.Run(ref_argument);

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
    }

    return 0;
}
