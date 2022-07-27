// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Gemm fused operation. Computes C_m_o = A_m_k * B0_k_n * B1_n_o
                                              |------------|
                                                   Gemm0
                                              |---------------------|
                                                       Gemm1
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using B0DataType       = F16;
using B1DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = F16;

using ALayout  = Row;
using B0Layout = Col;
using B1Layout = Row;
using CLayout  = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmGemm_Xdl_CShuffle<
    ALayout,
    B0Layout,
    B1Layout,
    CLayout,
    ADataType,
    B0DataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,
    256,
    128,         // MPerBlock
    128,         // NPerBlock
    32,          // KPerBlock
    128,          // Gemm1NPerBlock
    32,          // Gemm1KPerBlock
    8,           // AK1
    8,           // BK1
    2,           // B1K1
    32,          // MPerXDL
    32,          // NPerXDL
    1,           // MXdlPerWave
    4,           // NXdlPerWave
    4,           // Gemm1NXdlPerWave
    S<4, 64, 1>, // ABlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    true,
    S<4, 64, 1>, // BBlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    true,
    S<8, 32, 1>, // B1BlockTransfer
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    4,
    2,
    false,
    1,              // CShuffleMXdlPerWavePerShuffle
    2,              // CShuffleNXdlPerWavePerShuffle
    S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                         B0DataType,
                                                                         AccDataType,
                                                                         AccDataType,
                                                                         AElementOp,
                                                                         BElementOp,
                                                                         CElementOp>;
using ReferenceGemm1Instance = ck::tensor_operation::host::
    ReferenceGemm<AccDataType, B1DataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    // ck::index_t M = 1024;
    // ck::index_t N = 1024;
    // ck::index_t K = 64;
    // ck::index_t O = 64;

    // ck::index_t StrideA = 1024;
    // ck::index_t StrideB0 = 1024;
    // ck::index_t StrideB1 = 1024;
    // ck::index_t StrideC = 1024;

    ck::index_t M = 256;
    ck::index_t N = 128;
    ck::index_t K = 32;
    ck::index_t O = 128;
    ck::index_t StrideA = 32;
    ck::index_t StrideB0 = 32;
    ck::index_t StrideB1 = 128;
    ck::index_t StrideC = 128;

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
    else if(argc == 12)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        O = std::stoi(argv[7]);

        StrideA = std::stoi(argv[8]);
        StrideB0 = std::stoi(argv[9]);
        StrideB1 = std::stoi(argv[10]);
        StrideC = std::stoi(argv[11]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    // C_m_o = A_m_k * B0_k_n * B1_n_o
    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB0, B0Layout{}));
    Tensor<B1DataType> b1_n_o(f_host_tensor_descriptor(N, O, StrideB1, B1Layout{}));
    Tensor<CDataType> c_m_o_host_result(f_host_tensor_descriptor(N, O, StrideC, CLayout{}));
    Tensor<CDataType> c_m_o_device_result(f_host_tensor_descriptor(N, O, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "b1_n_o: " << b1_n_o.mDesc << std::endl;
    std::cout << "c_m_o: " << c_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        b1_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-5, 5});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{1});
        b1_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        // b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{1});
        b0_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        b1_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        // b1_n_o.GenerateTensorValue(GeneratorTensor_Sequential<0>{});
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b0_k_n_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpace());
    DeviceMem b1_n_o_device_buf(sizeof(B1DataType) * b1_n_o.mDesc.GetElementSpace());
    DeviceMem c_m_o_device_buf(sizeof(CDataType) * c_m_o_device_result.mDesc.GetElementSpace());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b0_k_n_device_buf.ToDevice(b0_k_n.mData.data());
    b1_n_o_device_buf.ToDevice(b1_n_o.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                      static_cast<B0DataType*>(b0_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<B1DataType*>(b1_n_o_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_m_o_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      O,
                                      StrideA,
                                      StrideB0,
                                      StrideB1,
                                      StrideC,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t)M * N * K * 2 + (size_t)M * N * O * 2;
    std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                            sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    c_m_o_device_buf.FromDevice(c_m_o_device_result.mData.data());

    if(do_verification)
    {
        // Output of Gemm0 is input A of Gemm1
        Tensor<AccDataType> a1_m_n(f_host_tensor_descriptor(M, N, N, Row{}));

        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_m_k, b0_k_n, a1_m_n, a_element_op, b_element_op, c_element_op);

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            a1_m_n, b1_n_o, c_m_o_host_result, a_element_op, b_element_op, c_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // LogRangeAsType<float>(std::cout << "a_m_k: ", a_m_k.mData, ",") << std::endl;
        // LogRangeAsType<float>(std::cout << "b0_k_n : ", b0_k_n.mData, ",") << std::endl;
        // LogRangeAsType<float>(std::cout << "b1_n_o : ", b1_n_o.mData, ",") << std::endl;
        // LogRangeAsType<float>(std::cout << "c_m_o_device_result : ", c_m_o_device_result.mData, ",") << std::endl;

        std::cout << "b0_k_n(0, 0) = " << (float)b0_k_n(0, 0) << ", b0_k_n(1, 0) = " << (float)b0_k_n(1, 0)
                  << ", b0_k_n(0, 1) = " << (float)b0_k_n(0, 1) << ", b0_k_n(1, 1) = " << (float)b0_k_n(1, 1)
                  << std::endl;

        return ck::utils::check_err(c_m_o_device_result.mData, c_m_o_host_result.mData) ? 0 : 1;
    }

    return 0;
}
