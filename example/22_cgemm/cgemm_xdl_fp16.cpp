// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_cgemm_4gemm_xdl_cshuffle.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_cgemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = F16;
using BDataType   = F16;
using CDataType   = F16;
using AccDataType = F32;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceCGemmInstance = ck::tensor_operation::device::DeviceCGemm_4Gemm_Xdl_CShuffle
    <ALayout,                    // typename ALayout
     BLayout,                    // typename BLayout
     CLayout,                    // typename CLayout
     ADataType,                  // typename ADataType
     BDataType,                  // typename BDataType
     CDataType,                  // typename CDataType
     AccDataType,                // typename GemmAccDataType
     CDataType,                  // typename CShuffleDataType
     PassThrough,                // typename AElementwiseOperation
     PassThrough,                // typename BElementwiseOperation
     PassThrough,                // typename CElementwiseOperation
     GemmDefault,                // GemmSpecialization GemmSpec
     1,                          // index_t NumGemmKPrefetchStage
     256,                        // index_t BlockSize
     256,                        // index_t MPerBlock
     128,                        // index_t NPerBlock
     32,                         // index_t KPerBlock
     8,                          // index_t AK1
     8,                          // index_t BK1
     32,                         // index_t MPerXDL
     32,                         // index_t NPerXDL
     4,                          // index_t MXdlPerWave
     2,                          // index_t NXdlPerWave
     S<4, 64, 1>,                // typename ABlockTransferThreadClusterLengths_AK0_M_AK1
     S<1, 0, 2>,                 // typename ABlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename ABlockTransferSrcAccessOrder
     2,                          // index_t ABlockTransferSrcVectorDim
     8,                          // index_t ABlockTransferSrcScalarPerVector
     8,                          // index_t ABlockTransferDstScalarPerVector_AK1
     1,                          // index_t ABlockLdsExtraM
     S<4, 64, 1>,                // typename BBlockTransferThreadClusterLengths_BK0_N_BK1
     S<1, 0, 2>,                 // typename BBlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename BBlockTransferSrcAccessOrder
     2,                          // index_t BBlockTransferSrcVectorDim
     8,                          // index_t BBlockTransferSrcScalarPerVector
     8,                          // index_t BBlockTransferDstScalarPerVector_BK1
     1,                          // index_t BBlockLdsExtraN
     1,                          // index_t CShuffleMXdlPerWavePerShuffle
     1,                          // index_t CShuffleNXdlPerWavePerShuffle
     S<1, 32, 1, 8>,             // typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
     8>;                         // index_t CShuffleBlockTransferScalarPerVector_NPerBlock
// clang-format on

using ReferenceCGemmInstance = ck::tensor_operation::host::
    ReferenceCGemm<ADataType, BDataType, CDataType, PassThrough, PassThrough, PassThrough>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // CGEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
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

    Tensor<ADataType> a_m_k_real(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<ADataType> a_m_k_imag(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n_real(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> b_k_n_imag(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<CDataType> c_m_n_real_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_imag_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k_real: " << a_m_k_real.mDesc << std::endl;
    std::cout << "a_m_k_imag: " << a_m_k_imag.mDesc << std::endl;
    std::cout << "b_k_n_real: " << b_k_n_real.mDesc << std::endl;
    std::cout << "b_k_n_imag: " << b_k_n_imag.mDesc << std::endl;
    std::cout << "c_m_n_real: " << c_m_n_real_device_result.mDesc << std::endl;
    std::cout << "c_m_n_imag: " << c_m_n_imag_device_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k_real.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        a_m_k_imag.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n_real.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b_k_n_imag.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    default:
        a_m_k_real.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        a_m_k_imag.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        b_k_n_real.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        b_k_n_imag.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    auto cgemm = DeviceCGemmInstance{};

    DeviceMem a_m_k_real_device_buf(sizeof(ADataType) * a_m_k_real.mDesc.GetElementSpaceSize());
    DeviceMem a_m_k_imag_device_buf(sizeof(ADataType) * a_m_k_imag.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_real_device_buf(sizeof(BDataType) * b_k_n_real.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_imag_device_buf(sizeof(BDataType) * b_k_n_imag.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_real_device_buf(sizeof(CDataType) *
                                    c_m_n_real_device_result.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_imag_device_buf(sizeof(CDataType) *
                                    c_m_n_imag_device_result.mDesc.GetElementSpaceSize());
    DeviceMem workspace_device_buf(cgemm.GetWorkspaceSize(M, N, K, StrideA, StrideB, StrideC));

    a_m_k_real_device_buf.ToDevice(a_m_k_real.mData.data());
    a_m_k_imag_device_buf.ToDevice(a_m_k_imag.mData.data());
    b_k_n_real_device_buf.ToDevice(b_k_n_real.mData.data());
    b_k_n_imag_device_buf.ToDevice(b_k_n_imag.mData.data());

    auto a_element_op = PassThrough{};
    auto b_element_op = PassThrough{};
    auto c_element_op = PassThrough{};

    // do GEMM
    auto invoker = cgemm.MakeInvoker();
    auto argument =
        cgemm.MakeArgument(static_cast<ADataType*>(a_m_k_real_device_buf.GetDeviceBuffer()),
                           static_cast<ADataType*>(a_m_k_imag_device_buf.GetDeviceBuffer()),
                           static_cast<BDataType*>(b_k_n_real_device_buf.GetDeviceBuffer()),
                           static_cast<BDataType*>(b_k_n_imag_device_buf.GetDeviceBuffer()),
                           static_cast<CDataType*>(c_m_n_real_device_buf.GetDeviceBuffer()),
                           static_cast<CDataType*>(c_m_n_imag_device_buf.GetDeviceBuffer()),
                           static_cast<CDataType*>(workspace_device_buf.GetDeviceBuffer()),
                           M,
                           N,
                           K,
                           StrideA,
                           StrideB,
                           StrideC,
                           a_element_op,
                           b_element_op,
                           c_element_op);

    if(!cgemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_cgemm with the specified compilation parameters does "
            "not support this CGEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(8) * M * N * K;
    std::size_t num_btype =
        std::size_t(2) *
        (sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << cgemm.GetTypeString() << std::endl;

    c_m_n_real_device_buf.FromDevice(c_m_n_real_device_result.mData.data());
    c_m_n_imag_device_buf.FromDevice(c_m_n_imag_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CDataType> c_m_n_real_host_result(
            f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_imag_host_result(
            f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

        auto ref_cgemm   = ReferenceCGemmInstance{};
        auto ref_invoker = ref_cgemm.MakeInvoker();

        auto ref_argument = ref_cgemm.MakeArgument(a_m_k_real,
                                                   a_m_k_imag,
                                                   b_k_n_real,
                                                   b_k_n_imag,
                                                   c_m_n_real_host_result,
                                                   c_m_n_imag_host_result,
                                                   a_element_op,
                                                   b_element_op,
                                                   c_element_op);

        ref_invoker.Run(ref_argument);

        ck::utils::check_err(c_m_n_real_device_result.mData,
                             c_m_n_real_host_result.mData,
                             "Verification error: incorrect results in real part!",
                             1e-2f,
                             1e-1f);
        ck::utils::check_err(c_m_n_imag_device_result.mData,
                             c_m_n_imag_host_result.mData,
                             "Verification error: incorrect results in imaginary part!",
                             1e-2f,
                             1e-1f);
    }

    return 0;
}
