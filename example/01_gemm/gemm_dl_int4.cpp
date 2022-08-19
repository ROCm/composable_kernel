// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_dl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using KernelADataType   = int8_t;
using KernelBDataType   = int8_t;
using KernelCDataType   = int8_t;
using KernelAccDataType = int32_t;

using ALayout = Col;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmDl
// ######|           AData|           BData|           CData|           AccData| ALayout| BLayout| CLayout|           A|           B|           C|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|       ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
// ######|            Type|            Type|            Type|              Type|        |        |        | Elementwise| Elementwise| Elementwise| Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
// ######|                |                |                |                  |        |        |        |   Operation|   Operation|   Operation|               |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder|  Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
// ######|                |                |                |                  |        |        |        |            |            |            |               |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                     |                   |                     |               |               |                    |                   |                    |                    |                |                   |
         < KernelADataType, KernelBDataType, KernelCDataType, KernelAccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   128,   128,    16,  4,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<2, 1, 4, 4>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>,      S<2, 1, 4, 4>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,       S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,               5,                  4>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<KernelADataType,
                                                                        KernelBDataType,
                                                                        KernelCDataType,
                                                                        KernelAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    if(argc == 1)
    {
        // do nothing
    }
    else if(argc == 4)
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
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");

        return EXIT_SUCCESS;
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

    using UserADataType = ck::int4_t;
    using UserBDataType = ck::int4_t;
    using UserCDataType = ck::int4_t;

    Tensor<UserADataType> a_m_k_user(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<UserBDataType> b_k_n_user(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k_user.GenerateTensorValue(GeneratorTensor_2<UserADataType>{ck::NumericLimits<UserADataType>::Min(), ck::NumericLimits<UserADataType>::Max()});
        b_k_n_user.GenerateTensorValue(GeneratorTensor_2<UserBDataType>{ck::NumericLimits<UserBDataType>::Min(), ck::NumericLimits<UserBDataType>::Max()});
        break;
    case 2:
        a_m_k_user.GenerateTensorValue(GeneratorTensor_3<UserADataType>{0.0, 1.0});
        b_k_n_user.GenerateTensorValue(GeneratorTensor_3<UserBDataType>{-0.5, 0.5});
        break;
    default:
        a_m_k_user.GenerateTensorValue(GeneratorTensor_Sequential<0>{});
        b_k_n_user.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
    }

    Tensor<KernelADataType> a_m_k(a_m_k_user);
    Tensor<KernelBDataType> b_k_n(b_k_n_user);

    if(!std::equal(std::begin(a_m_k), std::end(a_m_k), std::begin(a_m_k_user)) ||
       !std::equal(std::begin(b_k_n), std::end(b_k_n), std::begin(b_k_n_user)))
    {
        std::cerr << "content are not identical while converting between different-typed "
                     "\'Tensor<>\'s (martix A & B)"
                  << std::endl;
        return EXIT_FAILURE;
    }

    Tensor<KernelCDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<KernelCDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(KernelADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(KernelBDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(KernelCDataType) *
                               c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();
    auto argument =
        gemm.MakeArgument(static_cast<KernelADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<KernelCDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          StrideA,
                          StrideB,
                          StrideC,
                          a_element_op,
                          b_element_op,
                          c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return EXIT_SUCCESS;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(KernelADataType) * M * K + sizeof(KernelBDataType) * K * N +
                            sizeof(KernelCDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

    Tensor<UserCDataType> c_m_n_user(c_m_n_device_result);
    // NOTE: do whatever we want to this converted tensor

    if(do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        return !ck::utils::check_err(c_m_n_device_result.mData, c_m_n_host_result.mData);
    }
}
