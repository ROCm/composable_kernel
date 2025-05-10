// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx_b_preshuffle.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F4        = ck::f4x2_pk_t;
using F16       = ck::half_t;
using BF16      = ck::bhalf_t;
using F32       = float;
using XDataType = ck::e8m0_bexp_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = F4;
using A1DataType       = XDataType;
using B0DataType       = F4;
using B1DataType       = XDataType;
using AccDataType      = F32;
using DsDataType       = ck::Tuple<>;
using CDataType        = BF16;
using CShuffleDataType = CDataType;

using A0Layout = Row;
using B0Layout = Col;
using CLayout  = Row;

void preShuffleBuffer(const F4* src, F4* dst, int N, int K, int NXdl)
{
    int KPack = 32;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex / 2] = src[(n * K + k) / 2];
        }
    }
}

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough; // elementwise transformation for A matrix
using BElementOp = PassThrough; // elementwise transformation for B matrix
using CElementOp = PassThrough; // elementwise transformation for C matrix

constexpr ck::index_t ScaleBlockSize = 32; // scaling block size

constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3_BPreShuffle<
    A0Layout,    B0Layout,    CLayout,          
    A0DataType,  A1DataType,  B0DataType,   B1DataType,   CDataType,    AccDataType,  CShuffleDataType, 
    AElementOp, BElementOp, CElementOp,  GemmSpec,         
    ScaleBlockSize,   256,   
    128,  128,   128,        
    32,    32,               
    16,    16,               
    8,     2,                
    S<4, 64, 1>,   S<1, 0, 2>,   S<1, 0, 2>,   2,   32,   32,   0,            
    S<4, 64, 1>,   S<1, 0, 2>,   S<1, 0, 2>,   2,   32,   32,   0,            
    2,   1,   S<1, 32, 1, 8>,  8,                
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, A0DataType, B0DataType>;
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    bool flush_cache     = true;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideC = N;

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
    else if(argc == 8)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        flush_cache = std::stoi(argv[7]);

        StrideA = K;
        StrideB = K;
        StrideC = N;
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 6: M, N, K\n");
        printf("arg7: flush both I$ and L2$ (0=no, 1=yes)\n");
        exit(0);
    }

    ck::index_t Scale_Stride_AM = (K + ScaleBlockSize - 1) / ScaleBlockSize;
    ck::index_t Scale_Stride_BN = (K + ScaleBlockSize - 1) / ScaleBlockSize;

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

    Tensor<A0DataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, A0Layout{}));
    Tensor<A1DataType> a_m_k_scale(f_host_tensor_descriptor(
        M, (K + ScaleBlockSize - 1) / ScaleBlockSize, Scale_Stride_AM, A0Layout{}));
    Tensor<B0DataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<B0DataType> b_preshuffled(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<B1DataType> b_k_n_scale(f_host_tensor_descriptor(
        (K + ScaleBlockSize - 1) / ScaleBlockSize, N, Scale_Stride_BN, B0Layout{}));
    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "a_m_k_scale: " << a_m_k_scale.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "b_k_n_scale: " << b_k_n_scale.mDesc << std::endl;
    std::cout << "e_m_n: " << c_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 4:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        break;
    case 5:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        break;
    case 6:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{-0.5, 0.5});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
    }

    DeviceMem a_device_buf(sizeof(A0DataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem a_scale_device_buf(sizeof(A1DataType) * a_m_k_scale.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(B0DataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b_scale_device_buf(sizeof(B1DataType) * b_k_n_scale.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    a_scale_device_buf.ToDevice(a_m_k_scale.mData.data());
    b_scale_device_buf.ToDevice(b_k_n_scale.mData.data());

#if 0
    printf("print a_m_k_scale:\n");
    for(int m = 0; m < M; ++m)
    {
        for(int k = 0; k < (K + ScaleBlockSize - 1) / ScaleBlockSize; ++k)
        {
            printf("%f ", ck::type_convert<float>(a_m_k_scale(m, k)));
        }
        printf("\n");
    }
#endif

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    int NPerXdl    = device_op.GetPreShuffleParameters();

    preShuffleBuffer(b_k_n.mData.data(), b_preshuffled.mData.data(), N, K, NPerXdl);
    b_device_buf.ToDevice(b_preshuffled.mData.data());

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(static_cast<A0DataType*>(a_device_buf.GetDeviceBuffer()),
                               static_cast<XDataType*>(a_scale_device_buf.GetDeviceBuffer()),
                               static_cast<B0DataType*>(b_device_buf.GetDeviceBuffer()),
                               static_cast<XDataType*>(b_scale_device_buf.GetDeviceBuffer()),
                               static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                               M,
                               N,
                               K,
                               StrideA,
                               Scale_Stride_AM,
                               StrideB,
                               Scale_Stride_BN,
                               StrideC,
                               1, // KBatch
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    std::size_t flop = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / ScaleBlockSize;
    std::size_t num_btype = sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N +
                            sizeof(CDataType) * M * N +
                            sizeof(XDataType) * (M * K + K * N) / ScaleBlockSize;

    float ave_time = .0;

    if(flush_cache)
    {
        int rotating_buf = (512 * 1024 * 1024 + num_btype - 1) / num_btype;

        ave_time = invoker.Run(argument,
                               StreamConfig{nullptr, time_kernel, 0, 50, 100, true, rotating_buf});
    }
    else
    {
        ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 0, 50, 100});
    }

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << device_op.GetTypeString() << std::endl;

    if(do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMXGemm<A0DataType,
                                                                                  B0DataType,
                                                                                  CDataType,
                                                                                  AccDataType,
                                                                                  XDataType,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  float,
                                                                                  float>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  a_m_k_scale,
                                                  b_k_n,
                                                  b_k_n_scale,
                                                  c_m_n_host_result,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        c_device_buf.FromDevice(c_m_n_device_result.mData.data());

        return ck::utils::check_err(
                   c_m_n_device_result, c_m_n_host_result, "Error: Incorrect results!", 5e-2, 5e-2)
                   ? 0
                   : 1;
    }

    return 0;
}
