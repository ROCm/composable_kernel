// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_scale_ab_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I8  = int8_t;
using I32 = int32_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = I32;
using BDataType        = I32;
using AccDataType      = I32;
using CShuffleDataType = I32;
using EDataType        = I32;
using ComputeType      = I8;

using ALayout = Row;
using BLayout = Col;
using ELayout = Row;

struct i32_to_i8
{
    __host__ __device__ void operator()(I8& y, const I32& x) const
    {
        float scale = (1.0 / reduced_amax) * int8_max;
        y           = ck::type_convert<I8>(ck::type_convert<float>(x) * scale);
    }

    static constexpr float int8_max = 127;
    // this attribute will trigger a reduction op of the tensor to get the true amax scalue
    float reduced_amax = 1.0;
};

struct i8_to_i32
{
    __host__ __device__ void operator()(I32& y, const I8& x) const
    {
        float a_scale = (1.0 / a_reduced_amax) * int8_max;
        float b_scale = (1.0 / b_reduced_amax) * int8_max;
        float c_scale = (1.0 / (a_scale * b_scale));

        y = ck::type_convert<I32>(ck::type_convert<float>(x) * c_scale);
    }

    static constexpr float int8_max = 127;
    float a_reduced_amax            = 1.0;
    float b_reduced_amax            = 1.0;
};

using AElementOp   = i32_to_i8;
using BElementOp   = i32_to_i8;
using CDEElementOp = i8_to_i32;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleDScaleAB_Xdl_CShuffle<
    ALayout,
    BLayout,
    ck::Tuple<>,
    ELayout,
    ADataType,
    BDataType,
    AccDataType,
    CShuffleDataType,
    ck::Tuple<>,
    EDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmSpec,
    1,
    256,
    256,
    128,
    32,
    8,
    8,
    32,
    32,
    4,
    2,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    4,
    8,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    4,
    8,
    1,
    1,
    1,
    S<1, 32, 1, 8>,
    4,
    ck::LoopScheduler::Default,
    ck::PipelineVersion::v1,
    ComputeType>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = 1024;
    ck::index_t StrideB = 1024;
    ck::index_t StrideE = 1024;

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
        StrideE = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideE\n");
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

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    ADataType amax = 5;
    BDataType bmax = 8;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-amax, amax});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-bmax, bmax});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                           b_device_buf.GetDeviceBuffer(),
                                           {},
                                           e_device_buf.GetDeviceBuffer(),
                                           M,
                                           N,
                                           K,
                                           StrideA,
                                           StrideB,
                                           {},
                                           StrideE,
                                           a_element_op,
                                           b_element_op,
                                           cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_m_n({M, N});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                PassThrough,
                                                                                ComputeType>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  b_k_n,
                                                  c_m_n,
                                                  AElementOp{static_cast<float>(amax)},
                                                  BElementOp{static_cast<float>(bmax)},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);
        auto cde_element_op_ = CDEElementOp{static_cast<float>(amax), static_cast<float>(bmax)};

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op_(e_m_n_host_result(m, n), c_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
    }

    return 0;
}
