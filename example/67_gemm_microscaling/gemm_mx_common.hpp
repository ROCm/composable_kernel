// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
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

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ck::type_convert;

struct ExecutionConfig final
{
    int do_verification = 1;     // (0=no, 1=CPU)
    int init_method     = 2;     // (0=constant values, 1=integer values, 2=decimal values)
    bool time_kernel    = false; // (0=no, 1=yes)
    int verbosity       = 0;     // (0=no info, 1=verbose info)
};

struct ProblemSizeSplitK final
{

    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;

    ck::index_t KBatch = 1;
};

bool parse_cmd_args(int argc,
                    char* argv[],
                    ProblemSizeSplitK& problem_size,
                    ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 5)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        config.verbosity       = std::stoi(argv[4]);
    }
    else if(argc >= 11)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        config.verbosity       = std::stoi(argv[4]);

        problem_size.M = std::stoi(argv[5]);
        problem_size.N = std::stoi(argv[6]);
        problem_size.K = std::stoi(argv[7]);

        problem_size.StrideA = std::stoi(argv[8]);
        problem_size.StrideB = std::stoi(argv[9]);
        problem_size.StrideC = std::stoi(argv[10]);

        if(argc >= 12)
        {
            problem_size.KBatch = std::stoi(argv[11]);
        }
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=CPU)" << std::endl
                  << "arg2: initialization (0=constant values, 1=integer values, 2=decimal values)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4: verbosity (0=no info, 1=verbose info)" << std::endl
                  << "arg5 to 10: M(128x), N(128x), K(256x), StrideA, StrideB, StrideC" << std::endl
                  << "arg11: KBatch" << std::endl;
        return false;
    }

    return true;
}

template <typename DeviceOpInstance,
          typename ADataType,
          typename BDataType,
          typename XDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementOp,
          typename BElementOp,
          typename CElementOp,
          typename AccDataType,
          typename CShuffleDataType,
          ck::index_t ScaleBlockSize>
bool run_mx_gemm(const ProblemSizeSplitK& problem_size, const ExecutionConfig& config)
{

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1, stride});
            }
        };

    auto f_get_default_stride =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<ck::index_t>(col);
                }
                else
                {
                    return static_cast<ck::index_t>(row);
                }
            }
            else
                return static_cast<ck::index_t>(stride);
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    if(K % ScaleBlockSize != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of ScaleBlockSize.");
    };

    // Hardcode scale layouts as per pipeline assumptions
    // TODO: Allow user to specify scale layouts
    using AScaleLayout = Row;
    using BScaleLayout = Col;

    auto Scale_Stride_AM = f_get_default_stride(M, K / ScaleBlockSize, -1, AScaleLayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / ScaleBlockSize, N, -1, BScaleLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<XDataType> a_m_k_scale(f_host_tensor_descriptor(
        M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{})); // scales for A
    Tensor<XDataType> b_k_n_scale(f_host_tensor_descriptor(
        K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{})); // scales for B

    Tensor<CDataType> c_m_n_host_result(
        f_host_tensor_descriptor(M, N, StrideC, CLayout{})); // host verification
    Tensor<CDataType> c_m_n_device_result(
        f_host_tensor_descriptor(M, N, StrideC, CLayout{})); // device result downloaded to host

    if(config.verbosity >= 0)
    {
        std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
        std::cout << "a_m_k_scale: " << a_m_k_scale.mDesc << std::endl;
        std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
        std::cout << "b_k_n_scale: " << b_k_n_scale.mDesc << std::endl;
        std::cout << "c_m_n_device_result: " << c_m_n_device_result.mDesc << std::endl;
    }

    switch(config.init_method)
    {
    case 0: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(2.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(0.5f)}(b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(b_k_n_scale);
        if(config.verbosity > 0)
        {
            std::cout << "Init A = {1}" << std::endl;
            std::cout << "Init A scale = {2.0}" << std::endl;
            std::cout << "Init B = {0.5}" << std::endl;
            std::cout << "Init B scale = {1.0}" << std::endl;
            std::cout << "Expect C = {K}" << std::endl;
        }
        break;

    case 1:

        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 6}); // Z[-5,5]
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 6}); // Z[-5,5]

        if constexpr(ck::is_same_v<XDataType, ck::e8m0_bexp_t>)
        {
            a_m_k_scale.GenerateTensorValue(
                GeneratorTensor_2<XDataType>{125, 129}); // scales: {0.25, 0.5, 1, 2}
            b_k_n_scale.GenerateTensorValue(
                GeneratorTensor_2<XDataType>{125, 129}); // scales: {0.25, 0.5, 1, 2}
        }
        else
        {
            ck::utils::FillUniformDistributionIntegerValue<XDataType>{-1.0f, 1.0f}(a_m_k_scale);
            ck::utils::FillUniformDistributionIntegerValue<XDataType>{-1.0f, 1.0f}(b_k_n_scale);
        }

        break;

    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-2.0, 2.0});
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_3<XDataType>{powf(2.0f, -125.0f), 1.0f});

        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-2.0, 2.0});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_3<XDataType>{powf(2.0f, -125.0f), 1.0f});
        break;

    default:
        if(config.verbosity > 0)
        {
            std::cout << "NOTE: No input data initialization." << std::endl;
        }
    }

    if(config.verbosity > 0)
        std::cout << "Device memory allocation..." << std::endl;
    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem a_scale_device_buf(sizeof(XDataType) * a_m_k_scale.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b_scale_device_buf(sizeof(XDataType) * b_k_n_scale.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    if(config.verbosity > 0)
        std::cout << "Upload data to device..." << std::endl;
    a_device_buf.ToDevice(a_m_k.mData.data());
    a_scale_device_buf.ToDevice(a_m_k_scale.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    b_scale_device_buf.ToDevice(b_k_n_scale.mData.data());
    if(config.verbosity > 0)
        std::cout << "Done." << std::endl;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // run GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                               static_cast<XDataType*>(a_scale_device_buf.GetDeviceBuffer()),
                               static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
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
                               KBatch,
                               a_element_op,
                               b_element_op,
                               c_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong!\n"
                                 "Provided combination of compilation and runtime parameters is "
                                 "not consistent with the supported device_gemm arguments.");
    }

    if(config.verbosity > 0)
    {
        std::cout << "Computing GEMM on device..." << std::endl << std::endl;
    }

    float ave_time =
        invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, config.verbosity, 20, 50});

    bool res_verified = true;
    if(config.do_verification > 0)
    {
        c_device_buf.FromDevice(c_m_n_device_result.mData.data());
        if(config.verbosity > 0)
        {
            std::cout << "Done." << std::endl;
            std::cout << "Computing GEMM on host..." << std::endl;
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMXGemm<ADataType,
                                                                                  BDataType,
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

        if(config.verbosity > 0)
        {
            std::cout << "Done." << std::endl;
            std::cout << "Comparing results..." << std::endl;
        }

        if(config.init_method == 0)
        {
            auto expected = static_cast<float>(K);
            auto computed = type_convert<float>(c_m_n_device_result(1, 12));

            res_verified = res_verified && std::abs(expected - computed) <= 0.0f;
            std::cout << "\nExpected vs Computed: " << expected << " vs " << computed
                      << ((res_verified) ? " (PASSED!)" : " (FAILED!)") << std::endl
                      << std::endl;
        }

        res_verified = res_verified && ck::utils::check_err(c_m_n_device_result,
                                                            c_m_n_host_result,
                                                            "Error: Incorrect results!");

        if(config.verbosity > 0 && res_verified)
            std::cout << "Verification Successful!" << std::endl;
    }
    else
    {
        if(config.verbosity > 0)
            std::cout << "Done." << std::endl;
    }

    if(config.time_kernel)
    {
        // Output size(M*N) * [dot product(2K) + product of scales(K/ScaleBlockSize) + scaling of
        // partial sums(K/ScaleBlockSize)]
        // FLOPS = 2 * M * N * K + 2 * M * N * K / ScaleBlockSize
        std::size_t flop = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / ScaleBlockSize;
        std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(CDataType) * M * N +
                                sizeof(XDataType) * (M * K + K * N) / ScaleBlockSize;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << device_op.GetTypeString() << std::endl;
    }

    return res_verified;
}

template <typename DeviceOpInstance,
          typename ADataType,
          typename BDataType,
          typename XDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementOp,
          typename BElementOp,
          typename CElementOp,
          typename AccDataType,
          typename CShuffleDataType,
          ck::index_t MXVectorSize>
bool run_mx_gemm_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return parse_cmd_args(argc, argv, problem_size, config) &&
           run_mx_gemm<DeviceOpInstance,
                       ADataType,
                       BDataType,
                       XDataType,
                       CDataType,
                       ALayout,
                       BLayout,
                       CLayout,
                       AElementOp,
                       BElementOp,
                       CElementOp,
                       AccDataType,
                       CShuffleDataType,
                       MXVectorSize>(problem_size, config);
}
