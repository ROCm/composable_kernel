// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"

using ScaleDataType = ck::e8m0_bexp_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct ExecutionConfig final
{
    int do_verification = 1;     // (0=no, 1=CPU)
    int init_method     = 2;     // (0=no init, 1=integer value, 2=decimal value)
    bool time_kernel    = false; // (0=no, 1=yes)
    int verbosity       = 0;     // (0=no info, 1=verbose info)
};

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;
};

bool parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig& config)
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
    else if(argc == 11)
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
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=CPU)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4: verbosity (0=no info, 1=verbose info)" << std::endl
                  << "arg5 to 10: M (16x), N(16x), K(16x), StrideA, StrideB, StrideC" << std::endl;
        return false;
    }

    return true;
}

template <typename ADataType,
          typename BDataType,
          typename XDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename CElementWiseOp,
          typename AccDataType,
          typename CShuffleDataType,
          ck::index_t MXVectorSize>
bool run_mx_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using ELayout      = CLayout;
    using DsLayout     = ck::Tuple<>;
    using DsDataType   = ck::Tuple<>;
    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = CElementWiseOp;

    static constexpr auto GemmSpec      = ck::tensor_operation::device::GemmSpecialization::Default;
    static constexpr auto BlkGemmPSched = ck::BlockGemmPipelineScheduler::Intrawave;
    static constexpr auto BlkGemmPVer   = ck::BlockGemmPipelineVersion::v3;

#if 1
    // XXX: These parameters should not exist in MX-native GEMM kernel
    static constexpr ck::index_t Scale_Block_M = 128;
    static constexpr ck::index_t Scale_Block_N = 128;
#endif
    static constexpr ck::index_t Scale_Block_K = MXVectorSize;

    // XXX: DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3 is not designed to utilize MX-specific MFMA
    //      instructions.
    //
    // XXX: DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3 is not designed to utilize device-optimized
    //      scaled type convert functions.
    //
    // XXX: In DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3, KPerBlock is expected to be equal to
    //      ScaleBlockK (aka MXVectorSize).
    //      Additionally, the following is also expected:
    //         static_assert(ScaleBlockM % MPerBlock == 0);
    //         static_assert(ScaleBlockN % NPerBlock == 0);
    //         In MX-native GEMM kernel these requirements should be relaxed.
    //
    // XXX: It appears, by default we are using mfma_f32_16x16x4xf32
    //      MfmaSelector<ComputeTypeA, MPerXdl, NPerXdl, ComputeTypeB>::selected_mfma.k_per_blk =
    //          MfmaSelector<float, 16, 16, float>::selected_mfma.k_per_blk = mfma_f32_16x16x4xf32
    // XXX: GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3 assumes scale type is float

    // clang-format off
    using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3
    // ######| ALayout| BLayout| DsLayout| CLayout| ADataType|    AScale| BDataType|    BScale| DsDataType| CDataType|     GemmAcc| CShuffleDataType|AElementwise|BElementwise| CElementwise| GemmSpec|Block|   ScaleBlockM|   ScaleBlockN|   ScaleBlockK|    M|    N|             K| AK1| BK1|   M|   N|MXdl|NXdl|ABlockTransfer|ABlockTransfer|ABlockTransfer|ABlockTransfer|ABlockTransfer|ABlockTransfer|   ABlock|BBlockTransfer|BBlockTransfer|BBlockTransfer|BBlockTransfer|BBlockTransfer|BBlockTransfer|   BBlock|  CShuffle|  CShuffle|CShuffleBlockTransfer|CDEShuffleBlockTransfer|       BlkGemm|     BlkGemm|ComputeTypeA|ComputeTypeB|LDSTypeA|LDSTypeB|
    // ######|        |        |         |        |          |  DataType|          |  DataType|           |          |    DataType|                 |   Operation|   Operation|    Operation|         | Size|              |              |              |  Per|  Per|           Per|    |    | Per| Per| Per| Per| ThreadCluster| ThreadCluster|SrcAccessOrder|  SrcVectorDim|     SrcScalar|     DstScalar|LdsExtraM| ThreadCluster| ThreadCluster|SrcAccessOrder|     SrcVector|     SrcScalar|     DstScalar|LdsExtraN|      MXdl|      NXdl|       ClusterLengths|                 Scalar|     PipeSched| PipelineVer|            |            |        |        |
    // ######|        |        |         |        |          |          |          |          |           |          |            |                 |            |            |             |         |     |              |              |              |Block|Block|         Block|    |    | XDL| XDL|Wave|Wave|       Lengths|  ArrangeOrder|              |              |     PerVector| PerVector_AK1|         |       Lengths|  ArrangeOrder|              |           Dim|     PerVector| PerVector_BK1|         |   PerWave|   PerWave|     MBlock_MPerBlock|             PerVectors|              |            |            |            |        |        |
    // ######|        |        |         |        |          |          |          |          |           |          |            |                 |            |            |             |         |     |              |              |              |     |     |              |    |    |    |    |    |    |     AK0_M_AK1|              |              |              |              |              |         |     BK0_N_BK1|              |              |                             |              |         |PerShuffle|PerShuffle|     NBlock_NPerBlock|                       |              |            |            |            |        |        |
             < ALayout, BLayout, DsLayout, ELayout, ADataType, XDataType, BDataType, XDataType, DsDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp, GemmSpec,  256, Scale_Block_M, Scale_Block_N, Scale_Block_K,  128,  128,           128,  16,  16,  16,  16,   4,   4,   S<8, 32, 1>,    S<1, 0, 2>,    S<1, 0, 2>,             2,            16,            16,        0,   S<8, 32, 1>,    S<1, 0, 2>,    S<1, 0, 2>,             2,            16,            16,        0,         1,         2,       S<1, 32, 1, 8>,             S<8, 8, 1>, BlkGemmPSched, BlkGemmPVer,       float,       float,    float,  float>;
    // clang-format on

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;

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

    if(K % Scale_Block_K != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of Scale_Block_K (16 or 32)");
    };

    auto Scale_Stride_AM = f_get_default_stride(M, K / Scale_Block_K, StrideA, ALayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / Scale_Block_K, N, StrideB, BLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<XDataType> a_m_k_scale(
        f_host_tensor_descriptor(M, K / Scale_Block_K, Scale_Stride_AM, ALayout{})); // scales for A
    Tensor<XDataType> b_k_n_scale(
        f_host_tensor_descriptor(K / Scale_Block_K, N, Scale_Stride_BN, BLayout{})); // scales for B

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
    case 0:
        if(config.verbosity > 0)
        {
            std::cout << "NOTE: No input data initialization." << std::endl;
        }
        break;
    case 1:
    case 2:
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(0.5f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(1.0f)}(b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(2.0f)}(b_k_n_scale);
        if(config.verbosity > 0)
        {
            std::cout << "Init A = {1}" << std::endl;
            std::cout << "Init A scale = {0.5}" << std::endl;
            std::cout << "Init B = {1}" << std::endl;
            std::cout << "Init B scale = {2.0}" << std::endl;
            std::cout << "Expect C = {K}" << std::endl;
        }
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

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                           b_device_buf.GetDeviceBuffer(),
                                           std::array<const void*, NumDTensor>{},
                                           c_device_buf.GetDeviceBuffer(),
                                           M,
                                           N,
                                           K,
                                           StrideA,
                                           StrideB,
                                           std::array<ck::index_t, NumDTensor>{},
                                           StrideC,
                                           a_scale_device_buf.GetDeviceBuffer(),
                                           b_scale_device_buf.GetDeviceBuffer(),
                                           a_element_op,
                                           b_element_op,
                                           cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong!\n"
                                 "Provided combination of compilation and runtime parameters is "
                                 "not consistent with the supported device_gemm arguments.");
    }

    if(config.verbosity > 0)
        std::cout << "Computing GEMM on device..." << std::endl;
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
                                                                                  float,
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

        if(config.init_method == 1)
        {
            res_verified =
                res_verified && std::abs(static_cast<float>(K) - c_m_n_device_result(0, 0)) <= 0.0f;
            std::cout << "Expected vs Computed: " << 1.0f * K << " vs " << c_m_n_device_result(0, 0)
                      << ((res_verified) ? " (PASSED!)" : " (FAILED!)") << std::endl;
        }

        res_verified = res_verified && ck::utils::check_err(c_m_n_device_result,
                                                            c_m_n_host_result,
                                                            "Error: Incorrect results!");

        if(config.verbosity > 0 && res_verified)
            std::cout << "Done." << std::endl;
    }
    else
    {
        if(config.verbosity > 0)
            std::cout << "Done." << std::endl;
    }

    if(config.time_kernel)
    {
        std::size_t flop = std::size_t(2) * M * N * K + M * K + K * N; // GEMM + A scale + B scale
        std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(CDataType) * M * N +
                                sizeof(XDataType) * (M * K + K * N) / Scale_Block_K;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << std::endl;
    }

    return res_verified;
}

template <typename ADataType,
          typename BDataType,
          typename XDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename CElementWiseOp,
          typename AccDataType,
          typename CShuffleDataType,
          ck::index_t MXVectorSize>
bool run_mx_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return parse_cmd_args(argc, argv, problem_size, config) &&
           run_mx_gemm<ADataType,
                       BDataType,
                       XDataType,
                       CDataType,
                       ALayout,
                       BLayout,
                       CLayout,
                       CElementWiseOp,
                       AccDataType,
                       CShuffleDataType,
                       MXVectorSize>(problem_size, config);
}
