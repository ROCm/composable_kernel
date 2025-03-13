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
    int init_method     = 2;     // (0=no init, 1=integer value, 2=decimal value)
    bool time_kernel    = false; // (0=no, 1=yes)
    int verbosity       = 1;     // (0=no info, 1=verbose info)
};

struct ProblemSizeSplitK final
{

#if 0
    ck::index_t M = 256;
    ck::index_t N = 256;
    ck::index_t K = 384;
#else
    ck::index_t M                          = 3840;
    ck::index_t N                          = 4096;
    ck::index_t K                          = 4096;
#endif

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
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4: verbosity (0=no info, 1=verbose info)" << std::endl
                  << "arg5 to 10: M(256x), N(128x), K(32x), StrideA, StrideB, StrideC" << std::endl
                  << "arg11: KBatch" << std::endl;
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
          typename AElementOp,
          typename BElementOp,
          typename CElementOp,
          typename AccDataType,
          typename CShuffleDataType,
          ck::index_t MXVectorSize>
bool run_mx_gemm(const ProblemSizeSplitK& problem_size, const ExecutionConfig& config)
{

    static constexpr auto GemmSpec      = ck::tensor_operation::device::GemmSpecialization::Default;
    static constexpr auto BlkGemmPSched = ck::BlockGemmPipelineScheduler::Intrawave;
    static constexpr auto BlkGemmPVer =
        ck::BlockGemmPipelineVersion::v1; // can be v3 when the compiler bug is fixed.

    static constexpr ck::index_t ScaleBlockSize = MXVectorSize;

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

#if 1
    static constexpr ck::index_t KPerBlock = 64;
    using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
        ALayout,          // ALayout
        BLayout,          // BLayout
        CLayout,          // CLayout
        ADataType,        // ADataType
        XDataType,        // AScaleDataType
        BDataType,        // BDataType
        XDataType,        // BScaleDataType
        CDataType,        // CDataType
        AccDataType,      // GemmAccDataType
        CShuffleDataType, // CShuffleDataType
        AElementOp,       // AElementwiseOperation
        BElementOp,       // BElementwiseOperation
        CElementOp,       // CElementwiseOperation
        GemmSpec,         // GemmSpec
        MXVectorSize,     // ScaleBlockSize: Scaling block size
        256,              // BlockSize: Thread block size
        128,              // MPerBlock
        128,              // NPerBlock
        KPerBlock,        // KPerBlock
        16,               // AK1
        16,               // BK1
        32,               // MPerXDL
        32,               // NPerXDL
        2,                // MXdlPerWave
        2,                // NXdlPerWave
        S<4, 64, 1>,      // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
        2,                // ABlockTransferSrcVectorDim
        16,               // ABlockTransferSrcScalarPerVector
        16,               // ABlockTransferDstScalarPerVector_AK1
        false,            // ABlockLdsExtraM
        S<4, 64, 1>,      // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
        2,                // BBlockTransferSrcVectorDim
        16,               // BBlockTransferSrcScalarPerVector
        16,               // BBlockTransferDstScalarPerVector_BK1
        false,            // BBlockLdsExtraN
        1,                // CShuffleMXdlPerWavePerShuffle
        1,                // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>,   // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,                // CShuffleBlockTransferScalarPerVector_NPerBlock
        BlkGemmPSched,    // BlkGemmPipeSched
        BlkGemmPVer,      // BlkGemmPipelineVer
        ADataType,        // ComputeTypeA
        BDataType         // ComputeTypeB
        >;
#else
    static constexpr ck::index_t KPerBlock = 128;
    using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
        ALayout,          // ALayout
        BLayout,          // BLayout
        CLayout,          // CLayout
        ADataType,        // ADataType
        XDataType,        // AScaleDataType
        BDataType,        // BDataType
        XDataType,        // BScaleDataType
        CDataType,        // CDataType
        AccDataType,      // GemmAccDataType
        CShuffleDataType, // CShuffleDataType
        AElementOp,       // AElementwiseOperation
        BElementOp,       // BElementwiseOperation
        CElementOp,       // CElementwiseOperation
        GemmSpec,         // GemmSpec
        MXVectorSize,     // ScaleBlockSize: Scaling block size
        128,              // BlockSize: Thread block size
        16,               // MPerBlock
        32,               // NPerBlock
        KPerBlock,        // KPerBlock
        16,               // AK1
        16,               // BK1
        16,               // MPerXDL
        16,               // NPerXDL
        1,                // MXdlPerWave
        1,                // NXdlPerWave
        S<8, 16, 1>,      // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
        2,                // ABlockTransferSrcVectorDim
        16,               // ABlockTransferSrcScalarPerVector
        16,               // ABlockTransferDstScalarPerVector_AK1
        false,            // ABlockLdsExtraM
        S<8, 16, 1>,      // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
        2,                // BBlockTransferSrcVectorDim
        16,               // BBlockTransferSrcScalarPerVector
        16,               // BBlockTransferDstScalarPerVector_BK1
        false,            // BBlockLdsExtraN
        1,                // CShuffleMXdlPerWavePerShuffle
        1,                // CShuffleNXdlPerWavePerShuffle
        S<1, 16, 1, 8>,   // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4,                // CShuffleBlockTransferScalarPerVector_NPerBlock
        BlkGemmPSched,    // BlkGemmPipeSched
        BlkGemmPVer,      // BlkGemmPipelineVer
        ADataType,        // ComputeTypeA
        BDataType         // ComputeTypeB
        >;
#endif

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
        throw std::runtime_error("wrong! K must be multiple of ScaleBlockSize (16 or 32)");
    };

    auto Scale_Stride_AM = f_get_default_stride(M, K / ScaleBlockSize, -1, ALayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / ScaleBlockSize, N, -1, BLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<XDataType> a_m_k_scale(f_host_tensor_descriptor(
        M, K / ScaleBlockSize, Scale_Stride_AM, ALayout{})); // scales for A
    Tensor<XDataType> b_k_n_scale(f_host_tensor_descriptor(
        K / ScaleBlockSize, N, Scale_Stride_BN, BLayout{})); // scales for B

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

    case 2: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(a_m_k_scale);

        b_k_n.GenerateTensorValue(GeneratorTensor_3<ADataType>{-2.0, 2.0});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_3<XDataType>{-1.0f, 1.0f});
        break;

    case 10: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(0.5f)}(b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(2.0f)}(b_k_n_scale);
        if(config.verbosity > 0)
        {
            std::cout << "Init A = {1}" << std::endl;
            std::cout << "Init A scale = {1.0}" << std::endl;
            std::cout << "Init B = {0.5}" << std::endl;
            std::cout << "Init B scale = {2.0}" << std::endl;
            std::cout << "Expect C = {K}" << std::endl;
        }
        break;

    case 11: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(0.0f)}(b_k_n);
        // ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(b_k_n_scale);

        for(ck::index_t i = 0; i < K / ScaleBlockSize; i++)
        {
            for(ck::index_t j = 0; j < N / 4; j++)
            {
                auto j_offset = j * 4;
                if(i % 2 == 0)
                {
                    b_k_n_scale(i, j_offset + (0 + i) % 4) = ck::type_convert<XDataType>(1.0f / 4);
                    b_k_n_scale(i, j_offset + (1 + i) % 4) = ck::type_convert<XDataType>(1.0f / 2);
                    b_k_n_scale(i, j_offset + (2 + i) % 4) = ck::type_convert<XDataType>(1.0f);
                    b_k_n_scale(i, j_offset + (3 + i) % 4) = ck::type_convert<XDataType>(2.0f);
                }
                else
                {
                    b_k_n_scale(i, j_offset + (0 + i) % 4) = ck::type_convert<XDataType>(16.0f);
                    b_k_n_scale(i, j_offset + (1 + i) % 4) = ck::type_convert<XDataType>(8.0f);
                    b_k_n_scale(i, j_offset + (2 + i) % 4) = ck::type_convert<XDataType>(1.0f / 16);
                    b_k_n_scale(i, j_offset + (3 + i) % 4) = ck::type_convert<XDataType>(1.0f / 32);
                }
            }
        }

        {
            const ck::index_t n_freq   = 13;         // frequency of nonzero values in col(B)
            const ck::index_t pert_idx = 7 * n_freq; // location of perturbation

            for(ck::index_t i = 0; i < K; i++)
            {
                if(i % n_freq == 0)
                {
                    for(ck::index_t j = 0; j < N; j++)
                    {
                        float scale = ck::type_convert<float>(b_k_n_scale(i / ScaleBlockSize, j));
                        if(i == pert_idx)
                        {
                            b_k_n(i, j) = ck::type_convert<BDataType>(2.0f / scale);
                        }
                        else
                        {
                            b_k_n(i, j) = ck::type_convert<BDataType>(1.0f / scale);
                        }
                    }
                }
            }

            if(config.verbosity > 0)
            {
                std::cout << "Init A = {1}" << std::endl;
                std::cout << "Init A scale = {1.0}" << std::endl;
                std::cout << "Init B is real" << std::endl;
                std::cout << "Init B scale is real" << std::endl;
                std::cout << "Expect C = {"
                          << ((pert_idx < K) ? (K + n_freq - 1) / n_freq + 1
                                             : (K + n_freq - 1) / n_freq)
                          << "}" << std::endl;
            }
        }
        break;

    case 12: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(0.0f)}(b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(b_k_n_scale);

        for(ck::index_t i = 0; i < K / ScaleBlockSize; i++)
        {
            if(i % 2 == 0)
            {
                if(i % 4 == 0)
                    b_k_n_scale(i, 0) = ck::type_convert<XDataType>(1.0f / 4);
                else
                    b_k_n_scale(i, 0) = ck::type_convert<XDataType>(2.0f);
            }
            else
            {
                if(i % 4 == 1)
                    b_k_n_scale(i, 0) = ck::type_convert<XDataType>(16.0f);
                else
                    b_k_n_scale(i, 0) = ck::type_convert<XDataType>(1.0f / 32);
            }
        }

        {
            const ck::index_t n_freq   = 13;         // frequency of nonzero values in col(B)
            const ck::index_t pert_idx = 7 * n_freq; // location of perturbation

            for(ck::index_t i = 0; i < K; i++)
            {
                if(i % n_freq == 0)
                {
                    float scale = ck::type_convert<float>(b_k_n_scale(i / ScaleBlockSize, 0));
                    if(i == pert_idx)
                    {
                        b_k_n(i, 0) = ck::type_convert<BDataType>(2.0f / scale);
                    }
                    else
                    {
                        b_k_n(i, 0) = ck::type_convert<BDataType>(1.0f / scale);
                    }
                }
            }

            if(config.verbosity > 0)
            {
                std::cout << "Init A = {1}" << std::endl;
                std::cout << "Init A scale = {1.0}" << std::endl;
                std::cout << "Init B is real" << std::endl;
                std::cout << "Init B scale is real" << std::endl;
                if(config.init_method == 12)
                {
                    std::cout << "Expect C = {"
                              << ((pert_idx < K) ? (K + n_freq - 1) / n_freq + 1
                                                 : (K + n_freq - 1) / n_freq)
                              << "}" << std::endl;
                }
                else
                {
                    std::cout << "Expect C = {" << 2 << "}" << std::endl;
                }
            }
        }
        break;

    case 13: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(0.0f)}(a_m_k);
        for(ck::index_t j = 0; j < K; j++)
        {
            a_m_k(0, j) = ck::type_convert<ADataType>(1.0f);
        }
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(0.0f)}(b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(b_k_n_scale);

        {

#if 0
            std::set<int> col_ids = {74};
#else
            std::set<int> col_ids = {10, 31, 42, 103, 74, 205, 226, 187};
#endif
            for(auto col_id : col_ids)
            {
#if 1
                b_k_n_scale(0, col_id)     = ck::type_convert<XDataType>(1.0f / 4);
                b_k_n_scale(0, col_id + 1) = ck::type_convert<XDataType>(1.0f / 2);
                b_k_n_scale(1, col_id)     = ck::type_convert<XDataType>(2.0f / 1);
                // b_k_n_scale(5, col_id)     = ck::type_convert<XDataType>(-1.0f / 2);
                b_k_n_scale(11, col_id) = ck::type_convert<XDataType>(4.0f / 1);
#endif
                b_k_n(383, col_id) = ck::type_convert<BDataType>(-1.0f);

                for(size_t i = 00; i < 384; i += 7)
                {
                    auto coeff       = ((i / 7) % 2 == 0) ? 1.0f : -1.0f;
                    b_k_n(i, col_id) = ck::type_convert<BDataType>(coeff / 10.0f * i);
                }
            }
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
        // std::cout << device_op.GetTypeString() << std::endl << std::endl;
        // std::cout << device_op.GetObjectName().value() << std::endl;
        // std::cout << device_op.GetTemplateInfo().value() << std::endl << std::endl;
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

#if 1

        std::cout << "Submatrix of b_k_n (16x16):" << std::endl;
        for(int i = 0; i < 16; ++i)
        {
            for(int j = 0; j < 16; ++j)
            {
                std::cout << std::setw(11) << type_convert<float>(b_k_n(i, j));
            }
            // std::cout << "\t\t";
            // for(int j = 0; j < 16; ++j)
            // {
            //     std::cout << std::setw(9) << type_convert<float>(b_k_n(i + 128, j));
            // }

            std::cout << "\t\t";
            for(int j = 0; j < 16; ++j)
            {
                std::cout << std::setw(11) << type_convert<float>(b_k_n(i + 200, j));
            }

            std::cout << std::endl;
        }

        if(K < 600)
        {
            std::cout << "b_k_n(:,0):" << std::endl;
            for(int i = 0; i < K; ++i)
            {
                std::cout << type_convert<float>(b_k_n(i, 0)) << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Submatrix of b_k_n_scale (12x16):" << std::endl;
        for(int i = 0; i < 12; ++i)
        {
            for(int j = 0; j < 16; ++j)
            {
                std::cout << std::setw(11) << type_convert<float>(b_k_n_scale(i, j));
            }
            // std::cout << "\t\t";
            // for(int j = 0; j < 16; ++j)
            // {
            //     std::cout << std::setw(11) << type_convert<float>(b_k_n_scale(i, j + 128)) << "
            //     ";
            // }
            std::cout << "\t\t";
            for(int j = 0; j < 16; ++j)
            {
                std::cout << std::setw(11) << type_convert<float>(b_k_n_scale(i, j + 200)) << " ";
            }

            std::cout << std::endl;
        }
        std::cout << "Submatrix of c_m_n_device_result (16x16):" << std::endl;
        for(int i = 0; i < 16; ++i)
        {
            for(int j = 0; j < 16; ++j)
            {
                std::cout << std::setw(9) << type_convert<float>(c_m_n_device_result(i, j));
            }
            // std::cout << "\t\t";
            // for(int j = 0; j < 16; ++j)
            // {
            //     std::cout << std::setw(9) << type_convert<float>(c_m_n_device_result(i + 128,
            //     j));
            // }

            std::cout << "\t\t";
            for(int j = 0; j < 16; ++j)
            {
                std::cout << std::setw(9) << type_convert<float>(c_m_n_device_result(i + 200, j));
            }

            std::cout << std::endl;
        }

#endif

        if(config.init_method == 10)
        {
            auto expected = static_cast<float>(K);
            auto computed = type_convert<float>(c_m_n_device_result(1, 12));

            res_verified = res_verified && std::abs(expected - computed) <= 0.0f;
            std::cout << "\nExpected vs Computed: " << expected << " vs " << computed
                      << ((res_verified) ? " (PASSED!)" : " (FAILED!)") << std::endl
                      << std::endl;
        }
        else if(config.init_method == 11)
        {

#if 1
            std::cout << "Submatrix of b_k_n (16x16):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5) << type_convert<float>(b_k_n(i, j));
                }
                std::cout << "\t\t";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5) << type_convert<float>(b_k_n(i + 128, j));
                }

                std::cout << "\t\t";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5) << type_convert<float>(b_k_n(i + 200, j));
                }

                std::cout << std::endl;
            }
#if 1
            std::cout << "b_k_n(:,0):" << std::endl;
            for(int i = 0; i < K; ++i)
            {
                std::cout << type_convert<float>(b_k_n(i, 0)) << " ";
            }
            std::cout << std::endl;
#endif

            std::cout << "Submatrix of b_k_n_scale (12x16):" << std::endl;
            for(int i = 0; i < 12; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(10) << type_convert<float>(b_k_n_scale(i, j));
                }
                std::cout << "\t\t";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(10) << type_convert<float>(b_k_n_scale(i, j + 128))
                              << " ";
                }
                // std::cout << "\t\t";
                // for(int j = 0; j < 16; ++j)
                // {
                //     std::cout << std::setw(10) << type_convert<float>(b_k_n_scale(i, j + 200))
                //               << " ";
                // }

                std::cout << std::endl;
            }
#endif
#if 1
            std::cout << "Submatrix of c_m_n_device_result (16x16):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5) << type_convert<float>(c_m_n_device_result(i, j));
                }
                std::cout << "\t\t";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5)
                              << type_convert<float>(c_m_n_device_result(i + 128, j));
                }

                std::cout << "\t\t";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5)
                              << type_convert<float>(c_m_n_device_result(i + 200, j));
                }

                std::cout << std::endl;
            }
#endif

            auto computed = type_convert<float>(c_m_n_device_result(1, 12));

            std::cout << "\nComputed: " << computed << std::endl << std::endl;
        }
        else if(config.init_method == 12 || config.init_method == 13)
        {
#if 0
            std::cout << "Submatrix of a_m_k (16x16):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << type_convert<float>(a_m_k(i, j)) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Submatrix of a_m_k_scale (16x3):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    std::cout << type_convert<float>(a_m_k_scale(i, j)) << " ";
                }
                std::cout << std::endl;
            }
#endif
#if 0
            std::cout << "Submatrix of b_k_n (16x16):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << type_convert<float>(b_k_n(i, j)) << " ";
                }
                std::cout << "     ";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << type_convert<float>(b_k_n(i + ScaleBlockSize, j)) << " ";
                }

                std::cout << "     ";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << type_convert<float>(b_k_n(i + 2 * ScaleBlockSize, j)) << " ";
                }

                std::cout << std::endl;
            }
#endif
#if 1
            std::cout << "b_k_n(:,0):" << std::endl;
            for(int i = 0; i < K; ++i)
            {
                std::cout << type_convert<float>(b_k_n(i, 0)) << " ";
            }
            std::cout << std::endl;
#endif
            std::cout << "Submatrix of b_k_n_scale (12x16):" << std::endl;
            for(int i = 0; i < 12; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5) << type_convert<float>(b_k_n_scale(i, j));
                }
                std::cout << "\t\t";
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(5) << type_convert<float>(b_k_n_scale(i, j + 64)) << " ";
                }

                std::cout << std::endl;
            }
#if 0
            std::cout << "Submatrix of c_m_n_device_result (16x16):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << type_convert<float>(c_m_n_device_result(i, j)) << " ";
                }
                std::cout << std::endl;
            }
#endif
#if 1
            std::cout << "c_m_n_device_result(:,0):" << std::endl;
            for(int i = 0; i < M; ++i)
            {
                std::cout << type_convert<float>(c_m_n_device_result(i, 0)) << " ";
            }
            std::cout << std::endl;
            // std::cout << "c_m_n_device_result(:,1):" << std::endl;
            // for(int i = 0; i < M; ++i)
            // {
            //     std::cout << type_convert<float>(c_m_n_device_result(i, 1)) << " ";
            // }
            // std::cout << std::endl;
#endif

            auto computed = type_convert<float>(c_m_n_device_result(0, 0));
            std::cout << "\nComputed: " << computed << std::endl << std::endl;
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
                                sizeof(XDataType) * (M * K + K * N) / ScaleBlockSize;

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
           run_mx_gemm<ADataType,
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
