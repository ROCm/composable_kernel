// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include <vector>
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "profiler/include/profile_batched_gemm_masking_scale_softmax_gemm_permute_impl.hpp"
using ck::tensor_operation::device::GemmSpecialization;

template <ck::index_t N>
using I = ck::Number<N>;

using F16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
struct TestBatchedGemmMaskingScaleSoftmaxGemmPermute : public ::testing::Test
{
    using ADataType             = std::tuple_element_t<0, Tuple>;
    using B0DataType            = std::tuple_element_t<1, Tuple>;
    using B1DataType            = std::tuple_element_t<2, Tuple>;
    using CDataType             = std::tuple_element_t<3, Tuple>;
    using ALayout               = std::tuple_element_t<4, Tuple>;
    using B0Layout              = std::tuple_element_t<5, Tuple>;
    using B1Layout              = std::tuple_element_t<6, Tuple>;
    using CPermuteNumDims_G_M_O = std::tuple_element_t<7, Tuple>;

    std::vector<std::vector<int>> lengths_ = {
        {256, 256, 64, 64, 6, 4},
        {256, 256, 128, 128, 4, 6},
        {512, 512, 64, 64, 3, 2},
        {512, 512, 128, 128, 2, 3},
        {1024, 1024, 64, 64, 3, 1},
        {1024, 1024, 128, 128, 1, 1},
    };
    bool bench_  = false;
    bool verify_ = true;

    void RunSingle(int M, int N, int K, int O, int G0, int G1)
    {
        bool pass = ck::profiler::profile_batched_gemm_masking_scale_softmax_gemm_permute_impl<
            ADataType,
            B0DataType,
            B1DataType,
            CDataType,
            ALayout,
            B0Layout,
            B1Layout,
            CPermuteNumDims_G_M_O>(verify_, 1, false, bench_, M, N, K, O, G0, G1);

        EXPECT_TRUE(pass);
    }

    void Run()
    {
        for(auto lengths : this->lengths_)
        {
            int M  = lengths[0];
            int N  = lengths[1];
            int K  = lengths[2];
            int O  = lengths[3];
            int G0 = lengths[4];
            int G1 = lengths[5];

            this->RunSingle(M, N, K, O, G0, G1);
        }
    }
};

template <GemmSpecialization GemmSpec>
struct DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using Scale       = ck::tensor_operation::element_wise::Scale;

    using ALayout  = Row;
    using B0Layout = Col;
    using B1Layout = Row;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;
    using CPermuteNumDims_G_M_O =
        S<2, 1, 1>; // "using CLayout = Row" has been replaced by CPermuteNumDims_G_M_O

    using ADataType        = F16;
    using B0DataType       = F16;
    using B1DataType       = F16;
    using AccDataType      = float;
    using CShuffleDataType = F16;
    using CDataType        = F16;

    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using Acc0ElementOp = Scale;
    using B1ElementOp   = PassThrough;
    using CElementOp    = PassThrough;

    // static constexpr auto GemmSpec = std::tuple_element_t<0, Tuple>::value;

    using DeviceGemmGemmInstance =
        ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
            ALayout,
            B0Layout,
            B1Layout,
            CPermuteNumDims_G_M_O,
            ADataType,
            B0DataType,
            B1DataType,
            CDataType,
            AccDataType,
            CShuffleDataType,
            AElementOp,
            B0ElementOp,
            Acc0ElementOp,
            B1ElementOp,
            CElementOp,
            GemmSpec,
            1,
            256,
            128,         // MPerBlock
            128,         // NPerBlock
            32,          // KPerBlock
            128,         // Gemm1NPerBlock
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
            8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
            true>;          // Masking

    bool IsSupported(int M, int N, int K, int O)
    {
        auto gemm     = DeviceGemmGemmInstance{};
        auto invoker  = gemm.MakeInvoker();
        auto argument = gemm.MakeArgument(static_cast<ADataType*>(nullptr),
                                          static_cast<B0DataType*>(nullptr),
                                          static_cast<B1DataType*>(nullptr),
                                          static_cast<CDataType*>(nullptr),
                                          M,
                                          N,
                                          K,
                                          O,
                                          0,              // BatchCount
                                          {0, 0, M, O},   // gs ms ns lengths
                                          {0, O, 0, 1},   // gs ms ns strides
                                          0,              // StrideA
                                          0,              // StrideB0
                                          0,              // StrideB1
                                          0,              // BatchStrideA
                                          0,              // BatchStrideB0
                                          0,              // BatchStrideB1
                                          PassThrough{},  // a_element_op
                                          PassThrough{},  // b0_element_op
                                          Scale{1.f},     // acc0_element_op
                                          PassThrough{},  // b1_element_op
                                          PassThrough{}); // c_element_op

        return gemm.IsSupportedArgument(argument);
    }
};
