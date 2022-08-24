// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include <vector>
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm_xdl_cshuffle.hpp"
#include "profiler/include/profile_batched_gemm_gemm_impl.hpp"

using ck::tensor_operation::device::GemmSpecialization;

template <ck::index_t N>
using I = ck::Number<N>;

using F16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
struct TestBatchedGemmGemm : public ::testing::Test
{
    using A0DataType = std::tuple_element_t<0, Tuple>;
    using B0DataType = std::tuple_element_t<1, Tuple>;
    using B1DataType = std::tuple_element_t<2, Tuple>;
    using C1DataType = std::tuple_element_t<3, Tuple>;
    using A0Layout   = std::tuple_element_t<4, Tuple>;
    using B0Layout   = std::tuple_element_t<5, Tuple>;
    using B1Layout   = std::tuple_element_t<6, Tuple>;
    using C1Layout   = std::tuple_element_t<7, Tuple>;

    std::vector<std::vector<int>> lengths_ = {
        {256, 256, 64, 64, 4},
        {256, 256, 128, 128, 4},
        {512, 512, 64, 64, 2},
        {512, 512, 128, 128, 2},
        {1024, 1024, 64, 64, 1},
        {1024, 1024, 128, 128, 1},
    };
    bool bench_  = false;
    bool verify_ = true;

    void RunSingle(int M, int N, int K, int O, int BatchCount)
    {
        bool pass = ck::profiler::profile_batched_gemm_gemm_impl<A0DataType,
                                                                 B0DataType,
                                                                 B1DataType,
                                                                 C1DataType,
                                                                 A0Layout,
                                                                 B0Layout,
                                                                 B1Layout,
                                                                 C1Layout>(
            verify_, 1, false, bench_, M, N, K, O, BatchCount);

        EXPECT_TRUE(pass);
    }

    void Run()
    {
        for(auto lengths : this->lengths_)
        {
            int M          = lengths[0];
            int N          = lengths[1];
            int K          = lengths[2];
            int O          = lengths[3];
            int BatchCount = lengths[4];

            this->RunSingle(M, N, K, O, BatchCount);
        }
    }
};

template <bool PadGemm0M, bool PadGemm0N, bool PadGemm0K, bool PadGemm1N>
struct DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using A0Layout = Row;
    using B0Layout = Col;
    using B1Layout = Row;
    using C1Layout = Row;

    using A0DataType        = F16;
    using B0DataType        = F16;
    using Acc0DataType      = float;
    using B1DataType        = F16;
    using Acc1DataType      = float;
    using C1ShuffleDataType = float;
    using C1DataType        = F16;

    using A0ElementOp = PassThrough;
    using B0ElementOp = PassThrough;
    using C0ElementOp = PassThrough;
    using B1ElementOp = PassThrough;
    using C1ElementOp = PassThrough;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;

    using DeviceGemmGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmGemm_Xdl_CShuffle<
        A0Layout,
        B0Layout,
        B1Layout,
        C1Layout,
        A0DataType,
        B0DataType,
        Acc0DataType,
        B1DataType,
        Acc1DataType,
        C1ShuffleDataType,
        C1DataType,
        A0ElementOp,
        B0ElementOp,
        C0ElementOp,
        B1ElementOp,
        C1ElementOp,
        PadGemm0M,
        PadGemm0N,
        PadGemm0K,
        PadGemm1N,
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
        8>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

    bool IsSupported(int M, int N, int K, int O)
    {
        auto gemm     = DeviceGemmGemmInstance{};
        auto invoker  = gemm.MakeInvoker();
        auto argument = gemm.MakeArgument(static_cast<A0DataType*>(nullptr),
                                          static_cast<B0DataType*>(nullptr),
                                          static_cast<B1DataType*>(nullptr),
                                          static_cast<C1DataType*>(nullptr),
                                          M,
                                          N,
                                          K,
                                          O,
                                          0,              // BatchCount
                                          0,              // StrideA
                                          0,              // StrideB0
                                          0,              // StrideB1
                                          0,              // StrideC
                                          0,              // BatchStrideA
                                          0,              // BatchStrideB0
                                          0,              // BatchStrideB1
                                          0,              // BatchStrideC
                                          PassThrough{},  // a_element_op
                                          PassThrough{},  // b0_element_op
                                          PassThrough{},  // acc0_element_op
                                          PassThrough{},  // b1_element_op
                                          PassThrough{}); // c_element_op

        return gemm.IsSupportedArgument(argument);
    }
};
