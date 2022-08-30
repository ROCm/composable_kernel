// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_batched_gemm_gemm_util.hpp"

template <typename Tuple>
class TestBatchedGemmGemmFP16 : public TestBatchedGemmGemm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F16, F16, Row, Col, Row, Row>
    >;
// clang-format on

TYPED_TEST_SUITE(TestBatchedGemmGemmFP16, KernelTypes);

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16) { this->Run(); }

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {136, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 136, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 40, 128, 1},
        {128, 128, 136, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 136, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {129, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 129, 32, 128, 1},
    };
    this->Run();
}

// Currently expected that no kernels can support this case
TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 33, 128, 1},
        {128, 128, 129, 128, 1},
    };
    this->Run();
}

// If kernel B1Layout is RowMajor, expect not to support odd O size
TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 129, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, DISABLED_Bench_FP16)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {256, 256, 64, 64, 768},
        {256, 256, 128, 128, 768},
        {512, 512, 64, 64, 768},
        {512, 512, 128, 128, 768},
        {1024, 1024, 64, 64, 768},
        {1024, 1024, 128, 128, 768},
        {2048, 2048, 64, 64, 768},
        {2048, 2048, 128, 128, 768},
        {4096, 4096, 64, 64, 768},
        {4096, 4096, 128, 128, 768},
        {401408, 256, 64, 64, 1},
        {100352, 512, 128, 128, 1},
        {25088, 1024, 256, 256, 1},
        {6272, 2048, 512, 512, 1},
    };
    this->bench_  = true;
    this->verify_ = false;
    this->Run();
}

using ck::tensor_operation::device::GemmSpecialization;

TEST(TestBatchedGemmGemmInterface, GemmSpecializationSizeMatch)
{
    int P = 120; // requires padding
    int Q = 128; // do not require padding

    // IsSupported(M, N, K, O)
    // clang-format off
    // ############################################################     Pad|    Pad|    Pad|    Pad|
    // ############################################################  Gemm0M| Gemm0N| Gemm0K| Gemm1N|
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,  false,  false,  false>{}.IsSupported(Q, Q, Q, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,  false,  false,  false>{}.IsSupported(P, Q, Q, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,   true,  false,  false>{}.IsSupported(Q, P, Q, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,  false,   true,  false>{}.IsSupported(Q, Q, P, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,   true,  false,  false>{}.IsSupported(P, P, Q, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,  false,   true,  false>{}.IsSupported(P, Q, P, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,   true,   true,  false>{}.IsSupported(Q, P, P, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,   true,   true,  false>{}.IsSupported(P, P, P, Q)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,  false,  false,   true>{}.IsSupported(Q, Q, Q, P)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,  false,  false,   true>{}.IsSupported(P, Q, Q, P)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,   true,  false,   true>{}.IsSupported(Q, P, Q, P)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,  false,   true,   true>{}.IsSupported(Q, Q, P, P)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,   true,  false,   true>{}.IsSupported(P, P, Q, P)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,  false,   true,   true>{}.IsSupported(P, Q, P, P)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  false,   true,   true,   true>{}.IsSupported(Q, P, P, P)));
    EXPECT_TRUE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<   true,   true,   true,   true>{}.IsSupported(P, P, P, P)));
    // clang-format on
}

TEST(TestBatchedGemmGemmInterface, GemmSpecializationSizeMismatch)
{
    // IsSupported(M, N, K, O)
    // clang-format off
    // ############################################################     Pad|    Pad|    Pad|    Pad|
    // ############################################################  Gemm0M| Gemm0N| Gemm0K| Gemm1N|
    EXPECT_FALSE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128< false,  false,  false,  false>{}.IsSupported(128, 128, 120, 128)));
    EXPECT_FALSE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  true,   true,   true,  false>{}.IsSupported(128, 128, 128, 120)));
    // Kernel can't support odd K because K must be integer multiples of K1 values of either A or B
    // ############################################################     Pad|    Pad|    Pad|    Pad|
    // ############################################################  Gemm0M| Gemm0N| Gemm0K| Gemm1N|
    EXPECT_FALSE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  true,   true,   true,   true>{}.IsSupported(128, 128, 129, 128)));
    // Kernel can't support odd O size because it must satisfy SizeO % B1SrcScalarPerVector == 0
    // ############################################################     Pad|    Pad|    Pad|    Pad|
    // ############################################################  Gemm0M| Gemm0N| Gemm0K| Gemm1N|
    EXPECT_FALSE((DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<  true,   true,   true,   true>{}.IsSupported(128, 128, 128, 129)));
    // clang-format on
}
