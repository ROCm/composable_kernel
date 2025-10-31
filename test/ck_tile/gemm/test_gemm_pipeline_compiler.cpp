// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

// Concise test suite for compiler validation.
// Covers essential combinations of data types, layouts, and pipeline types.

template <typename T>
class TestCkTileGemmCompiler : public TestCkTileGemmPipeline<T, TestCkTileGemmCompiler<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmCompiler

using CompilerTestTypes = ::testing::Types<
    // ============================================================================
    // KernelTypes with Mem pipeline
    // Parameters: ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType,
    //             CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize,
    //             N_TileSize, K_TileSize, Scheduler, PipelineType
    // ============================================================================
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,

    // KernelTypes with WMMA Mem pipeline
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
	std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,

    // KernelTypes with CompV3 pipeline
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV3>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV3>,

    // KernelTypes with CompV3 pipeline (WMMA)
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV3>,
	std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV3>,

    // KernelTypes with CompV4 pipeline
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,         CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,         CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,         CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,         CompV4>,

    // KernelTypes with CompV4 pipeline (WMMA)
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV4>,

    // KernelTypes with CompV6 pipeline
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV6>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV6>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV6>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV6>,

    // ============================================================================
    // KernelTypes with Persistent CompV3 pipeline
    // Additional Parameter: Persistent (Persistent/NonPersistent mode)
    // ============================================================================
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         CompV3, NonPersistent>,

    // KernelTypes with Persistent CompV3 pipeline (WMMA)
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         CompV3, NonPersistent>
>;

TYPED_TEST_SUITE(TestCkTileGemmCompiler, CompilerTestTypes);

// ============================================================================
// Test Cases
// ============================================================================

// Test 1: Single tile - validates basic kernel compilation and execution
TYPED_TEST(TEST_SUITE_NAME, SingleTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

// Test 2: Small M - validates edge cases with small batch sizes
TYPED_TEST(TEST_SUITE_NAME, SmallM)
{
    std::vector<int> Ms{1, 4};  // Minimal set for compiler check
    constexpr int N = 1024;
    constexpr int K = 256;

    for(int M : Ms)
    {
        if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                    ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
        }
        else
        {
            this->Run(M, N, K);
        }
    }
}

// Test 3: Regular size - validates typical production workload
TYPED_TEST(TEST_SUITE_NAME, Regular)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;

    this->Run(M, N, K);
}

// Test 4: Padded K - validates handling of non-aligned K dimension
TYPED_TEST(TEST_SUITE_NAME, PaddK)
{
    constexpr int M = 128;
    constexpr int N = 1024;
    constexpr int K = 432;  // Non-aligned K

    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME