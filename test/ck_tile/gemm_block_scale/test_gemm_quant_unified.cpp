// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "test_gemm_quant_fixtures.hpp"
#include <gtest/gtest.h>

using namespace ck_tile;

// Simple GemmConfig for testing
struct SimpleGemmConfig {
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;
    
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

// Test fixtures for each quantization type
using AQuantTestFixture = TestCkTileGemmAQuant<std::tuple<
    ck_tile::tensor_layout::gemm::RowMajor,       // ALayout
    ck_tile::tensor_layout::gemm::ColumnMajor,    // BLayout  
    ck_tile::tensor_layout::gemm::RowMajor,       // CLayout
    ck_tile::fp8_t,                               // ADataType
    ck_tile::fp8_t,                               // BDataType
    float,                                        // AccDataType
    ck_tile::half_t,                              // CDataType
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>, // QuantType
    SimpleGemmConfig,                             // GemmConfig
    std::integral_constant<uint32_t, 128>         // QuantGroupSize
>>;

using BQuantTestFixture = TestCkTileGemmBQuant<std::tuple<
    ck_tile::tensor_layout::gemm::RowMajor,       // ALayout
    ck_tile::tensor_layout::gemm::ColumnMajor,    // BLayout  
    ck_tile::tensor_layout::gemm::RowMajor,       // CLayout
    ck_tile::fp8_t,                               // ADataType
    ck_tile::fp8_t,                               // BDataType
    float,                                        // AccDataType
    ck_tile::half_t,                              // CDataType
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::BQuantGrouped>, // QuantType
    SimpleGemmConfig,                             // GemmConfig
    std::integral_constant<uint32_t, 128>         // QuantGroupSize
>>;

using RowColQuantTestFixture = TestCkTileGemmRowColQuant<std::tuple<
    ck_tile::tensor_layout::gemm::RowMajor,       // ALayout
    ck_tile::tensor_layout::gemm::ColumnMajor,    // BLayout  
    ck_tile::tensor_layout::gemm::RowMajor,       // CLayout
    ck_tile::fp8_t,                               // ADataType
    ck_tile::fp8_t,                               // BDataType
    float,                                        // AccDataType
    ck_tile::half_t,                              // CDataType
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::RowColQuant>, // QuantType
    SimpleGemmConfig,                             // GemmConfig
    std::integral_constant<uint32_t, 128>         // QuantGroupSize
>>;

using TensorQuantTestFixture = TestCkTileGemmTensorQuant<std::tuple<
    ck_tile::tensor_layout::gemm::RowMajor,       // ALayout
    ck_tile::tensor_layout::gemm::ColumnMajor,    // BLayout  
    ck_tile::tensor_layout::gemm::RowMajor,       // CLayout
    ck_tile::fp8_t,                               // ADataType
    ck_tile::fp8_t,                               // BDataType
    float,                                        // AccDataType
    ck_tile::half_t,                              // CDataType
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::TensorQuant>, // QuantType
    SimpleGemmConfig,                             // GemmConfig
    std::integral_constant<uint32_t, 128>         // QuantGroupSize
>>;

// Test AQuant quantization
TEST_F(AQuantTestFixture, BasicTest) {
    EXPECT_NO_THROW(RunTest(128, 128, 128)) << "AQuant test failed";
}

// Test BQuant quantization  
TEST_F(BQuantTestFixture, BasicTest) {
    EXPECT_NO_THROW(RunTest(128, 128, 128)) << "BQuant test failed";
}

// Test RowColQuant quantization
TEST_F(RowColQuantTestFixture, BasicTest) {
    EXPECT_NO_THROW(RunTest(128, 128, 128)) << "RowColQuant test failed";
}

// Test TensorQuant quantization
TEST_F(TensorQuantTestFixture, BasicTest) {
    EXPECT_NO_THROW(RunTest(128, 128, 128)) << "TensorQuant test failed";
}
