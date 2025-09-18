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

// Wrapper class for AQuant testing
class AQuantTestWrapper : public TestCkTileGemmAQuant<std::tuple<
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
>> {
public:
    void TestBody() override {
        // Implementation of the required pure virtual function
        RunTest(128, 128, 128);
    }
};

// Test AQuant quantization
TEST(GemmQuantUnifiedTest, AQuantBasicTest) {
    AQuantTestWrapper test_wrapper;
    EXPECT_NO_THROW(test_wrapper.RunTest(128, 128, 128)) << "AQuant test failed";
}

// Wrapper class for BQuant testing
class BQuantTestWrapper : public TestCkTileGemmBQuant<std::tuple<
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
>> {
public:
    void TestBody() override {
        RunTest(128, 128, 128);
    }
};

// Test BQuant quantization
TEST(GemmQuantUnifiedTest, BQuantBasicTest) {
    BQuantTestWrapper test_wrapper;
    EXPECT_NO_THROW(test_wrapper.RunTest(128, 128, 128)) << "BQuant test failed";
}

// Wrapper class for RowColQuant testing
class RowColQuantTestWrapper : public TestCkTileGemmRowColQuant<std::tuple<
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
>> {
public:
    void TestBody() override {
        RunTest(128, 128, 128);
    }
};

// Test RowColQuant quantization
TEST(GemmQuantUnifiedTest, RowColQuantBasicTest) {
    RowColQuantTestWrapper test_wrapper;
    EXPECT_NO_THROW(test_wrapper.RunTest(128, 128, 128)) << "RowColQuant test failed";
}

// Wrapper class for TensorQuant testing
class TensorQuantTestWrapper : public TestCkTileGemmTensorQuant<std::tuple<
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
>> {
public:
    void TestBody() override {
        RunTest(128, 128, 128);
    }
};

// Test TensorQuant quantization
TEST(GemmQuantUnifiedTest, TensorQuantBasicTest) {
    TensorQuantTestWrapper test_wrapper;
    EXPECT_NO_THROW(test_wrapper.RunTest(128, 128, 128)) << "TensorQuant test failed";
}
