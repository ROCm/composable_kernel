// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_util.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using F4  = ck_tile::pk_fp4_t;
using F8  = ck_tile::fp8_t;
using F6  = ck_tile::pk_fp6x16_t;

// clang-format off
using MxTypes = ::testing::Types<std::tuple<F4, F4, MX_GemmConfig16,         Row, Col, Row>,
                                 std::tuple<F4, F4, MX_GemmConfigEightWaves, Row, Col, Row>,
                                 std::tuple<F8, F8, MX_GemmConfig16,         Row, Col, Row>,
                                 std::tuple<F8, F8, MX_GemmConfigEightWaves, Row, Col, Row>>;

// Preshuffle configs 
using MxTypesPreshuffle = ::testing::Types<
    std::tuple<F4, F4, MXfp4_GemmConfig_Preshuffle, Row, Col, Row>,
    std::tuple<F8, F8, MXfp8_GemmConfig_Preshuffle, Row, Col, Row>>;
// clang-format on

template <typename TypeParam>
class TestMxGemm : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemm, MxTypes);

TYPED_TEST(TestMxGemm, Default)
{
    // No M/N/K padding so we use 128x256x256 as smallest dimensions
    this->Run(128, 256, 256);
    this->Run(256, 256, 512);
    this->Run(1024, 1024, 1024);
}

// Preshuffle tests
template <typename TypeParam>
class TestMxGemmPreshuffle : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemmPreshuffle, MxTypesPreshuffle);

TYPED_TEST(TestMxGemmPreshuffle, Default)
{
    this->Run(128, 512, 256);
    this->Run(256, 512, 512);
    this->Run(1024, 1024, 1024);
}
