// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_util.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using MxFp4Types = ::testing::Types<
    std::tuple<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, MXfp4_GemmConfig16, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmFp4 : public TestMxGemmUtil<std::tuple_element_t<0, TypeParam>,
                                            std::tuple_element_t<1, TypeParam>,
                                            std::tuple_element_t<2, TypeParam>,
                                            std::tuple_element_t<3, TypeParam>,
                                            std::tuple_element_t<4, TypeParam>,
                                            std::tuple_element_t<5, TypeParam>>
{
};

TYPED_TEST_SUITE(TestMxGemmFp4, MxFp4Types);

TYPED_TEST(TestMxGemmFp4, BasicSizes)
{
    this->Run(64, 64, 256);
    this->Run(128, 128, 256);
    this->Run(64, 128, 512);
}
