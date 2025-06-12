// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_multi_d_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;
using F8   = ck_tile::fp8_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypesAdd = ::testing::Types<
    //          ALayout, BLayout, CLayout, D0Layout, D1Layout, ADataType, BDataType, D0DataType,  D1DataType, AccDataType, CDataType, CDElementWiseFn
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F32,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          BF16,       BF16,       F32,      F32,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F32,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F16,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,      F32,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           F8,         F8,         F32,      F16,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Col,     Col,      Col,      F16,       F16,          F32,        F32,        F32,      F32,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Col,     Col,      Col,      F16,       F16,          F32,        F32,        F32,      F32,     ck_tile::element_wise::ElementWiseAdd>,
    std::tuple<    Row,     Col,     Col,     Col,      Col,      F8,        F8,           F8,         F8,         F32,      F32,     ck_tile::element_wise::ElementWiseAdd>
    >;

using KernelTypesMultiply = ::testing::Types<
    //          ALayout, BLayout, CLayout, D0Layout, D1Layout, ADataType, BDataType, D0DataType,  D1DataType, AccDataType, CDataType, CDElementWiseFn
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F16,        F16,        F32,      F16,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          BF16,       BF16,       F32,      F32,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F32,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F16,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,      F32,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           F8,         F8,         F32,      F32,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Col,     Col,      Col,      F16,       F16,          F32,        F32,        F32,      F32,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Col,     Col,      Col,      F16,       F16,          F32,        F32,        F32,      F32,     ck_tile::element_wise::MultiplyMultiply>,
    std::tuple<    Row,     Col,     Col,     Col,      Col,      F8,        F8,           F32,        F32,        F32,      F32,     ck_tile::element_wise::MultiplyMultiply>
    >;
// clang-format on

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_256x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_512x256x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_512x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_256x256x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_512x768x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_512x1280x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_256x1280x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_768x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_1280x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDAddKBatch1_1280x256x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x256x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x256x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x768x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x1280x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x1280x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_768x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x512x256 : public TestCkTileGemmMultiD<T>
{
};

template <typename T>
class TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x256x256 : public TestCkTileGemmMultiD<T>
{
};

TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_256x512x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_512x256x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_512x512x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_256x256x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_512x768x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_512x1280x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_256x1280x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_768x512x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_1280x512x256, KernelTypesAdd);
TYPED_TEST_SUITE(TestCkTileGemmMultiDAddKBatch1_1280x256x256, KernelTypesAdd);

TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x512x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x256x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x512x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x256x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x768x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x1280x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x1280x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_768x512x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x512x256, KernelTypesMultiply);
TYPED_TEST_SUITE(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x256x256, KernelTypesMultiply);

#include "test_gemm_multi_d_ut_cases.inc"
