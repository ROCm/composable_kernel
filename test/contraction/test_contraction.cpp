// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <initializer_list>
#include <vector>
#include <tuple>
#include <gtest/gtest.h>

#include "profiler/profile_contraction_impl.hpp"

using F32 = float;
using F64 = double;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

struct MemoryParams
{
    std::vector<ck::index_t> M;
    std::vector<ck::index_t> N;
    std::vector<ck::index_t> K;
    std::vector<ck::index_t> StridesA;
    std::vector<ck::index_t> StridesB;
    std::vector<ck::index_t> StridesC;
    std::vector<ck::index_t> StridesD;
};

template <typename Tuple>
class TestContraction : public ::testing::Test
{
    protected:
    using ALayout        = std::tuple_element_t<0, Tuple>;
    using BLayout        = std::tuple_element_t<1, Tuple>;
    using CDLayout       = std::tuple_element_t<2, Tuple>;
    using DataType       = std::tuple_element_t<3, Tuple>;
    using DTupleDataType = std::tuple_element_t<4, Tuple>;
    using CDElementOp    = std::tuple_element_t<5, Tuple>;

    std::vector<MemoryParams> list_of_memory_params = {{{32, 32},
                                                        {32, 32},
                                                        {32, 32},
                                                        {32768, 1024, 32, 1},
                                                        {32768, 1024, 32, 1},
                                                        {32768, 1024, 32, 1},
                                                        {32768, 1024, 32, 1}},
                                                       {{16, 16},
                                                        {32, 32},
                                                        {16, 16},
                                                        {4096, 256, 16, 1},
                                                        {16, 1, 8192, 256},
                                                        {16384, 1024, 32, 1},
                                                        {16384, 1024, 32, 1}}};

    std::vector<ck::index_t> init_methods = {0, 1, 2};
    std::unique_ptr<CDElementOp> p_cd_element_op;
    void Run()
    {
        for(auto& memory_params : list_of_memory_params)
        {
            for(const ck::index_t init_method : init_methods)
            {
                bool pass =
                    ck::profiler::profile_contraction_impl<ALayout,
                                                           BLayout,
                                                           CDLayout,
                                                           DataType,
                                                           DTupleDataType,
                                                           CDElementOp>(true /*do_verification*/,
                                                                        init_method,
                                                                        false /*do_logs*/,
                                                                        false /*time_kernel*/,
                                                                        *p_cd_element_op,
                                                                        memory_params.M,
                                                                        memory_params.N,
                                                                        memory_params.K,
                                                                        memory_params.StridesA,
                                                                        memory_params.StridesB,
                                                                        memory_params.StridesC,
                                                                        memory_params.StridesD);
                EXPECT_TRUE(pass);
            }
        }
    }
};

template <typename Tuple>
class TestContractionScale : public TestContraction<Tuple>
{
};

template <typename Tuple>
class TestContractionBilinear : public TestContraction<Tuple>
{
};

using BilinearKernelTypes =
    ::testing::Types<std::tuple<Row, Row, Row, F32, ck::Tuple<F32>, Bilinear>,
                     std::tuple<Row, Col, Row, F32, ck::Tuple<F32>, Bilinear>,
                     std::tuple<Col, Row, Row, F32, ck::Tuple<F32>, Bilinear>,
                     std::tuple<Col, Col, Row, F32, ck::Tuple<F32>, Bilinear>,
                     std::tuple<Row, Row, Row, F64, ck::Tuple<F32>, Bilinear>,
                     std::tuple<Row, Col, Row, F64, ck::Tuple<F32>, Bilinear>,
                     std::tuple<Col, Row, Row, F64, ck::Tuple<F32>, Bilinear>,
                     std::tuple<Col, Col, Row, F64, ck::Tuple<F32>, Bilinear>>;

using ScaleKernelTypes = ::testing::Types<std::tuple<Row, Row, Row, F32, ck::Tuple<>, Scale>,
                                          std::tuple<Row, Col, Row, F32, ck::Tuple<>, Scale>,
                                          std::tuple<Col, Row, Row, F32, ck::Tuple<>, Scale>,
                                          std::tuple<Col, Col, Row, F32, ck::Tuple<>, Scale>,
                                          std::tuple<Row, Row, Row, F64, ck::Tuple<>, Scale>,
                                          std::tuple<Row, Col, Row, F64, ck::Tuple<>, Scale>,
                                          std::tuple<Col, Row, Row, F64, ck::Tuple<>, Scale>,
                                          std::tuple<Col, Col, Row, F64, ck::Tuple<>, Scale>>;

TYPED_TEST_SUITE(TestContractionBilinear, BilinearKernelTypes);
TYPED_TEST_SUITE(TestContractionScale, ScaleKernelTypes);

TYPED_TEST(TestContractionBilinear, bilinear)
{
    this->p_cd_element_op = std::make_unique<Bilinear>(1.f, 1.f);
    this->Run();
    this->p_cd_element_op = std::make_unique<Bilinear>(-0.5f, 0.5f);
    this->Run();
}

TYPED_TEST(TestContractionScale, scale)
{
    this->p_cd_element_op = std::make_unique<Scale>(1.f);
    this->Run();
    this->p_cd_element_op = std::make_unique<Scale>(0.5f);
    this->Run();
}
