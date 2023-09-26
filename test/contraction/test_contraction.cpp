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
#include "profiler/profile_contraction_utils.hpp"

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;
using F64  = double;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

struct Dimensions
{
    std::vector<ck::index_t> M;
    std::vector<ck::index_t> N;
    std::vector<ck::index_t> K;
};

template <typename Tuple>
class TestContraction : public ::testing::Test
{
    protected:
    using ALayout         = std::tuple_element_t<0, Tuple>;
    using BLayout         = std::tuple_element_t<1, Tuple>;
    using CDLayout        = std::tuple_element_t<2, Tuple>;
    using DataType        = std::tuple_element_t<3, Tuple>;
    using DTupleDataType  = std::tuple_element_t<4, Tuple>;
    using ComputeDataType = std::tuple_element_t<5, Tuple>;
    using CDElementOp     = std::tuple_element_t<6, Tuple>;

    std::vector<Dimensions> dimension_list = {{{32, 32}, {32, 32}, {32, 32}},
                                              {{16, 16}, {32, 32}, {16, 16}}};

    std::vector<ck::index_t> init_methods = {1, 2};
    std::unique_ptr<CDElementOp> p_cd_element_op;

    void Run()
    {
        for(auto& dimension_params : dimension_list)
        {
            std::vector<ck::index_t> StridesA;
            std::vector<ck::index_t> StridesB;
            std::vector<ck::index_t> StridesC;
            std::vector<ck::index_t> StridesD;

            const auto& M = dimension_params.M;
            const auto& N = dimension_params.N;
            const auto& K = dimension_params.K;

            assign_default_strides(ALayout{}, StridesA, {M[0], M[1], K[0], K[1]});
            assign_default_strides(BLayout{}, StridesB, {N[0], N[1], K[0], K[1]});
            assign_default_strides(CDLayout{}, StridesC, {M[0], M[1], N[0], N[1]});
            assign_default_strides(CDLayout{}, StridesD, {M[0], M[1], N[0], N[1]});

            for(const ck::index_t init_method : init_methods)
            {
                bool pass =
                    ck::profiler::profile_contraction_impl<ALayout,
                                                           BLayout,
                                                           CDLayout,
                                                           DataType,
                                                           ComputeDataType,
                                                           DTupleDataType,
                                                           CDElementOp>(true /*do_verification*/,
                                                                        init_method,
                                                                        false /*do_logs*/,
                                                                        false /*time_kernel*/,
                                                                        *p_cd_element_op,
                                                                        dimension_params.M,
                                                                        dimension_params.N,
                                                                        dimension_params.K,
                                                                        StridesA,
                                                                        StridesB,
                                                                        StridesC,
                                                                        StridesD);
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

#define ALL_LAYOUT_COMBINATIONS(dt, tuple_dt, compute_dt, op)    \
    std::tuple<Row, Row, Row, dt, tuple_dt, compute_dt, op>,     \
        std::tuple<Row, Col, Row, dt, tuple_dt, compute_dt, op>, \
        std::tuple<Col, Row, Row, dt, tuple_dt, compute_dt, op>, \
        std::tuple<Col, Col, Row, dt, tuple_dt, compute_dt, op>

using BilinearKernelTypes =
    ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<F32>, F32, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<F64>, F64, Bilinear)>;

using ScaleKernelTypes = ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<>, F32, Scale),
                                          ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<>, F64, Scale)>;

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

template <typename Tuple>
class TestContractionScaleMixedPrecision : public TestContraction<Tuple>
{
};

template <typename Tuple>
class TestContractionBilinearMixedPrecision : public TestContraction<Tuple>
{
};

using BilinearKernelTypesMixedPrecision =
    ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<F32>, F16, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<F32>, BF16, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<F64>, F32, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F16, ck::Tuple<F16>, F32, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(BF16, ck::Tuple<BF16>, F32, Bilinear)>;

using ScaleKernelTypesMixedPrecision =
    ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<>, F16, Scale),
                     ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<>, BF16, Scale),
                     ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<>, F32, Scale),
                     ALL_LAYOUT_COMBINATIONS(F16, ck::Tuple<>, F32, Scale),
                     ALL_LAYOUT_COMBINATIONS(BF16, ck::Tuple<>, F32, Scale)>;

TYPED_TEST_SUITE(TestContractionBilinearMixedPrecision, BilinearKernelTypesMixedPrecision);
TYPED_TEST_SUITE(TestContractionScaleMixedPrecision, ScaleKernelTypesMixedPrecision);

TYPED_TEST(TestContractionBilinearMixedPrecision, bilinear)
{
    this->p_cd_element_op = std::make_unique<Bilinear>(1.f, 1.f);
    this->Run();
    this->p_cd_element_op = std::make_unique<Bilinear>(-0.5f, 0.5f);
    this->Run();
}

TYPED_TEST(TestContractionScaleMixedPrecision, scale)
{
    this->p_cd_element_op = std::make_unique<Scale>(1.f);
    this->Run();
    this->p_cd_element_op = std::make_unique<Scale>(0.5f);
    this->Run();
}
