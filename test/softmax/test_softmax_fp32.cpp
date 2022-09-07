// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

template <typename Tuple>
class TestSoftmaxFP32 : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<float, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<4>, I<8>>, // mixed precision
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<2>, I<256>, I<8>, I<32>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<2>, I<256>, I<4>, I<64>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<3>, I<256>, I<8>, I<32>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<3>, I<256>, I<4>, I<64>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<3>, I<256>, I<2>, I<128>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<4>, I<4>>
    >;
// clang-format on
TYPED_TEST_SUITE(TestSoftmaxFP32, KernelTypes);
TYPED_TEST(TestSoftmaxFP32, Test_FP32) { this->Run(); }

template <typename Tuple>
class TestSoftmaxFP32v2 : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypesReduceSingleDim = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<float, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<4>, I<8>>, // mixed precision
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<4>, I<4>>
    >;
// clang-format on
TYPED_TEST_SUITE(TestSoftmaxFP32v2, KernelTypesReduceSingleDim);

TYPED_TEST(TestSoftmaxFP32v2, Test_FP32_reduce_inner_dim)
{
    std::vector<ck::index_t> reduce_dims{1};
    this->Run(reduce_dims);
}

TYPED_TEST(TestSoftmaxFP32v2, Test_FP32_reduce_outermost_dim)
{
    std::vector<ck::index_t> reduce_dims{0};
    this->Run(reduce_dims);
}