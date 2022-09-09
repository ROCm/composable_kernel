// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <stdexcept>

#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

template <typename Tuple>
class TestSoftmaxFP16Reduce1Dim : public ck::TestSoftmax<Tuple>
{
};

template <typename Tuple>
class TestSoftmaxFP16Reduce2Dims : public ck::TestSoftmax<Tuple>
{
};

template <typename Tuple>
class TestSoftmaxFP16ReduceAllDims : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypesReduce1Dim = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<ck::half_t, float, float, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<4>>, // mixed precision
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<8>, I<8>>
    >;
using KernelTypesReduce2Dims = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<8>, I<8>>
    >;
using KernelTypesReduceAllDims = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<3>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<3>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<3>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<8>, I<8>>
    >;
// clang-format on

TYPED_TEST_SUITE(TestSoftmaxFP16Reduce1Dim, KernelTypesReduce1Dim);
TYPED_TEST_SUITE(TestSoftmaxFP16Reduce2Dims, KernelTypesReduce2Dims);
TYPED_TEST_SUITE(TestSoftmaxFP16ReduceAllDims, KernelTypesReduceAllDims);

TYPED_TEST(TestSoftmaxFP16Reduce1Dim, ReduceInnermostDim)
{
    std::vector<ck::index_t> reduce_dims{2};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxFP16Reduce1Dim, ReduceInnerDims)
{
    std::vector<ck::index_t> reduce_dims{1};
    this->Run(reduce_dims);
    reduce_dims = std::vector<ck::index_t>{0};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxFP16Reduce1Dim, ReduceMoreDims)
{
    std::vector<ck::index_t> reduce_dims{1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{0, 1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxFP16Reduce2Dims, Reduce2Dims)
{
    std::vector<ck::index_t> reduce_dims{1, 2};
    this->Run(reduce_dims);
    reduce_dims = std::vector<ck::index_t>{0, 2};
    this->Run(reduce_dims);
    reduce_dims = std::vector<ck::index_t>{0, 1};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxFP16Reduce2Dims, ReduceTooLessDims)
{
    std::vector<ck::index_t> reduce_dims{2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{1};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxFP16Reduce2Dims, ReduceTooManyDims)
{
    std::vector<ck::index_t> reduce_dims{0, 1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxFP16ReduceAllDims, ReduceAllDims)
{
    std::vector<ck::index_t> reduce_dims{0, 1, 2};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxFP16ReduceAllDims, ReduceTooLessDims)
{
    std::vector<ck::index_t> reduce_dims{2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
