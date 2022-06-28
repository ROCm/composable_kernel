// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

template <typename Tuple>
class TestSoftmaxFP16 : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<ck::half_t, float, float, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<4>>, // mixed precision
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<8>, I<8>>
    >;
// clang-format on
TYPED_TEST_SUITE(TestSoftmaxFP16, KernelTypes);
TYPED_TEST(TestSoftmaxFP16, Test_FP16) { this->Run(); }
