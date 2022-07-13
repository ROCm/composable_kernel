// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_layernorm_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

template <typename Tuple>
class TestLayernormFP32 : public ck::TestLayernorm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
//  XDataType, GammaDataType, BetaDataType, AccDataType, YDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, , GammaSrcVectorSize, BetaSrcVectorSize, YDstVectorSize>
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>,
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<8>, I<32>, I<2>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>,
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>,
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<4>, I<64>, I<2>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>,
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>,
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<2>, I<128>, I<2>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>,
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>,
    std::tuple<float, float, float, float, float, I<2>, I<1>, I<256>, I<1>, I<256>, I<2>, I<8>, I<1>, I<4>, I<4>, I<4>, I<4>>
    >;
// clang-format on
TYPED_TEST_SUITE(TestLayernormFP32, KernelTypes);
TYPED_TEST(TestLayernormFP32, Test_FP32) { this->Run(); }
