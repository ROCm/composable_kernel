// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_common.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

namespace {

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<std::tuple<Xs...>, std::tuple<Ys...>>
{
    using type = std::tuple<Xs..., Ys...>;
};

} // namespace

template <typename Tuple>
class TestGemmBiasAddReduce_FP16_MK_NK
    : public ck::test::TestGemmBiasAddReduceCommon<
          typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_NK = ::testing::Types<
    std::tuple< F16, F16, F16, F16, F16, F32>
   >;
// clang-format on

TYPED_TEST_SUITE(TestGemmBiasAddReduce_FP16_MK_NK, KernelTypes_MK_NK);

TYPED_TEST(TestGemmBiasAddReduce_FP16_MK_NK, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 1024;

    for(int M : Ms)
        this->Run(M, N, K);
}
