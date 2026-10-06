// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Large-tensor BQuantGrouped + PreshuffleB coverage: exercises the flat weight-preshuffle B
// view whose element count (N * K) exceeds 2^31. GemmConfigPreshuffleBLargeTensor sets
// LargeTensors=true so the kernel takes the 64-bit global load/store path for the flat B view.
//
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType,
// CDataType, QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BPreshuffleBLargeTensorTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigPreshuffleBLargeTensor, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8, float, Half, BQuantGrouped, GemmConfigPreshuffleBLargeTensor, GroupSize1D_128>
>;
// clang-format on

template <typename Tuple>
class TestCkTileGemmPreshuffleBBQuantLargeTensor : public TestCkTileGemmPreshuffleBBQuant<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGemmPreshuffleBBQuantLargeTensor, BPreshuffleBLargeTensorTypes);

// Compile-time-flag regression check: 1024^3 is below the runtime large-tensor gate (2^31), so
// the executed branch is the normal preshuffle-B path. The purpose is to prove that enabling the
// compile-time LargeTensors opt-in does not regress numerical correctness of the common path
// (validated against the host reference via the shared BQuant validation driver).
TYPED_TEST(TestCkTileGemmPreshuffleBBQuantLargeTensor, LargeTensorFlagValidated)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// M=128, N=8192, K=262272 -> B has N*K = 2,148,532,224 elements ~= 2.001 GiB, exceeding the
// 2^31 32-bit-offset limit by ~2^20, so the runtime large-tensor gate engages and the flat B
// view is addressed in 64-bit. A host GEMM reference is intractable at this K, so correctness is
// spot-checked: B is filled directly in flat storage order (near region < 2^31 = 1, far region
// >= 2^31 = 7) and A is nonzero only at hot_k = K-1, so C(m, n) = (m % 8) * (n >= 8176 ? 7 : 1).
// The far spot-checks (n >= 8176) exercise flat offsets >= 2^31; the near spot-checks are
// controls. Validated for both FP8 and BF8.
TYPED_TEST(TestCkTileGemmPreshuffleBBQuantLargeTensor, BoundaryCheck)
{
    // clang-format off
    this->run_test_boundary_check_bquant(
        128, 8192, 262272, /*hot_k=*/262271,
        /*spot_checks=*/{// far region (n >= 8176): expected (m%8)*7 -- exercises flat offset >= 2^31
                         {1, 8191},
                         {7, 8191},
                         {3, 8176},
                         {100, 8188},
                         {127, 8180},
                         // near region (n < 8176): expected (m%8)*1 -- controls
                         {1, 0},
                         {127, 0},
                         {3, 4096},
                         {50, 8000},
                         {5, 8175}});
    // clang-format on
}
