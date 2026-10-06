// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_util.hpp"

// FP64 16x16x4 MFMA warp gemm test case
// A=fp64, B=fp64, Acc=fp64, TransposeC=false, SwizzleA=false, USS=false, NumAccess=Single
using F64M16N16K4Case = ck_tile::test::warp_gemm::WGDispCase<ck_tile::fp64_t,
                                                             ck_tile::fp64_t,
                                                             ck_tile::fp64_t,
                                                             false,
                                                             false,
                                                             false,
                                                             ck_tile::WGAttrNumAccessEnum::Single>;

TEST(WarpGemmF64, MFMA_16x16x4)
{
    // RunCompareDispatcherAndReference<Case, M, N, K, UseScale, IsScale16>
    // UseScale = false because FP64 MFMA is a plain GEMM (no MX scaling)
    ck_tile::test::warp_gemm::RunCompareDispatcherAndReference<F64M16N16K4Case, 16, 16, 4, false>();
}
