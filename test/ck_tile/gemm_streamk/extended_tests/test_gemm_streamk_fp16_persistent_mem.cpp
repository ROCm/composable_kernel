// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKFp16PersistentMem : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKFp16PersistentMem

TYPED_TEST_SUITE(TestCkTileStreamKFp16PersistentMem, KernelTypesStreamKFp16PersistentMem);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
