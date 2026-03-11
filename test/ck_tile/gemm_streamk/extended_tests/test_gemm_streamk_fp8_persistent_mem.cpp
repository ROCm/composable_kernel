// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKFp8PersistentMem : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKFp8PersistentMem

TYPED_TEST_SUITE(TestCkTileStreamKFp8PersistentMem, KernelTypesStreamKFp8PersistentMem);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
