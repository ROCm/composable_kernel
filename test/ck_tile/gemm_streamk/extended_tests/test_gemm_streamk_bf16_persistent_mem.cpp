// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKBf16PersistentMem : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKBf16PersistentMem

TYPED_TEST_SUITE(TestCkTileStreamKBf16PersistentMem, KernelTypesStreamKBf16PersistentMem);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
