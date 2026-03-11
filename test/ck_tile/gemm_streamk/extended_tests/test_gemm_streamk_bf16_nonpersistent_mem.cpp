// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKBf16NonPersistentMem : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKBf16NonPersistentMem

TYPED_TEST_SUITE(TestCkTileStreamKBf16NonPersistentMem, KernelTypesStreamKBf16NonPersistentMem);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
