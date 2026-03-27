// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKBf16PersistentCompV4 : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKBf16PersistentCompV4

TYPED_TEST_SUITE(TestCkTileStreamKBf16PersistentCompV4, KernelTypesStreamKBf16PersistentCompV4);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
