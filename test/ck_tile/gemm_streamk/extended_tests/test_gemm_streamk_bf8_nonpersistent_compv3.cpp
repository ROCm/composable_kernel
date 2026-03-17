// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKBf8NonPersistentCompV3 : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKBf8NonPersistentCompV3

TYPED_TEST_SUITE(TestCkTileStreamKBf8NonPersistentCompV3, KernelTypesStreamKBf8NonPersistentCompV3);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
