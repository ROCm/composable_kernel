// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKFp8NonPersistentCompV4 : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKFp8NonPersistentCompV4

TYPED_TEST_SUITE(TestCkTileStreamKFp8NonPersistentCompV4, KernelTypesStreamKFp8NonPersistentCompV4);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
