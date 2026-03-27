// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKFp16NonPersistentCompV4 : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKFp16NonPersistentCompV4

TYPED_TEST_SUITE(TestCkTileStreamKFp16NonPersistentCompV4,
                 KernelTypesStreamKFp16NonPersistentCompV4);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
