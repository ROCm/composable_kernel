// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_amdgcn_mma_layout.inc"
TYPED_TEST_SUITE(TestMmaLayout, Gfx950Intrinsics);
TYPED_TEST(TestMmaLayout, Gfx950Intrinsics) { run_mma_layout_test<TypeParam>(); }
