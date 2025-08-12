// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_bwd.hpp"
#include "example/ck_tile/01_fmha/fmha_bwd_runner.hpp"

#include "gtest/gtest.h"

using KernelTypes = ::testing::Types<FmhaBwdFp16>;

TYPED_TEST_SUITE(TestCkTileFmhaBwd, KernelTypes);

#include "test_fmha_bwd.inc"
