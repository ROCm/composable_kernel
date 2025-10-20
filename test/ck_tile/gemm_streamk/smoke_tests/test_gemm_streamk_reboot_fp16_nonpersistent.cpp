// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_gemm_streamk_reboot_types.hpp"
#include "test_gemm_streamk_reboot_util.hpp"
#include "gtest/gtest.h"

template <typename Tuple>
class TestCkTileStreamKRebootFp16NonPersistent : public TestCkTileStreamKReboot<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKRebootFp16NonPersistent

TYPED_TEST_SUITE(TestCkTileStreamKRebootFp16NonPersistent, KernelTypesStreamKFp16NonPersistent);

#include "test_gemm_streamk_reboot_smoke_cases.inc"

#undef TEST_SUITE_NAME
