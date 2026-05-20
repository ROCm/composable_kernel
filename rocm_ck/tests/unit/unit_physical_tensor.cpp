// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/physical_tensor.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::DataType;
using ::rocm_ck::FixedString;
using ::rocm_ck::kMaxPhysicalTensors;
using ::rocm_ck::Layout;
using ::rocm_ck::PhysicalTensor;

namespace {

TEST(PhysicalTensor, InitializesWithFP32RowAndSlotZero)
{
    constexpr PhysicalTensor pt{};
    EXPECT_EQ(pt.dtype, DataType::FP32);
    EXPECT_EQ(pt.layout, Layout::Row);
    EXPECT_EQ(pt.args_slot, 0);
}

TEST(PhysicalTensor, StoresAllFieldsFromConstruction)
{
    constexpr PhysicalTensor pt{FixedString<16>("bias"), DataType::FP16, Layout::Col, 3};
    EXPECT_TRUE(pt.name == "bias");
    EXPECT_EQ(pt.dtype, DataType::FP16);
    EXPECT_EQ(pt.layout, Layout::Col);
    EXPECT_EQ(pt.args_slot, 3);
}

TEST(PhysicalTensor, LimitsCapacityTo8) { EXPECT_EQ(kMaxPhysicalTensors, 8); }

} // namespace
