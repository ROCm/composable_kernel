// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/grid_dim.hpp>
#include <rocm_ck/ops/fmha_bwd/common.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::FmhaBiasType;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::GridDim;

// ============================================================================
// FmhaMode
// ============================================================================

TEST(FmhaBwdCommon, FmhaMode_BatchAndGroupAreDistinct)
{
    EXPECT_NE(static_cast<int>(FmhaMode::BATCH), static_cast<int>(FmhaMode::GROUP));
}

// ============================================================================
// FmhaBiasType
// ============================================================================

TEST(FmhaBwdCommon, FmhaBiasType_ThreeDistinctVariants)
{
    EXPECT_NE(static_cast<int>(FmhaBiasType::NONE), static_cast<int>(FmhaBiasType::ELEMENTWISE));
    EXPECT_NE(static_cast<int>(FmhaBiasType::NONE), static_cast<int>(FmhaBiasType::ALIBI));
    EXPECT_NE(static_cast<int>(FmhaBiasType::ELEMENTWISE), static_cast<int>(FmhaBiasType::ALIBI));
}

// ============================================================================
// GridDim
// ============================================================================

TEST(FmhaBwdCommon, GridDim_DefaultsToOnes)
{
    constexpr GridDim g{};
    EXPECT_EQ(g.x, 1u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

TEST(FmhaBwdCommon, GridDim_StoresExplicitValues)
{
    constexpr GridDim g{4, 8, 2};
    EXPECT_EQ(g.x, 4u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}
