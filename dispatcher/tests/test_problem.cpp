// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for Problem using Google Test

#include "ck_tile/dispatcher/problem.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;

TEST(ProblemTest, DefaultConstruction)
{
    Problem p;
    EXPECT_EQ(p.M, 0);
    EXPECT_EQ(p.N, 0);
    EXPECT_EQ(p.K, 0);
    EXPECT_EQ(p.k_batch, 1);
    EXPECT_FALSE(p.is_valid());
}

TEST(ProblemTest, ConstructorWithDimensions)
{
    Problem p(1024, 1024, 1024);
    EXPECT_EQ(p.M, 1024);
    EXPECT_EQ(p.N, 1024);
    EXPECT_EQ(p.K, 1024);
    EXPECT_TRUE(p.is_valid());
}

TEST(ProblemTest, Validation)
{
    Problem p;

    // Invalid: all zeros
    p.M = 0;
    p.N = 0;
    p.K = 0;
    EXPECT_FALSE(p.is_valid());

    // Invalid: negative
    p.M = -1;
    p.N = 1024;
    p.K = 1024;
    EXPECT_FALSE(p.is_valid());

    // Invalid: zero K
    p.M = 1024;
    p.N = 1024;
    p.K = 0;
    EXPECT_FALSE(p.is_valid());

    // Valid
    p.M = 1024;
    p.N = 1024;
    p.K = 1024;
    EXPECT_TRUE(p.is_valid());

    // Invalid k_batch
    p.k_batch = 0;
    EXPECT_FALSE(p.is_valid());

    p.k_batch = 1;
    EXPECT_TRUE(p.is_valid());
}

TEST(ProblemTest, NumOps)
{
    Problem p(100, 200, 300);

    // 2 * M * N * K (multiply-add = 2 ops)
    std::int64_t expected = 2 * 100 * 200 * 300;
    EXPECT_EQ(p.num_ops(), expected);
}

TEST(ProblemTest, Configuration)
{
    Problem p(1024, 1024, 1024);

    // Set preferences
    p.prefer_persistent = true;
    p.enable_validation = true;
    p.smem_budget       = 65536;
    p.k_batch           = 2;

    EXPECT_TRUE(p.prefer_persistent);
    EXPECT_TRUE(p.enable_validation);
    EXPECT_EQ(p.smem_budget, 65536);
    EXPECT_EQ(p.k_batch, 2);
}

TEST(ProblemTest, LargeDimensions)
{
    Problem p(1024, 1024, 1024); // Use smaller but still large dimensions
    EXPECT_TRUE(p.is_valid());
    EXPECT_GT(p.num_ops(), 0);
}
