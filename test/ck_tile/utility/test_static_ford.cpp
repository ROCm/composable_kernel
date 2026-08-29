// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/utility/functional.hpp"

using namespace ck_tile;

// ============================================================================
// static_ford Tests - Identity Order (default)
// ============================================================================

TEST(CkTileStaticFord, Identity2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    static_ford<sequence<2, 3>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(0, 1));
    EXPECT_EQ(visited[2], std::make_pair(0, 2));
    EXPECT_EQ(visited[3], std::make_pair(1, 0));
    EXPECT_EQ(visited[4], std::make_pair(1, 1));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(CkTileStaticFord, Identity3D)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    static_ford<sequence<2, 3, 2>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 12u);
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[2], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[3], std::make_tuple(0, 1, 1));
    EXPECT_EQ(visited[4], std::make_tuple(0, 2, 0));
    EXPECT_EQ(visited[5], std::make_tuple(0, 2, 1));
    EXPECT_EQ(visited[6], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[7], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[8], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[9], std::make_tuple(1, 1, 1));
    EXPECT_EQ(visited[10], std::make_tuple(1, 2, 0));
    EXPECT_EQ(visited[11], std::make_tuple(1, 2, 1));
}

TEST(CkTileStaticFord, Identity1D)
{
    std::vector<index_t> visited;

    static_ford<sequence<5>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        visited.push_back(i);
    });

    ASSERT_EQ(visited.size(), 5u);
    for(index_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(visited[i], i);
    }
}

TEST(CkTileStaticFord, SingleElement1D)
{
    std::vector<index_t> visited;

    static_ford<sequence<1>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        visited.push_back(i);
    });

    ASSERT_EQ(visited.size(), 1u);
    EXPECT_EQ(visited[0], 0);
}

TEST(CkTileStaticFord, SingleElement2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    static_ford<sequence<1, 1>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 1u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
}

TEST(CkTileStaticFord, IdentityWithUnitDim)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    static_ford<sequence<2, 1, 3>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[2], std::make_tuple(0, 0, 2));
    EXPECT_EQ(visited[3], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[4], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[5], std::make_tuple(1, 0, 2));
}

// ============================================================================
// static_ford Tests - Non-Identity Order (primary template with decompose_reordered)
// ============================================================================

TEST(CkTileStaticFord, ReversedOrder2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    // Order (1, 0): dim 1 is outer, dim 0 is inner (column-major)
    static_ford<sequence<2, 3>, sequence<1, 0>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(1, 0));
    EXPECT_EQ(visited[2], std::make_pair(0, 1));
    EXPECT_EQ(visited[3], std::make_pair(1, 1));
    EXPECT_EQ(visited[4], std::make_pair(0, 2));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(CkTileStaticFord, CustomOrder3D_201)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    // Orders<2,0,1>: dim 2 outermost, dim 0 middle, dim 1 innermost
    static_ford<sequence<2, 3, 4>, sequence<2, 0, 1>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 24u);
    // With orders (2,0,1): k varies slowest, then i, then j fastest
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[2], std::make_tuple(0, 2, 0));
    EXPECT_EQ(visited[3], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[4], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[5], std::make_tuple(1, 2, 0));
    EXPECT_EQ(visited[6], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[7], std::make_tuple(0, 1, 1));
    // Tail: last element should be (1, 2, 3)
    EXPECT_EQ(visited[23], std::make_tuple(1, 2, 3));
}

TEST(CkTileStaticFord, CustomOrder3D_120)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    // Orders<1,2,0>: dim 1 outermost, dim 2 middle, dim 0 innermost
    static_ford<sequence<2, 3, 2>, sequence<1, 2, 0>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 12u);
    // With orders (1,2,0): j varies slowest, then k, then i fastest
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[2], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[3], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[4], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[5], std::make_tuple(1, 1, 0));
    // Tail: last element should be (1, 2, 1)
    EXPECT_EQ(visited[11], std::make_tuple(1, 2, 1));
}

TEST(CkTileStaticFord, NonIdentityWithUnitDim)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    // Unit dim at position 1 with non-trivial order
    static_ford<sequence<2, 1, 3>, sequence<2, 0, 1>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 6u);
    // All entries must have j == 0 (unit dimension)
    for(size_t idx = 0; idx < visited.size(); ++idx)
    {
        EXPECT_EQ(std::get<1>(visited[idx]), 0) << "Unit dim not zero at iteration " << idx;
    }
}

TEST(CkTileStaticFord, CustomOrder4D)
{
    std::vector<std::tuple<index_t, index_t, index_t, index_t>> visited;

    // 4D with order <3,1,0,2>
    static_ford<sequence<2, 3, 2, 4>, sequence<3, 1, 0, 2>>{}([&](auto multi_id) {
        constexpr index_t a = multi_id[number<0>{}];
        constexpr index_t b = multi_id[number<1>{}];
        constexpr index_t c = multi_id[number<2>{}];
        constexpr index_t d = multi_id[number<3>{}];
        visited.emplace_back(a, b, c, d);
    });

    ASSERT_EQ(visited.size(), 48u);
    // dim 3 (size 4) outermost, dim 1 (size 3) next, dim 0 (size 2) next, dim 2 (size 2) inner
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 1, 0));
    EXPECT_EQ(visited[2], std::make_tuple(1, 0, 0, 0));
    EXPECT_EQ(visited[3], std::make_tuple(1, 0, 1, 0));
    EXPECT_EQ(visited[4], std::make_tuple(0, 1, 0, 0));
    EXPECT_EQ(visited[5], std::make_tuple(0, 1, 1, 0));
}

TEST(CkTileStaticFord, AsymmetricDimsWithOrder)
{
    std::vector<std::pair<index_t, index_t>> visited;

    // Asymmetric: 3x5 with reversed order
    static_ford<sequence<3, 5>, sequence<1, 0>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 15u);
    // dim 1 (size 5) outer, dim 0 (size 3) inner
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(1, 0));
    EXPECT_EQ(visited[2], std::make_pair(2, 0));
    EXPECT_EQ(visited[3], std::make_pair(0, 1));
    EXPECT_EQ(visited[4], std::make_pair(1, 1));
    EXPECT_EQ(visited[5], std::make_pair(2, 1));
}

// ============================================================================
// Consistency: identity order matches explicit identity order
// ============================================================================

TEST(CkTileStaticFord, IdentityOrderMatchesExplicit)
{
    std::vector<std::pair<index_t, index_t>> default_visited;
    std::vector<std::pair<index_t, index_t>> explicit_visited;

    static_ford<sequence<3, 4>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        default_visited.emplace_back(i, j);
    });

    static_ford<sequence<3, 4>, sequence<0, 1>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        explicit_visited.emplace_back(i, j);
    });

    ASSERT_EQ(default_visited.size(), explicit_visited.size());
    for(size_t i = 0; i < default_visited.size(); ++i)
    {
        EXPECT_EQ(default_visited[i], explicit_visited[i]) << "Mismatch at iteration " << i;
    }
}

// index_decomposer and inverse_perm are implementation details tested
// indirectly through the static_ford behavioral tests above.
// The IdentityOrderMatchesExplicit test verifies both code paths
// (identity specialization and primary template) produce identical results.
