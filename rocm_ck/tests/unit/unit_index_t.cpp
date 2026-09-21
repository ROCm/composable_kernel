// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/index_t.hpp>

#include <ck_tile/core/numeric/integer.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::index_t;
using ::rocm_ck::long_index_t;

namespace {

TEST(IndexTypes, IndexTypeIs32Bit) { EXPECT_EQ(sizeof(index_t), 4); }

TEST(IndexTypes, LongIndexTypeIs64Bit) { EXPECT_EQ(sizeof(long_index_t), 8); }

TEST(IndexTypes, IndexTypeIsSigned) { EXPECT_TRUE(index_t(-1) < 0); }

TEST(IndexTypes, LongIndexTypeIsSigned) { EXPECT_TRUE(long_index_t(-1) < 0); }

TEST(IndexTypes, MatchesCkTileIndexType)
{
    EXPECT_TRUE((std::is_same_v<index_t, ck_tile::index_t>));
}

TEST(IndexTypes, MatchesCkTileLongIndexType)
{
    EXPECT_TRUE((std::is_same_v<long_index_t, ck_tile::long_index_t>));
}

} // namespace
