// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include "ck_tile/core/utility/xcd_remap.hpp"

using namespace ck_tile;

namespace {

// The branching form GemmSpatiallyLocalTilePartitioner::RemapXCD writes out.
index_t remap_xcd_reference(index_t id, index_t total, index_t num_xcds)
{
    const index_t ids_per_xcd = (total + num_xcds - 1) / num_xcds;
    const index_t tall_xcds   = (total % num_xcds == 0) ? num_xcds : total % num_xcds;
    const index_t xcd         = id % num_xcds;
    const index_t local_id    = id / num_xcds;
    return xcd < tall_xcds
               ? xcd * ids_per_xcd + local_id
               : tall_xcds * ids_per_xcd + (xcd - tall_xcds) * (ids_per_xcd - 1) + local_id;
}

} // namespace

static_assert(remap_xcd(8, 16, 8) == 1 && remap_xcd(1, 16, 8) == 2 && remap_xcd(15, 16, 8) == 15);
static_assert(remap_xcd_inverse(1, 16, 8) == 8 && remap_xcd_inverse(2, 16, 8) == 1);

TEST(CkTileXcdRemap, MatchesReferenceAndInverts)
{
    for(index_t num_xcds : {1, 2, 3, 4, 8, 16})
    {
        for(index_t total = 1; total <= 200; ++total)
        {
            std::vector<bool> hit(total, false);
            for(index_t id = 0; id < total; ++id)
            {
                const index_t r = remap_xcd(id, total, num_xcds);
                ASSERT_EQ(r, remap_xcd_reference(id, total, num_xcds))
                    << "id " << id << " total " << total << " xcds " << num_xcds;
                ASSERT_GE(r, 0);
                ASSERT_LT(r, total);
                ASSERT_FALSE(hit[r]) << "not a bijection: total " << total << " xcds " << num_xcds;
                hit[r] = true;
                ASSERT_EQ(remap_xcd_inverse(r, total, num_xcds), id)
                    << "id " << id << " total " << total << " xcds " << num_xcds;
            }
        }
    }
}

// Each XCD's ids, taken in dispatch order, cover one contiguous range.
TEST(CkTileXcdRemap, PerXcdSegmentsAreContiguous)
{
    for(index_t total : {7, 8, 9, 100, 256, 1001})
    {
        constexpr index_t num_xcds = 8;
        index_t next               = 0;
        for(index_t xcd = 0; xcd < num_xcds; ++xcd)
            for(index_t id = xcd; id < total; id += num_xcds)
                ASSERT_EQ(remap_xcd(id, total, num_xcds), next++) << "total " << total;
        EXPECT_EQ(next, total);
    }
}
