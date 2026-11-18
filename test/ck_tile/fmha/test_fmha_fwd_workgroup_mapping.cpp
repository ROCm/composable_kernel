// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/fmha/kernel/fmha_fwd_workgroup_mapping.hpp"
#include "gtest/gtest.h"

#include <array>
#include <numeric>
#include <vector>

namespace {

using namespace ck_tile;

// Exercise the complete mapper with the same input properties as the kernel.
// No policy enum or decoded workgroup ID is supplied by the caller.
template <bool Mask>
FmhaFwdTileIndex GridMapping(const FmhaFwdWorkgroupGrid& grid, const FmhaFwdWorkgroup& block)
{
    return MapFmhaFwdWorkgroup<Mask, false, false>({}, grid, block, 0);
}

template <bool Mask>
FmhaFwdTileIndex
HeadMajorMapping(const FmhaFwdWorkgroupGrid& grid, const FmhaFwdWorkgroup& block, index_t xcds)
{
    FmhaFwdWorkgroupFeatures features{};
    features.is_bhsd         = true;
    features.uniform_lengths = true;
    return MapFmhaFwdWorkgroup<Mask, true, false>(features, grid, block, xcds);
}

FmhaFwdTileIndex CausalMapping(const FmhaFwdWorkgroupGrid& grid, const FmhaFwdWorkgroup& block)
{
    FmhaFwdWorkgroupFeatures features{};
    features.is_bhsd       = true;
    features.square_causal = true;
    return MapFmhaFwdWorkgroup<true, true, true>(features, grid, block, 8);
}

FmhaFwdTileIndex
PairedMapping(const FmhaFwdWorkgroupGrid& grid, const FmhaFwdWorkgroup& block, index_t ratio)
{
    return MapFmhaFwdWorkgroup<true, false, false, true>({}, grid, block, 8, ratio);
}

FmhaFwdWorkgroup PhysicalBlock(index_t linear, const FmhaFwdWorkgroupGrid& g)
{
    const index_t head = linear % g.num_heads;
    const index_t rest = linear / g.num_heads;
    return g.has_padded_seqlen_k
               ? FmhaFwdWorkgroup{head, rest / g.num_batches, rest % g.num_batches}
               : FmhaFwdWorkgroup{head, rest % g.num_qv_tiles, rest / g.num_qv_tiles};
}

TEST(FmhaFwdWorkgroupMapping, DisabledHeadMajorAndCausalGatesRetainTheirFallbacks)
{
    const FmhaFwdWorkgroupGrid grid{2, 4, 1, 1, false};
    const FmhaFwdWorkgroup block{1, 0, 0};
    FmhaFwdWorkgroupFeatures features{};
    features.is_bhsd       = true;
    features.square_causal = true;
    auto tile              = MapFmhaFwdWorkgroup<true, true, true>(features, grid, block, 0);
    EXPECT_EQ(tile.q_tile, 3);
    EXPECT_EQ(tile.head, 1);
    tile = MapFmhaFwdWorkgroup<true, true, false>(features, grid, block, 0);
    EXPECT_EQ(tile.q_tile, 2);
    EXPECT_EQ(tile.head, 0);
    tile = MapFmhaFwdWorkgroup<true, false, false>(features, grid, block, 0);
    EXPECT_EQ(tile.q_tile, 3);
    EXPECT_EQ(tile.head, 1);
}

TEST(FmhaFwdWorkgroupMapping, LongTilesAcrossHeadsAndDenseHeadMajor)
{
    const FmhaFwdWorkgroupGrid g{2, 4, 1, 1, false};
    const std::array<std::array<int, 2>, 8> causal{
        {{3, 0}, {3, 1}, {2, 0}, {2, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1}}};
    const std::array<std::array<int, 2>, 8> dense{
        {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}}};
    for(int l = 0; l < 8; ++l)
    {
        const auto a = CausalMapping(g, PhysicalBlock(l, g));
        const auto b = HeadMajorMapping<false>(g, PhysicalBlock(l, g), 0);
        EXPECT_EQ(a.q_tile, causal[l][0]);
        EXPECT_EQ(a.head, causal[l][1]);
        EXPECT_EQ(b.q_tile, dense[l][0]);
        EXPECT_EQ(b.head, dense[l][1]);
    }
}

TEST(FmhaFwdWorkgroupMapping, XcdSegmentsUseTheMatchingDecoder)
{
    const FmhaFwdWorkgroupGrid g{2, 8, 1, 1, false};
    // Physical IDs 0,8 are XCD 0's jobs; they must map to adjacent Q tiles of head 0.
    const auto a = HeadMajorMapping<true>(g, PhysicalBlock(0, g), 8);
    const auto b = HeadMajorMapping<true>(g, PhysicalBlock(8, g), 8);
    EXPECT_EQ(a.q_tile, 7);
    EXPECT_EQ(b.q_tile, 6);
    EXPECT_EQ(a.head, 0);
    EXPECT_EQ(b.head, 0);
    const auto c = HeadMajorMapping<true>(g, PhysicalBlock(4, g), 8);
    EXPECT_EQ(c.head, 1);
    EXPECT_EQ(c.q_tile, 7);
}

TEST(FmhaFwdWorkgroupMapping, PaddedGridPreservesPhysicalLinearization)
{
    const FmhaFwdWorkgroupGrid g{2, 3, 2, 1, true};
    // Physical (x,y,z)=(0,1,0) is normalized to (head,tile,batch)=(0,0,1).
    const FmhaFwdWorkgroup block{0, 0, 1};
    EXPECT_EQ(FmhaFwdLinearWorkgroupId(g, block), 2);
    const auto direct = GridMapping<false>(g, block);
    EXPECT_EQ(direct.batch, 1);
    EXPECT_EQ(direct.q_tile, 0);
    const auto head_major = HeadMajorMapping<false>(g, block, 0);
    EXPECT_EQ(head_major.batch, 0);
    EXPECT_EQ(head_major.head, 0);
    EXPECT_EQ(head_major.q_tile, 2);
}

TEST(FmhaFwdWorkgroupMapping, MultipleVTilesKeepHeadMajorOrder)
{
    const FmhaFwdWorkgroupGrid g{2, 6, 1, 2, false};
    const auto h = HeadMajorMapping<true>(g, {0, 0, 0}, 0);
    EXPECT_EQ(h.q_tile, 2);
    EXPECT_EQ(h.v_tile, 0);
}

TEST(FmhaFwdWorkgroupMapping, MaskedGridReversesOnlyQTiles)
{
    // Three Q tiles and two V tiles: masking cannot turn Q tile 0 into tile 5.
    // Cover both the direct grid fallback and the XCD locality path.
    for(bool padded : {false, true})
    {
        const FmhaFwdWorkgroupGrid g{2, 6, 1, 2, padded};
        for(index_t tile = 0; tile < 6; ++tile)
        {
            const auto out = GridMapping<true>(g, {0, tile, 0});
            EXPECT_EQ(out.q_tile, 2 - tile / 2);
            EXPECT_EQ(out.v_tile, tile % 2);
        }
    }
    const FmhaFwdWorkgroupGrid g{2, 6, 1, 2, false};
    const auto out = HeadMajorMapping<true>(g, {0, 0, 0}, 8);
    EXPECT_EQ(out.q_tile, 2);
    EXPECT_EQ(out.v_tile, 0);
}

template <typename Mapping>
void CheckCoverage(const FmhaFwdWorkgroupGrid& g, Mapping map)
{
    std::vector<int> visits(g.num_heads * g.num_qv_tiles * g.num_batches);
    for(index_t l = 0; l < static_cast<index_t>(visits.size()); ++l)
    {
        const auto block = PhysicalBlock(l, g);
        ASSERT_EQ(FmhaFwdLinearWorkgroupId(g, block), l);
        const auto out = map(g, block);
        ASSERT_GE(out.q_tile, 0);
        ASSERT_LT(out.q_tile, (g.num_qv_tiles / g.num_v_tiles));
        ASSERT_GE(out.v_tile, 0);
        ASSERT_LT(out.v_tile, g.num_v_tiles);
        ASSERT_GE(out.head, 0);
        ASSERT_LT(out.head, g.num_heads);
        ASSERT_GE(out.batch, 0);
        ASSERT_LT(out.batch, g.num_batches);
        const int linear =
            ((out.batch * g.num_heads + out.head) * (g.num_qv_tiles / g.num_v_tiles) + out.q_tile) *
                g.num_v_tiles +
            out.v_tile;
        ASSERT_EQ(++visits[linear], 1);
    }
}

TEST(FmhaFwdWorkgroupMapping, BijectionsOnUnevenGridsAndMultipleVTiles)
{
    for(index_t batches : {1, 2, 3})
        for(index_t heads : {1, 2, 5, 8, 10})
            for(index_t qtiles : {1, 2, 3, 7, 16})
                for(index_t vtiles : {1, 2, 3})
                    for(bool padded : {false, true})
                    {
                        const FmhaFwdWorkgroupGrid g{
                            heads, qtiles * vtiles, batches, vtiles, padded};
                        CheckCoverage(g, GridMapping<false>);
                        CheckCoverage(g, GridMapping<true>);
                        if(vtiles == 1)
                            CheckCoverage(g, CausalMapping);
                        for(index_t xcds : {0, 1, 2, 3, 8})
                        {
                            CheckCoverage(g, [=](auto grid, auto block) {
                                return HeadMajorMapping<false>(grid, block, xcds);
                            });
                            CheckCoverage(g, [=](auto grid, auto block) {
                                return HeadMajorMapping<true>(grid, block, xcds);
                            });
                            // Also cover the general-grid XCD path and its masked
                            // threshold/padded-grid fallbacks through the full mapper.
                            FmhaFwdWorkgroupFeatures features{};
                            features.uniform_lengths = true;
                            CheckCoverage(g, [=](auto grid, auto block) {
                                return MapFmhaFwdWorkgroup<false, false, false>(
                                    features, grid, block, xcds);
                            });
                            CheckCoverage(g, [=](auto grid, auto block) {
                                return MapFmhaFwdWorkgroup<true, false, false>(
                                    features, grid, block, xcds);
                            });
                        }
                    }
}

TEST(FmhaFwdWorkgroupMapping, PairedOriginalOneFillOrder)
{
    const FmhaFwdWorkgroupGrid g{8, 64, 1, 1, false};
    for(index_t slot : {0, 31, 32, 63})
    {
        const auto tile = PairedMapping(g, {0, slot, 0}, 1);
        EXPECT_EQ(tile.q_tile, slot < 32 ? 63 - slot : slot - 32);
        EXPECT_EQ(tile.head, 0);
    }
    // The original GQA-dependent stream threshold switches local ordering.
    const FmhaFwdWorkgroupGrid streams{8, 16, 9, 1, false};
    const auto row_major = PairedMapping(streams, {0, 1, 0}, 1);
    const auto q_major   = PairedMapping(streams, {0, 1, 0}, 2);
    EXPECT_EQ(row_major.q_tile, 14);
    EXPECT_EQ(row_major.head, 0);
    EXPECT_EQ(q_major.q_tile, 15);
    EXPECT_EQ(q_major.head, 1);
    for(index_t batches : {1, 3, 9})
        for(index_t heads : {1, 2, 3, 8})
            for(index_t tiles : {1, 3, 16, 64, 256})
            {
                const FmhaFwdWorkgroupGrid test{heads, tiles, batches, 1, false};
                for(index_t ratio : {1, 2})
                    CheckCoverage(test, [=](auto grid, auto block) {
                        return PairedMapping(grid, block, ratio);
                    });
            }
}

FmhaFwdWorkgroupFeatures XcdCausalFeatures()
{
    FmhaFwdWorkgroupFeatures f{};
    f.is_bhsd         = true;
    f.uniform_lengths = f.square_causal = true;
    return f;
}

TEST(FmhaFwdWorkgroupMapping, XcdCausalKeepsWholeHeadsAndBoundsConcurrentGroups)
{
    const auto features = XcdCausalFeatures();
    for(int rows : {32, 64, 96, 256})
        for(int qtiles : {3, 4, 16})
        {
            SCOPED_TRACE(rows);
            SCOPED_TRACE(qtiles);
            const FmhaFwdWorkgroupGrid g{2, qtiles, rows / 2, 1, false};
            const int rows_per_xcd = rows / 8;
            const int group        = std::gcd(rows_per_xcd, 16);
            std::vector<FmhaFwdTileIndex> expected(rows * qtiles);
            // Independently enumerate groups using std::gcd, without the
            // production bit tests or inverse-coordinate arithmetic.
            for(int xcd = 0; xcd < 8; ++xcd)
            {
                int slot = 0;
                for(int first = xcd * rows_per_xcd; first < (xcd + 1) * rows_per_xcd;
                    first += group)
                    for(int q = qtiles - 1; q >= 0; --q)
                        for(int row = first; row < first + group; ++row, ++slot)
                            expected[xcd + 8 * slot] = {q, 0, row % 2, row / 2};
            }
            for(int l = 0; l < rows * qtiles; ++l)
            {
                const auto out = MapFmhaFwdWorkgroup<true, true, true, false, true>(
                    features, g, PhysicalBlock(l, g), 8, 1);
                EXPECT_EQ(out.q_tile, expected[l].q_tile);
                EXPECT_EQ(out.v_tile, 0);
                EXPECT_EQ(out.head, expected[l].head);
                EXPECT_EQ(out.batch, expected[l].batch);
            }
        }
}

TEST(FmhaFwdWorkgroupMapping, XcdCausalCoverageAndBalancedWork)
{
    const auto features = XcdCausalFeatures();
    for(int rows : {32, 64, 96, 128, 160, 256, 1024})
        for(int qtiles : {2, 3, 4, 7, 16})
        {
            const FmhaFwdWorkgroupGrid g{2, qtiles, rows / 2, 1, false};
            std::array<int, 8> costs{};
            CheckCoverage(g, [&](auto grid, auto block) {
                const auto out =
                    MapFmhaFwdWorkgroup<true, true, true, false, true>(features, grid, block, 8, 1);
                const int xcd = FmhaFwdLinearWorkgroupId(grid, block) % 8;
                costs[xcd] += out.q_tile + 1;
                // A row belongs to exactly one XCD across all Q depths.
                EXPECT_EQ((out.batch * 2 + out.head) / (rows / 8), xcd);
                return out;
            });
            for(int cost : costs)
                EXPECT_EQ(cost, rows / 8 * qtiles * (qtiles + 1) / 2);
        }
}

TEST(FmhaFwdWorkgroupMapping, XcdCausalEligibilityRetainsFallbacks)
{
    for(int variant = 0; variant < 12; ++variant)
    {
        auto f = XcdCausalFeatures();
        FmhaFwdWorkgroupGrid g{2, 4, 128, 1, false};
        int xcds = 8, ratio = 1;
        if(variant == 0)
            f.uniform_lengths = false;
        if(variant == 1)
            f.square_causal = false;
        if(variant == 2)
            g.has_padded_seqlen_k = true;
        if(variant == 3)
            g.num_v_tiles = 2;
        if(variant == 4)
            g.num_batches = 1;
        if(variant == 5)
            g.num_batches = 127;
        if(variant == 6)
            g.num_qv_tiles = 1;
        if(variant == 7)
            xcds = 0;
        if(variant == 8)
            xcds = 4;
        if(variant == 9)
            ratio = 2;
        if(variant == 10)
            g.num_batches = 12;
        if(variant == 11)
            g.num_qv_tiles = 17;
        for(int l = 0; l < g.num_heads * g.num_qv_tiles * g.num_batches; ++l)
        {
            const auto block = PhysicalBlock(l, g);
            const auto old   = MapFmhaFwdWorkgroup<true, true, true>(f, g, block, xcds, ratio);
            const auto out =
                MapFmhaFwdWorkgroup<true, true, true, false, true>(f, g, block, xcds, ratio);
            EXPECT_EQ(out.q_tile, old.q_tile);
            EXPECT_EQ(out.v_tile, old.v_tile);
            EXPECT_EQ(out.head, old.head);
            EXPECT_EQ(out.batch, old.batch);
            const auto dense_old = MapFmhaFwdWorkgroup<false, true, true>(f, g, block, xcds, ratio);
            const auto dense =
                MapFmhaFwdWorkgroup<false, true, true, false, true>(f, g, block, xcds, ratio);
            EXPECT_EQ(dense.q_tile, dense_old.q_tile);
            EXPECT_EQ(dense.v_tile, dense_old.v_tile);
            EXPECT_EQ(dense.head, dense_old.head);
            EXPECT_EQ(dense.batch, dense_old.batch);
        }
    }
}

TEST(FmhaFwdWorkgroupMapping, PairedOptInKeepsItsOwnFallbackWithXcdCausalEnabled)
{
    const auto features = XcdCausalFeatures();
    for(int qtiles : {3, 4, 7, 16})
    {
        const FmhaFwdWorkgroupGrid grid{2, qtiles, 128, 1, false};
        for(int l = 0; l < 256 * qtiles; ++l)
        {
            const auto block = PhysicalBlock(l, grid);
            const auto paired =
                MapFmhaFwdWorkgroup<true, true, true, true>(features, grid, block, 8, 1);
            const auto both =
                MapFmhaFwdWorkgroup<true, true, true, true, true>(features, grid, block, 8, 1);
            EXPECT_EQ(both.q_tile, paired.q_tile);
            EXPECT_EQ(both.v_tile, paired.v_tile);
            EXPECT_EQ(both.head, paired.head);
            EXPECT_EQ(both.batch, paired.batch);
        }
    }
}

} // namespace
