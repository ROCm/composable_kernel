// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/amd_wave_read_first_lane.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/xcd_remap.hpp"

namespace ck_tile {

// Normalize the launch's Y/Z axes once. num_qv_tiles includes both Q and V tiles.
// Padded group launches use (head, batch, tile); other launches use (head, tile, batch).
struct FmhaFwdWorkgroupGrid
{
    index_t num_heads;
    index_t num_qv_tiles;
    index_t num_batches;
    index_t num_v_tiles;
    bool has_padded_seqlen_k;
};

struct FmhaFwdWorkgroup
{
    index_t head;
    index_t tile;
    index_t batch;
};

struct FmhaFwdTileIndex
{
    index_t q_tile;
    index_t v_tile;
    index_t head;
    index_t batch;
};

// The kernel supplies pipeline/architecture eligibility separately from runtime
// layout and sequence properties. No Kargs fields or launch ABI are added.
struct FmhaFwdWorkgroupFeatures
{
    bool is_bhsd         = false;
    bool uniform_lengths = false;
    bool square_causal   = false;
};

// Existing opt-in tm5r topology assumptions. These are not inferred from num_xcds
// and must be revalidated before enabling the experiment on another target.
struct FmhaFwdPairedXccConfig
{
    static constexpr index_t kCount             = 8;
    static constexpr index_t kWgpsPerXcc        = 32;
    static constexpr index_t kFullWgsPerXcc     = 512;
    static constexpr index_t kFullStreamsPerXcc = 8;
    static_assert((kCount & (kCount - 1)) == 0, "paired XCC count must be a power of two");
};

// Measured topology and working-set bounds for scheduled BF16 D128 causal MHA.
// A different CK_TILE_XCD_REMAP_NUM_XCDS override retains the existing fallback.
struct FmhaFwdXcdCausalConfig
{
    static constexpr index_t kCount        = 8;
    static constexpr index_t kMinGroupRows = 4;
    static constexpr index_t kMaxGroupRows = 16;
    static constexpr index_t kRowDivisor   = kCount * kMinGroupRows;
    static constexpr index_t kMaxQTiles    = 16;
    static constexpr index_t kMinGroupBits = integer_log2_floor(kMinGroupRows);
    static constexpr index_t kMaxGroupBits = integer_log2_floor(kMaxGroupRows);
    static_assert(is_power_of_two_integer(kCount) && is_power_of_two_integer(kMinGroupRows) &&
                  is_power_of_two_integer(kMaxGroupRows));
    static_assert(kMaxGroupRows == 4 * kMinGroupRows,
                  "group selection assumes three consecutive power-of-two sizes");
};

struct FmhaFwdPairedPartition
{
    index_t rows;
    index_t parts;
};

CK_TILE_HOST_DEVICE constexpr FmhaFwdPairedPartition
FmhaFwdGetPairedPartition(const FmhaFwdWorkgroupGrid& grid)
{
    const index_t rows = grid.num_batches * grid.num_heads;
    const index_t gcd  = ck_tile::min(FmhaFwdPairedXccConfig::kCount, rows & -rows);
    return {rows / gcd, FmhaFwdPairedXccConfig::kCount / gcd};
}

CK_TILE_HOST_DEVICE constexpr FmhaFwdTileIndex FmhaFwdBroadcastTileIndex(FmhaFwdTileIndex tile)
{
#if defined(__HIP_DEVICE_COMPILE__)
    tile.q_tile = amd_wave_read_first_lane(tile.q_tile);
    tile.head   = amd_wave_read_first_lane(tile.head);
    tile.batch  = amd_wave_read_first_lane(tile.batch);
#endif
    return tile;
}

CK_TILE_HOST_DEVICE constexpr index_t FmhaFwdLinearWorkgroupId(const FmhaFwdWorkgroupGrid& grid,
                                                               const FmhaFwdWorkgroup& block)
{
    return grid.has_padded_seqlen_k
               ? block.head + grid.num_heads * (block.batch + grid.num_batches * block.tile)
               : block.head + grid.num_heads * (block.tile + grid.num_qv_tiles * block.batch);
}

// Pure coordinates for the existing paired-XCC experiment. The device caller
// still broadcasts the results to SGPRs, as the original implementation did.
CK_TILE_HOST_DEVICE constexpr FmhaFwdTileIndex MapFmhaFwdPairedWorkgroup(
    const FmhaFwdWorkgroupGrid& grid, const FmhaFwdWorkgroup& block, index_t nhead_ratio_qk)
{
    using Config            = FmhaFwdPairedXccConfig;
    const index_t linear_id = FmhaFwdLinearWorkgroupId(grid, block);
    const index_t num_row   = grid.num_batches * grid.num_heads;
    const auto partition    = FmhaFwdGetPairedPartition(grid);
    const index_t parts     = partition.parts;
    const index_t xcc       = linear_id % Config::kCount;
    const index_t slot      = linear_id / Config::kCount;
    const index_t rows      = partition.rows;
    const index_t pairs     = grid.num_qv_tiles / (2 * parts);
    const index_t deep      = rows * pairs;
    const index_t tiles     = 2 * deep;
    const index_t first     = xcc * rows;
    const index_t stripe    = first / num_row;
    const index_t row0      = first - stripe * num_row;

    index_t row_off = 0;
    index_t k       = 0;
    bool shallow    = false;
    if(tiles <= Config::kFullWgsPerXcc && rows <= Config::kFullStreamsPerXcc * nhead_ratio_qk)
    {
        // Queue slots s and s + kWgpsPerXcc share a WGP in the original one-fill
        // experiment. Keep its deep/shallow pairing and stream threshold intact.
        index_t rank = slot;
        if(tiles <= 2 * Config::kWgpsPerXcc && slot >= Config::kWgpsPerXcc)
            rank = tiles - 1 - (slot - Config::kWgpsPerXcc);
        shallow            = rank >= deep;
        const index_t r    = shallow ? rank - deep : rank;
        const index_t step = r / rows;
        row_off            = r - step * rows;
        k                  = shallow ? pairs - 1 - step : step;
    }
    else
    {
        // Bound the concurrent K/V set by visiting one row at a time.
        row_off         = slot / (2 * pairs);
        const index_t r = slot - row_off * 2 * pairs;
        shallow         = r >= pairs;
        k               = shallow ? 2 * pairs - 1 - r : r;
    }
    const index_t q     = stripe + k * parts;
    const index_t row   = row0 + row_off;
    const index_t batch = row / grid.num_heads;
    return {shallow ? q : grid.num_qv_tiles - 1 - q, 0, row - batch * grid.num_heads, batch};
}

// Pure coordinates for XCD-owned rows, with heavy Q tiles first in bounded groups.
// Requires a valid grid/block admitted by the XcdCausal selector: rows divisible
// by kRowDivisor, one V tile, no padded K lengths and the configured XCD count.
CK_TILE_HOST_DEVICE constexpr FmhaFwdTileIndex
MapFmhaFwdXcdCausalWorkgroup(const FmhaFwdWorkgroupGrid& grid, const FmhaFwdWorkgroup& block)
{
    using Config                = FmhaFwdXcdCausalConfig;
    const uint32_t rows         = grid.num_batches * grid.num_heads;
    const uint32_t rows_per_xcd = rows / Config::kCount;
    // gcd(rows_per_xcd, kMaxGroupRows) is one of the three configured group sizes.
    // Explicit shifts avoid a runtime division by the group size.
    const uint32_t group_bits =
        rows % (Config::kCount * Config::kMaxGroupRows) == 0       ? Config::kMaxGroupBits
        : rows % (Config::kCount * Config::kMaxGroupRows / 2) == 0 ? Config::kMaxGroupBits - 1
                                                                   : Config::kMinGroupBits;
    const uint32_t group     = 1u << group_bits;
    const uint32_t linear    = FmhaFwdLinearWorkgroupId(grid, block);
    const uint32_t slot      = linear / Config::kCount;
    const uint32_t tile_slot = slot >> group_bits;
    const uint32_t chunk     = tile_slot / static_cast<uint32_t>(grid.num_qv_tiles);
    const uint32_t q         = grid.num_qv_tiles - 1 - (tile_slot - chunk * grid.num_qv_tiles);
    const uint32_t row =
        (linear % Config::kCount) * rows_per_xcd + chunk * group + (slot & (group - 1));
    const uint32_t batch = row / static_cast<uint32_t>(grid.num_heads);
    return {static_cast<index_t>(q),
            0,
            static_cast<index_t>(row - batch * grid.num_heads),
            static_cast<index_t>(batch)};
}

// Map a normalized physical workgroup directly to its logical tile. Pipeline
// eligibility is compile-time; only layout, sequence and topology facts are runtime.
// Paired opt-in and its fallback take precedence over the measured XCD grouping.
template <bool HasMask,
          bool EnableHeadMajor,
          bool EnableBatchCausal,
          bool EnablePaired    = false,
          bool EnableXcdCausal = false>
CK_TILE_HOST_DEVICE constexpr FmhaFwdTileIndex
MapFmhaFwdWorkgroup(const FmhaFwdWorkgroupFeatures& features,
                    const FmhaFwdWorkgroupGrid& grid,
                    const FmhaFwdWorkgroup& block,
                    index_t num_xcds,
                    index_t nhead_ratio_qk = 1)
{
    // These are internal code paths, not independently tunable policies.
    enum class Path
    {
        Grid,
        HeadMajor,
        BatchCausal,
        BhsdXcd,
        GridXcd,
        PairedCausal,
        XcdCausal,
    };
    const auto path = [&]() {
        // This source structure is codegen-sensitive with HIP 7.16 on gfx1250:
        // checking XCD eligibility first added ~0.2 us to small causal kernels.
        // Retest structural/compiler changes; this does not enforce runtime order.
        const auto fallback = [&]() {
            if constexpr(HasMask && EnablePaired)
            {
                if(grid.num_v_tiles == 1)
                {
                    const auto partition = FmhaFwdGetPairedPartition(grid);
                    return grid.num_qv_tiles % (2 * partition.parts) == 0 ? Path::PairedCausal
                                                                          : Path::BatchCausal;
                }
            }

            if constexpr(EnableHeadMajor)
            {
                if(features.is_bhsd)
                {
                    // Preserve the batch-local causal override ahead of generic XCD remap.
                    if constexpr(HasMask && EnableBatchCausal)
                    {
                        if(features.square_causal && grid.num_v_tiles == 1)
                            return Path::BatchCausal;
                    }
                    return num_xcds > 0 && features.uniform_lengths ? Path::BhsdXcd
                                                                    : Path::HeadMajor;
                }
            }

            // Whole-batch XCD segments can be unbalanced with variable effective lengths.
            if(num_xcds > 0 && !grid.has_padded_seqlen_k && features.uniform_lengths &&
               grid.num_qv_tiles > 1 &&
               (!HasMask || grid.num_heads * grid.num_batches >= num_xcds * grid.num_qv_tiles))
                return Path::GridXcd;
            return Path::Grid;
        }();
        if constexpr(HasMask && EnableXcdCausal && !EnablePaired)
        {
            using Config = FmhaFwdXcdCausalConfig;
            // Whole rows retain K/V locality; bounded groups avoid the measured
            // regressions from traversing every owned head at each Q depth.
            // M128 keeps Q <= 2048: larger working sets retain the existing order.
            if(num_xcds == Config::kCount && features.uniform_lengths && features.square_causal &&
               !grid.has_padded_seqlen_k && grid.num_v_tiles == 1 && nhead_ratio_qk == 1 &&
               (grid.num_batches * grid.num_heads) % Config::kRowDivisor == 0 &&
               grid.num_qv_tiles > 1 && grid.num_qv_tiles <= Config::kMaxQTiles)
                return Path::XcdCausal;
        }

        return fallback;
    }();
    if constexpr(HasMask && EnablePaired)
    {
        if(path == Path::PairedCausal)
            return FmhaFwdBroadcastTileIndex(
                MapFmhaFwdPairedWorkgroup(grid, block, nhead_ratio_qk));
    }
    else if constexpr(HasMask && EnableXcdCausal)
    {
        if(path == Path::XcdCausal)
            return FmhaFwdBroadcastTileIndex(MapFmhaFwdXcdCausalWorkgroup(grid, block));
    }
    if(path == Path::BatchCausal)
        return {grid.num_qv_tiles - 1 - block.tile, 0, block.head, block.batch};

    index_t tile  = block.tile;
    index_t head  = block.head;
    index_t batch = block.batch;
    if(path == Path::HeadMajor || path == Path::BhsdXcd)
    {
        index_t linear_id = FmhaFwdLinearWorkgroupId(grid, block);
        if(path == Path::BhsdXcd)
            linear_id = remap_xcd(
                linear_id, grid.num_heads * grid.num_qv_tiles * grid.num_batches, num_xcds);
        const index_t blocks_per_batch = grid.num_heads * grid.num_qv_tiles;
        batch                          = linear_id / blocks_per_batch;
        const index_t rem              = linear_id - batch * blocks_per_batch;
        head                           = rem / grid.num_qv_tiles;
        tile                           = rem - head * grid.num_qv_tiles;
    }
    else if(path == Path::GridXcd)
    {
        // Same logical order as BhsdXcd, but decode the packed tile first.
        // Retain both arithmetic forms: merging them measured slower in the
        // early gfx1250 standalone experiment, despite identical coordinates.
        const index_t linear_id = remap_xcd(FmhaFwdLinearWorkgroupId(grid, block),
                                            grid.num_heads * grid.num_qv_tiles * grid.num_batches,
                                            num_xcds);
        const index_t rest      = linear_id / grid.num_qv_tiles;
        tile                    = linear_id - rest * grid.num_qv_tiles;
        batch                   = rest / grid.num_heads;
        head                    = rest - batch * grid.num_heads;
    }

    index_t q       = tile / grid.num_v_tiles;
    const index_t v = tile - q * grid.num_v_tiles;
    if constexpr(HasMask)
    {
        const index_t num_q_tiles = grid.num_qv_tiles / grid.num_v_tiles;
        q                         = num_q_tiles - 1 - q;
    }
    return {q, v, head, batch};
}

} // namespace ck_tile
