// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Tests that TDM loads leave zeros in LDS for the part of a tile past the bound of its
 *        global tensor view, for the tiles the MX GEMM TDM pipelines load.
 * @details tile_window_with_static_distribution::tdm_load_to_lds clamps the TDM tensor dimensions
 *          to the remaining extent of the view. The MX GEMM TDM pipelines rely on this for K
 *          tails: the elements of a partial last K tile past K are zero, so no masking is needed.
 *          This test covers the loads; the MX GEMM tests cover the pipelines that issue them.
 *
 *          Each case loads one tile of A or B, 64 rows of 128 or 384 values of K (a 128-deep or
 *          384-deep tile; K is the contiguous dimension), the way the pipelines do: 4 waves each
 *          load a 16-row box, the part of the tile one TDM load copies
 *          (GemmPipelineAgBgCrCompTDMV1), or one wave loads the whole tile as one box
 *          (GemmPipelineAgBgCrCompTDMV2). The LDS layout and padding are those of the TDM default
 *          policy: 4 pad dwords after every 256 bytes for 128-deep tiles, no padding for 384-deep
 *          tiles. Odd K values give rows that start at 1- or 2-byte-aligned addresses.
 *
 *          Each case runs twice, with two sentinel bytes. The LDS arena is filled with the
 *          sentinel first; every arena byte is then compared with the expected image: global
 *          memory inside the bound, zero past it, and the sentinel in the pad bytes and in the
 *          guard bytes after the tile, so a load that writes outside its box fails. Global memory
 *          past the bound holds non-zero bytes, which a load_tile_tdm_full_tile load of the same
 *          tile (TDM tensor dimensions set to the box) reads.
 */

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace {

using ck_tile::index_t;
using F8 = ck_tile::fp8_t;
using F4 = ck_tile::pk_fp4_t;

constexpr index_t kRows  = 64;
constexpr index_t kWarps = 4;
// LDS padding of the TDM default policy for 128-deep tiles (GetLdsPaddingConfig; keep in sync):
// 16 pad bytes after every 256 bytes.
constexpr index_t kPadIntervalBytes = 256;
constexpr index_t kPadBytes         = 16;
// The TDM skips pad_amount + 1 dwords after every 2^(pad_interval + 1) dwords it writes, counted
// from the first element of each box.
constexpr uint32_t kPadInterval = 5;
constexpr uint32_t kPadAmount   = kPadBytes / 4 - 1;
// The TDM descriptor must pad at the interval the LDS layout below assumes.
static_assert((4 << (kPadInterval + 1)) == kPadIntervalBytes);
// Sentinel bytes after the tile in the LDS arena.
constexpr index_t kGuardBytes = 256;
// Extra global memory after the furthest byte a tile can reach.
constexpr index_t kGlobalSlackBytes = 256;

// Values per storage element; fp8_t and pk_fp4_t are one byte each, so this is values per byte.
template <typename DataType>
constexpr index_t kPackedSize = ck_tile::numeric_traits<DataType>::PackedSize;

// The LDS layout of a tile of kRows rows of Cols values (K is the contiguous dimension).
template <typename DataType, index_t Cols>
struct TileLayout
{
    // The tile depths whose policy padding this layout encodes: the policy pads 128-deep tiles
    // and does not pad 384-deep ones.
    static_assert(Cols == 128 || Cols == 384);

    static constexpr index_t kRowBytes = Cols / kPackedSize<DataType>;
    static constexpr bool kPadded      = Cols == 128;
    // Rows per pad group, and the LDS bytes of a group including its pad.
    static constexpr index_t kGroupRows  = kPadded ? kPadIntervalBytes / kRowBytes : kRows;
    static constexpr index_t kGroupBytes = kGroupRows * kRowBytes + (kPadded ? kPadBytes : 0);
    static constexpr index_t kTileBytes  = kRows / kGroupRows * kGroupBytes;
    static constexpr index_t kArenaBytes = kTileBytes + kGuardBytes;
    // A pad group holds whole rows, and every 16-row box starts on a group boundary, where the
    // TDM starts counting its pad interval.
    static_assert(!kPadded || kPadIntervalBytes % kRowBytes == 0);
    static_assert((kRows / kWarps) % kGroupRows == 0 || kGroupRows == kRows);

    // Host-side byte offset of (row, byte): the same layout as ClipLoad's LDS descriptor.
    static constexpr std::size_t lds_offset(index_t row, index_t byte)
    {
        return static_cast<std::size_t>(row / kGroupRows) * kGroupBytes +
               static_cast<std::size_t>(row % kGroupRows) * kRowBytes + byte;
    }

    // Where an arena byte lies: a tile row and byte, a pad, or the guard.
    static std::string describe(std::size_t offset)
    {
        if(offset >= static_cast<std::size_t>(kTileBytes))
            return "guard byte " + std::to_string(offset - kTileBytes);
        const auto group  = static_cast<index_t>(offset / kGroupBytes);
        const auto within = static_cast<index_t>(offset % kGroupBytes);
        if(within >= kGroupRows * kRowBytes)
            return "pad after row " + std::to_string((group + 1) * kGroupRows - 1);
        return "row " + std::to_string(group * kGroupRows + within / kRowBytes) + ", byte " +
               std::to_string(within % kRowBytes);
    }
};

// Loads the tile at (origin_row, origin_col) of a bound_rows x bound_cols global view with row
// stride `stride` (all in values) into LDS and copies the whole LDS arena to `output`. OneBox
// loads the tile as one box from one wave; otherwise each of the kWarps waves loads 16 rows.
// FullTile loads with load_tile_tdm_full_tile.
template <typename DataType, index_t Cols, bool OneBox, bool FullTile>
struct ClipLoad
{
    using Layout = TileLayout<DataType, Cols>;

    // kentry's __launch_bounds__ reads Kernel::kBlockSize: kWarps 32-lane waves.
    static constexpr index_t kBlockSize = kWarps * 32;

    __device__ void operator()(const DataType* global,
                               index_t bound_rows,
                               index_t bound_cols,
                               index_t stride,
                               index_t origin_row,
                               index_t origin_col,
                               index_t sentinel,
                               unsigned char* output) const
    {
        using namespace ck_tile;
        alignas(256) __shared__ unsigned char arena[Layout::kArenaBytes];
        const index_t thread = get_thread_local_1d_id();
        for(index_t offset = thread; offset < Layout::kArenaBytes; offset += kBlockSize)
            arena[offset] = static_cast<unsigned char>(sentinel);
        block_sync_lds();

        // Row groups of kGroupRows rows, each followed by its pad, as MakeA/BLdsBlockDescriptor of
        // the TDM default policy lay them out (keep in sync); a 384-deep tile is one group without
        // a pad. Strides are in values; (group, row in group) merge into one row index.
        constexpr index_t pack          = kPackedSize<DataType>;
        constexpr index_t groups        = kRows / Layout::kGroupRows;
        constexpr auto lds_desc_grouped = make_naive_tensor_descriptor(
            make_tuple(number<groups>{}, number<Layout::kGroupRows>{}, number<Cols>{}),
            make_tuple(number<Layout::kGroupBytes * pack>{}, number<Cols>{}, number<1>{}));
        constexpr auto lds_descriptor = transform_tensor_descriptor(
            lds_desc_grouped,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<groups>{}, number<Layout::kGroupRows>{})),
                       make_pass_through_transform(number<Cols>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        auto lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(arena), lds_descriptor);
        auto lds_window =
            make_tile_window(lds_view, make_tuple(number<kRows>{}, number<Cols>{}), {0, 0});

        auto global_view =
            make_naive_tensor_view<address_space_enum::global>(global,
                                                               make_tuple(bound_rows, bound_cols),
                                                               make_tuple(stride, 1),
                                                               number<1>{},
                                                               number<1>{});
        // The rows split among the loading waves, as MakeA/BDramTileDistribution of the TDM
        // default policy split them (keep in sync).
        constexpr index_t loading_waves = OneBox ? 1 : kWarps;
        auto distribution               = make_static_tile_distribution(
            tile_distribution_encoding<
                              sequence<>,
                              tuple<sequence<loading_waves, kRows / loading_waves>, sequence<Cols>>,
                              tuple<sequence<1>>,
                              tuple<sequence<0>>,
                              sequence<1, 2>,
                              sequence<1, 0>>{},
            bool_constant<true>{});
        // The pipelines reach later K tiles by moving the window, so the test does too.
        auto global_window = make_tile_window(global_view,
                                              make_tuple(number<kRows>{}, number<Cols>{}),
                                              {origin_row, 0},
                                              distribution);
        move_tile_window(global_window, {0, origin_col});

        TDMConfig config;
        config.pad_enable              = Layout::kPadded;
        config.pad_config.pad_interval = Layout::kPadded ? kPadInterval : 0;
        config.pad_config.pad_amount   = Layout::kPadded ? kPadAmount : 0;
        if(!OneBox || get_warp_id() == 0)
        {
            if constexpr(FullTile)
                load_tile_tdm_full_tile(config, lds_window, global_window);
            else
                load_tile_tdm(config, lds_window, global_window);
        }
        // Waits for this wave's TDM loads, then for the whole workgroup.
        s_wait_tensorcnt_barrier<0>();

        for(index_t offset = thread; offset < Layout::kArenaBytes; offset += kBlockSize)
            output[offset] = arena[offset];
    }
};

// One load: the global view and the tile origin, in values.
struct ClipCase
{
    const char* name;
    index_t bound_rows;
    index_t bound_cols;
    index_t stride;
    index_t origin_row;
    index_t origin_col;
};

// Each run fills the LDS arena with one of these first. A pad or guard byte must change with the
// sentinel, so stale data in a reused output buffer cannot pass.
constexpr unsigned char kSentinels[] = {0x5a, 0xa5};

// Non-zero bytes that differ from both sentinels and vary with the address, so a byte copied to
// the wrong place is caught.
unsigned char global_byte(std::size_t index)
{
    const auto value       = static_cast<unsigned char>(1 + (index * 97 + 13) % 251);
    const bool is_sentinel = value == kSentinels[0] || value == kSentinels[1];
    return is_sentinel ? static_cast<unsigned char>(value ^ 0x01) : value;
}

template <typename DataType, index_t Cols, bool OneBox, bool FullTile>
::testing::AssertionResult run_case(const ClipCase& c)
{
    using Kernel                = ClipLoad<DataType, Cols, OneBox, FullTile>;
    using Layout                = typename Kernel::Layout;
    constexpr index_t pack      = kPackedSize<DataType>;
    constexpr index_t row_bytes = Layout::kRowBytes;
    const std::string label     = std::string(c.name) + " (" +
                              (std::is_same_v<DataType, F4> ? "fp4" : "fp8") + ", tile " +
                              std::to_string(kRows) + " x " + std::to_string(Cols) +
                              (OneBox ? ", one " + std::to_string(kRows) + "-row box"
                                      : ", " + std::to_string(kWarps) + " boxes of " +
                                            std::to_string(kRows / kWarps) + " rows") +
                              (FullTile ? ", load_tile_tdm_full_tile)" : ", load_tile_tdm)");

    // tdm_load_to_lds divides value offsets and extents by PackedSize, so no case may split a
    // byte between inside and past the bound.
    if(c.stride % pack != 0 || c.bound_cols % pack != 0 || c.origin_col % pack != 0 ||
       c.stride < c.bound_cols || c.origin_row < 0 || c.origin_col < 0)
        return ::testing::AssertionFailure()
               << label << ": the stride, bound and origin must be whole storage elements, with "
               << "the stride at least the bound";
    // ClipLoad gives each of its kWarps 32-lane waves 16 rows.
    if(!ck_tile::is_wave32())
        return ::testing::AssertionFailure() << label << ": needs wave32";

    const index_t stride_bytes = c.stride / pack;
    const index_t bound_bytes  = c.bound_cols / pack;
    const index_t origin_byte  = c.origin_col / pack;
    // The buffer covers the whole tile, so the bytes past the bound are allocated, non-zero
    // memory (the full-tile control reads them).
    const std::size_t global_bytes = static_cast<std::size_t>(c.origin_row + kRows) * stride_bytes +
                                     origin_byte + row_bytes + kGlobalSlackBytes;
    std::vector<unsigned char> global(global_bytes);
    for(std::size_t i = 0; i < global.size(); ++i)
        global[i] = global_byte(i);

    // Expected arena, apart from the sentinel; `written` marks the bytes the load writes and
    // `past` the bytes past the bound.
    std::vector<unsigned char> expected(Layout::kArenaBytes, 0);
    std::vector<bool> written(Layout::kArenaBytes, false);
    std::vector<bool> past(Layout::kArenaBytes, false);
    index_t bytes_past_bound = 0;
    for(index_t row = 0; row < kRows; ++row)
    {
        for(index_t byte = 0; byte < row_bytes; ++byte)
        {
            const index_t global_row      = c.origin_row + row;
            const index_t global_col_byte = origin_byte + byte;
            const bool inside         = global_row < c.bound_rows && global_col_byte < bound_bytes;
            const std::size_t lds     = Layout::lds_offset(row, byte);
            const std::size_t address = static_cast<std::size_t>(global_row) * stride_bytes +
                                        static_cast<std::size_t>(global_col_byte);
            expected[lds] = (inside || FullTile) ? global[address] : 0;
            written[lds]  = true;
            past[lds]     = !inside;
            bytes_past_bound += inside ? 0 : 1;
        }
    }
    if(bytes_past_bound == 0)
        return ::testing::AssertionFailure() << label << ": the tile does not cross the bound";

    ck_tile::DeviceMem global_device(global.size());
    ck_tile::DeviceMem output_device(Layout::kArenaBytes);
    global_device.ToDevice(global.data());
    const ck_tile::stream_config stream{nullptr, false, 0, 0, 1};
    std::vector<unsigned char> output(Layout::kArenaBytes);
    for(const index_t sentinel : kSentinels)
    {
        ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel(Kernel{},
                                 dim3(1),
                                 dim3(Kernel::kBlockSize),
                                 0,
                                 static_cast<const DataType*>(global_device.GetDeviceBuffer()),
                                 c.bound_rows,
                                 c.bound_cols,
                                 c.stride,
                                 c.origin_row,
                                 c.origin_col,
                                 sentinel,
                                 static_cast<unsigned char*>(output_device.GetDeviceBuffer())));
        output_device.FromDevice(output.data());
        const auto want = [&](std::size_t offset) {
            return written[offset] ? expected[offset] : static_cast<unsigned char>(sentinel);
        };
        std::size_t mismatches = 0;
        std::size_t first      = 0;
        for(std::size_t offset = 0; offset < output.size(); ++offset)
        {
            if(output[offset] == want(offset))
                continue;
            if(mismatches == 0)
                first = offset;
            ++mismatches;
        }
        if(mismatches != 0)
        {
            const char* region = !written[first] ? "LDS pad or guard"
                                 : past[first]   ? "past the bound"
                                                 : "inside the bound";
            // gtest's AssertionResult drops stream manipulators, so format the sentinel here.
            std::ostringstream sentinel_hex;
            sentinel_hex << "0x" << std::hex << sentinel;
            return ::testing::AssertionFailure()
                   << label << ", sentinel " << sentinel_hex.str() << ": " << mismatches << " of "
                   << output.size() << " LDS bytes differ; first: " << Layout::describe(first)
                   << " (" << region << "), expected " << static_cast<unsigned int>(want(first))
                   << ", got " << static_cast<unsigned int>(output[first]);
        }
    }
    return ::testing::AssertionSuccess();
}

// Runs a case in both ways the TDM pipelines load a tile.
template <typename DataType, index_t Cols, bool FullTile = false>
void check(const ClipCase& c)
{
    EXPECT_TRUE((run_case<DataType, Cols, false, FullTile>(c)));
    EXPECT_TRUE((run_case<DataType, Cols, true, FullTile>(c)));
}

// Cases shared by a clamped load and its full-tile control.
constexpr ClipCase kFp8KTail{"K = 240, K tile 1", 64, 240, 240, 0, 128};
constexpr ClipCase kFp4KTail{"K = 152, K tile 1", 64, 152, 152, 0, 128};

TEST(TdmClipZeroFill, Fp8)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "TDM requires a gfx125 device";
    // The partial last K tile: 112 of its 128 values lie inside K = 240.
    check<F8, 128>(kFp8KTail);
    // K below the K tile: the first tile is partial, and the second tile, which the pipelines load
    // in their prologue but do not read, lies entirely past K.
    check<F8, 128>({"K = 24, K tile 0", 64, 24, 24, 0, 0});
    check<F8, 128>({"K = 24, K tile 1", 64, 24, 24, 0, 128});
    // An odd K: rows of 161 bytes start at 1-byte-aligned addresses.
    check<F8, 128>({"K = 161, K tile 1", 64, 161, 161, 0, 128});
    // The last row tile with a K tail: the tile covers rows 64-127 of a 112-row view, so rows
    // 112-127 lie past the bound.
    check<F8, 128>({"112 rows, K = 240, row tile 1, K tile 1", 112, 240, 240, 64, 128});
    // M = 1: only rows are clipped; three of the four boxes lie entirely past the bound.
    check<F8, 128>({"1 row, K = 512", 1, 512, 512, 0, 0});
    // A leading dimension longer than K: the bound is K, not the stride.
    check<F8, 128>({"K = 240, stride 256, K tile 1", 64, 240, 256, 0, 128});
    // A 384-deep K tile (no LDS padding): the last K tile of K = 900.
    check<F8, 384>({"K = 900, K tile 2", 64, 900, 900, 0, 768});
}

TEST(TdmClipZeroFill, Fp4)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "TDM requires a gfx125 device";
    check<F4, 128>(kFp4KTail);
    // K below the K tile with rows of 46 bytes (2-byte-aligned row starts), and the second tile,
    // entirely past K.
    check<F4, 128>({"K = 92, K tile 0", 64, 92, 92, 0, 0});
    check<F4, 128>({"K = 92, K tile 1", 64, 92, 92, 0, 128});
    // Rows of 45 bytes (1-byte-aligned row starts).
    check<F4, 128>({"K = 90, K tile 0", 64, 90, 90, 0, 0});
    check<F4, 128>({"112 rows, K = 152, row tile 1, K tile 1", 112, 152, 152, 64, 128});
    check<F4, 384>({"K = 900, K tile 2", 64, 900, 900, 0, 768});
}

// The K-tail tiles above, loaded with the TDM tensor dimensions set to the box, read the non-zero
// global bytes past the bound: the zeros in the other tests come from the clamped TDM tensor
// dimensions.
TEST(TdmClipZeroFill, FullTileReadsPastTheBound)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "TDM requires a gfx125 device";
    check<F8, 128, true>(kFp8KTail);
    check<F4, 128, true>(kFp4KTail);
}

} // namespace
