// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/load_tile_transpose.hpp"

namespace {

using namespace ck_tile;

template <typename Encoding>
using TileDistribution = decltype(make_static_tile_distribution(Encoding{}));

using MergedEncoding = tile_distribution_encoding<sequence<>,
                                                  tuple<sequence<1, 4>, sequence<32>>,
                                                  tuple<sequence<1>>,
                                                  tuple<sequence<0>>,
                                                  sequence<1, 2>,
                                                  sequence<1, 0>>;

using SplitEncoding = tile_distribution_encoding<sequence<>,
                                                 tuple<sequence<1, 4>, sequence<2, 16>>,
                                                 tuple<sequence<1>>,
                                                 tuple<sequence<0>>,
                                                 sequence<1, 2, 2>,
                                                 sequence<1, 0, 1>>;

using ReorderedSplitEncoding = tile_distribution_encoding<sequence<>,
                                                          tuple<sequence<1, 4>, sequence<2, 16>>,
                                                          tuple<sequence<1>>,
                                                          tuple<sequence<0>>,
                                                          sequence<2, 1, 2>,
                                                          sequence<0, 1, 1>>;

using SameLengthSplitEncoding = tile_distribution_encoding<sequence<>,
                                                           tuple<sequence<1, 2>, sequence<2, 16>>,
                                                           tuple<sequence<1>>,
                                                           tuple<sequence<0>>,
                                                           sequence<1, 2, 2>,
                                                           sequence<1, 0, 1>>;

using SameLengthReorderedSplitEncoding =
    tile_distribution_encoding<sequence<>,
                               tuple<sequence<1, 2>, sequence<2, 16>>,
                               tuple<sequence<1>>,
                               tuple<sequence<0>>,
                               sequence<2, 1, 2>,
                               sequence<0, 1, 1>>;

using ReorderedXEncoding = tile_distribution_encoding<sequence<>,
                                                      tuple<sequence<4, 1>, sequence<2, 16>>,
                                                      tuple<sequence<1>>,
                                                      tuple<sequence<1>>,
                                                      sequence<1, 2, 2>,
                                                      sequence<0, 0, 1>>;

using TwoStepSplitEncoding = tile_distribution_encoding<sequence<>,
                                                        tuple<sequence<1, 4>, sequence<2, 1, 16>>,
                                                        tuple<sequence<1>>,
                                                        tuple<sequence<0>>,
                                                        sequence<1, 2, 2, 2>,
                                                        sequence<1, 0, 1, 2>>;

using ShortKEncoding = tile_distribution_encoding<sequence<>,
                                                  tuple<sequence<1, 4>, sequence<16>>,
                                                  tuple<sequence<1>>,
                                                  tuple<sequence<0>>,
                                                  sequence<1, 2>,
                                                  sequence<1, 0>>;

using OddVectorMergedEncoding = tile_distribution_encoding<sequence<>,
                                                           tuple<sequence<1, 4>, sequence<30>>,
                                                           tuple<sequence<1>>,
                                                           tuple<sequence<0>>,
                                                           sequence<1, 2>,
                                                           sequence<1, 0>>;

using OddVectorSplitEncoding = tile_distribution_encoding<sequence<>,
                                                          tuple<sequence<1, 4>, sequence<2, 15>>,
                                                          tuple<sequence<1>>,
                                                          tuple<sequence<0>>,
                                                          sequence<1, 2, 2>,
                                                          sequence<1, 0, 1>>;

using NormalizedPackedEncoding =
    typename detail::NormalizeEncodingForTranspose<MergedEncoding, pk_fp4_t>::type;

static_assert(std::is_same_v<NormalizedPackedEncoding, SplitEncoding>);

static_assert(detail::is_transpose_output_compatible_v<TileDistribution<MergedEncoding>,
                                                       TileDistribution<MergedEncoding>,
                                                       half_t>);

static_assert(detail::is_transpose_output_compatible_v<TileDistribution<MergedEncoding>,
                                                       TileDistribution<MergedEncoding>,
                                                       pk_fp4_t>);

static_assert(detail::is_transpose_output_compatible_v<TileDistribution<SplitEncoding>,
                                                       TileDistribution<MergedEncoding>,
                                                       half_t>);

static_assert(detail::is_transpose_output_compatible_v<TileDistribution<SplitEncoding>,
                                                       TileDistribution<MergedEncoding>,
                                                       pk_fp4_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<ReorderedSplitEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        half_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<ReorderedSplitEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        pk_fp4_t>);

// Same Y lengths as SameLengthSplitEncoding, but y0/y1 map to different semantic RHS dims.
static_assert(
    !detail::is_transpose_output_compatible_v<TileDistribution<SameLengthReorderedSplitEncoding>,
                                              TileDistribution<SameLengthSplitEncoding>,
                                              half_t>);

static_assert(
    !detail::is_transpose_output_compatible_v<TileDistribution<SameLengthReorderedSplitEncoding>,
                                              TileDistribution<SameLengthSplitEncoding>,
                                              pk_fp4_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<ReorderedXEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        half_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<ReorderedXEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        pk_fp4_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<TwoStepSplitEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        half_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<TwoStepSplitEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        pk_fp4_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<ShortKEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        half_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<ShortKEncoding>,
                                                        TileDistribution<MergedEncoding>,
                                                        pk_fp4_t>);

static_assert(detail::is_transpose_output_compatible_v<TileDistribution<OddVectorSplitEncoding>,
                                                       TileDistribution<OddVectorMergedEncoding>,
                                                       half_t>);

static_assert(!detail::is_transpose_output_compatible_v<TileDistribution<OddVectorSplitEncoding>,
                                                        TileDistribution<OddVectorMergedEncoding>,
                                                        pk_fp4_t>);

} // namespace

TEST(LoadTileTransposeDistribution, StaticCompatibilityChecks) { SUCCEED(); }
