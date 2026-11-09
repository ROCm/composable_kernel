// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"

namespace ck_tile {

namespace detail {

template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_qkv_v =
    (std::is_same_v<typename Problem::QDataType, bf16_t> &&
     std::is_same_v<typename Problem::KDataType, bf16_t> &&
     std::is_same_v<typename Problem::VDataType, bf16_t>) ||
    (std::is_same_v<typename Problem::QDataType, half_t> &&
     std::is_same_v<typename Problem::KDataType, half_t> &&
     std::is_same_v<typename Problem::VDataType, half_t>);

// V gets a second LDS buffer (V_{i+1} is issued a whole trip ahead of its PV gemm) where
// measured on gfx1250 to pay off: the 128-row tiles, whose occupancy is VGPR bound so the
// extra LDS is free, the d64 masked 64-row tile, whose earlier V outweighs one fewer
// resident block, and the d512 64-row tile, whose 32-key trips are too short to cover a V
// transfer issued in the same trip (+1%). The d128/d256 64-row tiles and the 16-row
// short-query tiles keep one V buffer.
template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_double_v_v =
    Problem::BlockFmhaShape::kM0 >= 128 ||
    (Problem::BlockFmhaShape::kM0 >= 64 && Problem::BlockFmhaShape::kSubQKHeaddim <= 64 &&
     Problem::FmhaMask::IsMasking) ||
    (Problem::BlockFmhaShape::kM0 >= 64 && Problem::BlockFmhaShape::kSubQKHeaddim >= 512);

// A power-of-2 row is padded once, at its end: a row stride, with affine reader address
// math (no merge/unmerge div/mod). Other rows (96, 160, 192 columns) take qr_tdm's
// padding every power-of-2 interval, several per row.
template <typename DataType, index_t Rows, index_t Cols, typename PaddingConfig>
CK_TILE_HOST_DEVICE constexpr auto make_qr_tdm_pingpong_lds_descriptor()
{
    static_assert(numeric_traits<DataType>::PackedSize == 1);
    constexpr index_t kAccessElements = kQrTdmLdsAccessBytes / sizeof(DataType);
    constexpr index_t kPadElements    = PaddingConfig::kPadBytes / sizeof(DataType);

    if constexpr(PaddingConfig::kEnabled &&
                 PaddingConfig::kIntervalBytes != Cols * sizeof(DataType))
    {
        return make_qr_tdm_row_major_lds_descriptor<DataType,
                                                    Rows,
                                                    Cols,
                                                    PaddingConfig,
                                                    kQrTdmLdsAccessBytes>();
    }
    else
    {
        static_assert(!PaddingConfig::kEnabled || PaddingConfig::kPadBytes % sizeof(DataType) == 0);
        return make_naive_tensor_descriptor(make_tuple(number<Rows>{}, number<Cols>{}),
                                            make_tuple(number<Cols + kPadElements>{}, number<1>{}),
                                            number<kAccessElements>{},
                                            number<1>{});
    }
}

// Rows of whole 16-byte LDS accesses; power-of-2 rows are padded once (at their end), the
// others every power-of-2 interval, as qr_tdm pads them.
template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_padding_enabled_v =
    is_qr_tdm_padding_supported_arch_v && Problem::BlockFmhaShape::IsVLayoutRowMajor &&
    (Problem::BlockFmhaShape::kSubQKHeaddim * sizeof(typename Problem::KDataType)) %
            kQrTdmLdsAccessBytes ==
        0 &&
    (Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType)) % kQrTdmLdsAccessBytes ==
        0 &&
    Problem::BlockFmhaShape::kSubQKHeaddim * sizeof(typename Problem::KDataType) >= 16 &&
    Problem::BlockFmhaShape::kSubQKHeaddim * sizeof(typename Problem::KDataType) <= 1024 &&
    Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType) >= 16 &&
    Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType) <= 1024;

// 16 B per Q and K row (row-strided ds_load_b128), 32 B per V row (transposed ds_load_tr;
// only a row-major V is padded).
template <typename Problem, bool Enabled = is_qr_tdm_pingpong_padding_enabled_v<Problem>>
struct QrTdmPingPongPadding
{
    using Q = LdsPaddingConfig<false, 0, 0>;
    using K = LdsPaddingConfig<false, 0, 0>;
    using V = LdsPaddingConfig<false, 0, 0>;
};

template <typename Problem>
struct QrTdmPingPongPadding<Problem, true>
{
    using Shape = typename Problem::BlockFmhaShape;

    // one row, or (non-power-of-2 rows) qr_tdm's largest power-of-2 interval of the row
    template <index_t RowBytes>
    static constexpr index_t kInterval =
        is_power_of_two_integer(RowBytes) ? RowBytes : qr_tdm_interval_bytes<1, RowBytes>();
    static constexpr index_t kQRow = Shape::kSubQKHeaddim * sizeof(typename Problem::QDataType);
    static constexpr index_t kKRow = Shape::kSubQKHeaddim * sizeof(typename Problem::KDataType);
    static constexpr index_t kVRow = Shape::kN1 * sizeof(typename Problem::VDataType);

    using Q = LdsPaddingConfig<true, kInterval<kQRow>, kQrTdmLdsAccessBytes>;
    using K = LdsPaddingConfig<true, kInterval<kKRow>, kQrTdmLdsAccessBytes>;
    // As in qr_tdm, a 160-column V row skips padding: its 64-byte interval bloats LDS more
    // than it saves in bank conflicts.
    using V =
        std::conditional_t<Shape::kN1 == 160,
                           LdsPaddingConfig<false, 0, 0>,
                           LdsPaddingConfig<true, kInterval<kVRow>, 2 * kQrTdmLdsAccessBytes>>;
};

// [K buffer 0][K buffer 1][S][V, aliased by Q]. Q is read once before the main loop and
// V's first write follows the loop-head barrier, so Q sits on V and K's first transfer can
// be issued right behind Q's. With a second V buffer the layout is
// [K0][K1][S][V0][V1, aliased by Q]: V0 is issued right behind K's first transfer, and V1's
// first write follows the first loop-head barrier. S only takes space with the gemm_0
// warps on N. d128 fp16/bf16, padded, warps on M: single V, M=64: K0=0, K1=17408,
// Q/V0=34816, end=53248; double V, M=64: V0=34816, Q/V1=53248, end=71680; double V,
// M=128: end=88064. The 16-row short-query tile: S=34800, Q/V0=39168, end=57600. Q may be
// larger than a V buffer (d128's 128-row Q: 34816 B against 18432 B); it then runs past
// V1's end into the arena's tail, which is sized for it. Only where that tail would push
// the arena over budget (d512's 64-row Q, 64 KB) does Q start on V0 instead and span
// [V0][V1] (kQOnBothV).
template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmPingPongLdsArenaLayout
{
    using Shape = typename Problem::BlockFmhaShape;

    static constexpr index_t kArenaAlignment  = 256;
    static constexpr index_t kRegionAlignment = 256;

    static constexpr index_t kQBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::QDataType,
                                            Shape::kM0,
                                            Shape::kSubQKHeaddim,
                                            QPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::QDataType);
    static constexpr index_t kKBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                            Shape::kN0,
                                            Shape::kSubQKHeaddim,
                                            KPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::KDataType);
    static constexpr index_t kVBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::VDataType,
                                            Shape::kN0,
                                            Shape::kN1,
                                            VPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::VDataType);

    static constexpr index_t kSBytes =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemSizeS<Problem>();
    static constexpr index_t kSAlignment =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemNPackS<Problem>() *
        sizeof(typename Problem::SaccDataType);

    // K1 starts on a whole padding interval, so the TDM writer's padding phase is the
    // same in both K buffers (an interval is one K row: 1 KB at d512).
    static constexpr index_t kKAlignment =
        KPadding::kEnabled ? max(kRegionAlignment, KPadding::kIntervalBytes) : kRegionAlignment;
    static constexpr index_t kK0Offset = 0;
    static constexpr index_t kK1Offset = integer_least_multiple(kK0Offset + kKBytes, kKAlignment);
    static constexpr index_t kSOffset  = integer_least_multiple(kK1Offset + kKBytes, kSAlignment);
    static constexpr index_t kV0Offset =
        integer_least_multiple(kSOffset + kSBytes, kRegionAlignment);
    static constexpr bool kDoubleV = is_qr_tdm_pingpong_double_v_v<Problem>;
    static constexpr index_t kV1Offset =
        kDoubleV ? integer_least_multiple(kV0Offset + kVBytes, kRegionAlignment) : kV0Offset;
    // Q starts on the last V buffer and may run past its end. d512's Q (64 KB on the
    // 64-row tile) would push the arena past its budget from there, so it starts on V0
    // and spans both; V0 is then issued only after Q is read (see the pipeline prologue).
    static constexpr bool kQOnBothV   = kDoubleV && Shape::kSubQKHeaddim >= 512;
    static constexpr index_t kQOffset = kQOnBothV ? kV0Offset : kV1Offset;
    static constexpr index_t kArenaBytes =
        integer_least_multiple(max(kQOffset + kQBytes, kV1Offset + kVBytes), kArenaAlignment);

    static_assert(!KPadding::kEnabled || (kK1Offset - kK0Offset) % KPadding::kIntervalBytes == 0);
    static_assert(!kDoubleV || !VPadding::kEnabled ||
                  (kV1Offset - kV0Offset) % VPadding::kIntervalBytes == 0);
    // at least two resident blocks per WGP: the main loop relies on another block's
    // gemms to cover its softmax
    static_assert(kArenaBytes <= 160 * 1024);
};

} // namespace detail

// qr_tdm_pingpong: the qr_tdm policy with whole-tile K staged in two buffers, one or two
// V buffers (is_qr_tdm_pingpong_double_v_v), and Q aliasing V: from the last V buffer on,
// or from V0 where kQOnBothV (see QrTdmPingPongLdsArenaLayout).
struct BlockFmhaPipelineQRKSVSTdmPingPongPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{

    template <typename Problem>
    using LdsPaddingConfigQ = typename detail::QrTdmPingPongPadding<Problem>::Q;

    template <typename Problem>
    using LdsPaddingConfigK = typename detail::QrTdmPingPongPadding<Problem>::K;

    template <typename Problem>
    using LdsPaddingConfigV = typename detail::QrTdmPingPongPadding<Problem>::V;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::QDataType,
                                                           Problem::BlockFmhaShape::kM0,
                                                           Problem::BlockFmhaShape::kSubQKHeaddim,
                                                           LdsPaddingConfigQ<Problem>>();
    }

    // The whole K tile; the pipeline walks its kK0 slices with the read window.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                                           Problem::BlockFmhaShape::kN0,
                                                           Problem::BlockFmhaShape::kSubQKHeaddim,
                                                           LdsPaddingConfigK<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        // row-major V: [kN0][kN1]; column-major V: [kN1][kN0] (unpadded, so the arena's
        // kVBytes holds either way)
        if constexpr(Problem::BlockFmhaShape::IsVLayoutRowMajor)
            return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::VDataType,
                                                               Problem::BlockFmhaShape::kN0,
                                                               Problem::BlockFmhaShape::kN1,
                                                               LdsPaddingConfigV<Problem>>();
        else
            return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::VDataType,
                                                               Problem::BlockFmhaShape::kN1,
                                                               Problem::BlockFmhaShape::kN0,
                                                               LdsPaddingConfigV<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return LdsArenaLayout<Problem>::kArenaBytes;
    }

    // S comes back out of LDS (gemm_0 warps on N) to be softmaxed and fed into gemm_1
    // as its A operand, so it takes P's distribution. An encoding built from the QK
    // gemm's warp layout would permute P's columns within a row once the warps sit on
    // N: the row sums (and LSE) stay right while each key meets the wrong value row.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        return MakePRegTileDistribution<Problem>();
    }

    // Row sum of P on the matrix unit: P (M x kN0) times ones (kN0 x one warp tile), with
    // the PV gemm's types and warp gemm, so the sum lands replicated in o_acc's layout.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetRowSumBlockGemm()
    {
        using PVBlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        using WarpGemm    = typename PVBlockGemm::WarpGemm;

        static_assert(Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{}) == 1,
                      "one warp tile of columns covers the row sum only with the gemm_1 "
                      "warps all on M");

        using GemmProblem = BlockGemmProblem<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            Problem::kBlockSize,
            TileGemmShape<
                sequence<Problem::BlockFmhaShape::kM0, WarpGemm::kN, Problem::BlockFmhaShape::kK1>,
                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::KMN>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
