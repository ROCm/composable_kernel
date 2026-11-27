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
     std::is_same_v<typename Problem::VDataType, half_t>) ||
    (std::is_same_v<typename Problem::QDataType, fp8_t> &&
     std::is_same_v<typename Problem::KDataType, fp8_t> &&
     std::is_same_v<typename Problem::VDataType, fp8_t> &&
     std::is_same_v<typename Problem::PDataType, fp8_t>);

// A K tile of 64 KB or more (MX d512: 128 keys of 512 bytes, its PV slice) leaves room for
// one K and one V buffer only: two of either would take the arena past two resident blocks.
template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_wide_k_tile_v =
    Problem::BlockFmhaShape::kN0 * Problem::BlockFmhaShape::kSubQKHeaddim *
        static_cast<index_t>(sizeof(typename Problem::KDataType)) >=
    64 * 1024;

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
    (Problem::BlockFmhaShape::kM0 >= 64 && Problem::BlockFmhaShape::kSubQKHeaddim >= 512 &&
     !is_qr_tdm_pingpong_wide_k_tile_v<Problem>);

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

// 16 B per Q and K row (row-strided ds_load_b128), 16 lanes' banks per V row for the
// transposed ds_load_tr: 32 B (16-bit tr16_b128) or 16 B (fp8 tr8_b64), as qr_tdm pads
// it; only a row-major V is padded.
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
    using V = std::conditional_t<
        Shape::kN1 == 160,
        LdsPaddingConfig<false, 0, 0>,
        LdsPaddingConfig<true, kInterval<kVRow>, 16 * sizeof(typename Problem::VDataType)>>;
};

// [K buffer 0][K buffer 1][S][V, aliased by Q]. Q is read once before the main loop and
// V's first write follows the loop-head barrier, so Q sits on V and K's first transfer can
// be issued right behind Q's. With a second V buffer the layout is
// [K0][K1][S][V0][V1, aliased by Q]: V0 is issued right behind K's first transfer, and V1's
// first write follows the first loop-head barrier. S only takes space with the gemm_0
// warps on N. d128 fp16/bf16, padded, warps on M: single V, M=64: K0=0, K1=17408,
// Q/V0=34816, end=53248; double V, M=64: V0=34816, Q/V1=53248, end=71680; double V,
// M=128: end=88064. The 16-row short-query tiles hold one K buffer (kSingleK):
// [K][S][V, aliased by Q]. Q may be larger than a V buffer (d128's 128-row Q: 34816 B against 18432
// B); it then runs past V1's end into the arena's tail, which is sized for it. Only where that tail
// would push the arena over budget (d512's 64-row Q, 64 KB) does Q start on V0 instead and span
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
    // The short-query tiles (decode) hold one K buffer: their QK gemm is short, so the next
    // K goes out once it has read the current one, and the buffer it saves leaves room for
    // another resident block (see the pipeline's kSingleK); so does a wide K tile.
    static constexpr bool kSingleK = Shape::kM0 <= 16 || is_qr_tdm_pingpong_wide_k_tile_v<Problem>;
    static constexpr index_t kK0Offset = 0;
    static constexpr index_t kK1Offset =
        kSingleK ? kK0Offset : integer_least_multiple(kK0Offset + kKBytes, kKAlignment);
    static constexpr index_t kSOffset = integer_least_multiple(kK1Offset + kKBytes, kSAlignment);
    static constexpr index_t kV0Offset =
        integer_least_multiple(kSOffset + kSBytes, kRegionAlignment);
    static constexpr index_t kRandValBytes =
        qr_tdm_randval_lds_bytes<Problem, BlockFmhaPipelineQRKSVSTdmDefaultPolicy>();
    // at least two resident blocks per WGP: the main loop relies on another block's
    // gemms to cover its softmax
    static constexpr index_t kArenaBudget = 160 * 1024;
    // V1 (with a second V buffer), Q, and the arena's end. Q starts on the last V buffer and
    // may run past its end. d512's Q (64 KB on the 64-row tile) would push the arena past its
    // budget from there, so it starts on V0 and spans both; V0 is then issued only after Q is
    // read (see the pipeline prologue). Dropout's randval shuffle (BlockDropout::Run) goes
    // past the Q/K/S/V regions.
    static constexpr index_t V1Offset(bool double_v)
    {
        return double_v ? integer_least_multiple(kV0Offset + kVBytes, kRegionAlignment) : kV0Offset;
    }
    static constexpr bool QOnBothV(bool double_v)
    {
        return double_v && Shape::kSubQKHeaddim >= 512;
    }
    static constexpr index_t ArenaBytes(bool double_v)
    {
        const index_t q_offset = QOnBothV(double_v) ? kV0Offset : V1Offset(double_v);
        return integer_least_multiple(
            max(q_offset + kQBytes, V1Offset(double_v) + kVBytes) + kRandValBytes, kArenaAlignment);
    }
    // a second V buffer where it pays off (is_qr_tdm_pingpong_double_v_v) and fits the
    // budget (bf16 d256 128-row: Q 64 KB, two K and two V of 32 KB would not)
    static constexpr bool kDoubleV =
        is_qr_tdm_pingpong_double_v_v<Problem> && ArenaBytes(true) <= kArenaBudget;
    static constexpr index_t kV1Offset      = V1Offset(kDoubleV);
    static constexpr bool kQOnBothV         = QOnBothV(kDoubleV);
    static constexpr index_t kQOffset       = kQOnBothV ? kV0Offset : kV1Offset;
    static constexpr index_t kRandValOffset = max(kQOffset + kQBytes, kV1Offset + kVBytes);
    static constexpr index_t kArenaBytes    = ArenaBytes(kDoubleV);

    static_assert(!KPadding::kEnabled || (kK1Offset - kK0Offset) % KPadding::kIntervalBytes == 0);
    static_assert(!kDoubleV || !VPadding::kEnabled ||
                  (kV1Offset - kV0Offset) % VPadding::kIntervalBytes == 0);
    static_assert(kArenaBytes <= kArenaBudget);
};

// The persistent option's arena (BlockFmhaPipelineQRKSVSTdmPingPongPersistentPolicy):
// qr_tdm_pingpong's (K0, K1, S, V0, V1) with Q in a region of its own instead of on V1, and a
// region for O behind it: the next item's Q is issued as soon as every wave holds the
// current one, and a finished item's O is copied out of its region under the next item.
template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmPingPongPersistentLdsArenaLayout
    : QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>
{
    using PingPong = QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;
    using Shape    = typename Problem::BlockFmhaShape;
    using ODataType = typename Problem::ODataType;

    static constexpr bool kQOnBothV   = false;
    static constexpr index_t kQOffset = integer_least_multiple(
        PingPong::kV1Offset + PingPong::kVBytes, PingPong::kRegionAlignment);
    // O: kM0 rows of kN1 values, a row's bytes padded to 32 past a multiple of 128 (the O
    // tile's 16-byte LDS writes then spread over the banks), as the GDPA q-loop stages it
    static constexpr index_t kOVec      = 16 / static_cast<index_t>(sizeof(ODataType));
    static constexpr index_t kORowBytes = Shape::kN1 * static_cast<index_t>(sizeof(ODataType));
    static constexpr index_t kORowStride =
        (kORowBytes + ((32 - kORowBytes) % 128 + 128) % 128) /
        static_cast<index_t>(sizeof(ODataType));
    static constexpr index_t kOBytes =
        Shape::kM0 * kORowStride * static_cast<index_t>(sizeof(ODataType));
    static constexpr index_t kOOffset =
        integer_least_multiple(kQOffset + PingPong::kQBytes, PingPong::kRegionAlignment);
    static constexpr index_t kArenaBytes = integer_least_multiple(kOOffset + kOBytes, 1024);
};

} // namespace detail

// qr_tdm_pingpong: the qr_tdm policy with whole-tile K staged in two buffers (one on the
// short-query tiles), one or two V buffers (is_qr_tdm_pingpong_double_v_v), and Q aliasing
// V: from the last V buffer on, or from V0 where kQOnBothV (see QrTdmPingPongLdsArenaLayout).
struct BlockFmhaPipelineQRKSVSTdmPingPongPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    // an fp8 P takes a fixed scale (kPExp2Offset), no scaled gemm1
    static constexpr bool kQuantizesPItself = true;

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

    // S comes back out of LDS (gemm_0 warps on N exchanging S) to be softmaxed and fed into
    // gemm_1 as its A operand, so it takes P's distribution.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        return MakePRegTileDistribution<Problem>();
    }

    // Row sum of P on the matrix unit: P (M x kN0) times ones (kN0 x one warp tile per
    // gemm_1 warp on N), with the PV gemm's types and warp gemm, so every warp holds whole row
    // sums, replicated across its columns, in o_acc's row layout.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetRowSumBlockGemm()
    {
        using PVBlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        using WarpGemm    = typename PVBlockGemm::WarpGemm;
        using Warps       = typename Problem::BlockFmhaShape::Gemm1BlockWarps;

        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    WarpGemm::kN * Warps::at(number<1>{}),
                                                    Problem::BlockFmhaShape::kK1>,
                                           Warps,
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

// A broadcast Q block's query tiles of a (batch, head) launched as one cluster (grid y), up to
// kQTileCluster: each K / V transfer goes to all of them at once (TDM workgroup_mask), so the keys
// are read once for the cluster. Without a mask (every tile walks the same key tiles).
struct BlockFmhaPipelineQRKSVSTdmPingPongClusterPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    static constexpr index_t kQTileCluster = 4;
};

// qr_tdm_pingpong, persistent: FmhaFwdKernel runs a workgroup's items (a batch's query tile of up
// to kM0 rows, one head) one after another through qr_tdm_pingpong's main loop
// (BlockFmhaPipelineQRKSVSTdmPingPong::RunItems), the transfers carried across the items: the
// next item's Q goes out once every wave holds the current one, the current item's last trip
// issues the next item's first K and V, and an item's O leaves through LDS, not waited for.
// With few keys an item (TCX's jagged attention: 128 queries over 1 .. 2000 keys) a workgroup per
// item spent most of its time starting (the kernel arguments, the first transfers) and finishing
// (waiting for its O store) rather than in its gemms. Unmasked softmax, group mode, an item's
// query rows one tile (max_seqlen_q <= kM0).
struct BlockFmhaPipelineQRKSVSTdmPingPongPersistentPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    static constexpr bool kPersistent = true;
    // rounds of workgroups a slot (FmhaFwdKernel::GridSize): whole rounds, and a few of them to
    // balance jagged batches
    // (gfx1250, TCX's jagged attention: one round to four within 2%; two the best on jagged
    // batches)
    static constexpr index_t kRounds = 2;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmPingPongPersistentLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;
};

} // namespace ck_tile
