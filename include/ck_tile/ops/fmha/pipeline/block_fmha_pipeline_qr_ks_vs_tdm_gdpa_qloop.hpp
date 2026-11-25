// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong_policy.hpp"

namespace ck_tile {

namespace detail {

// qr_tdm_pingpong's arena (K0, K1, S, V0, V1) with Q in a region of its own, so the next Q
// tile's transfer runs under the current tile's gemms, O staged in Q's region, a third and a
// fourth K buffer and a third V buffer.
template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmGdpaQLoopLdsArenaLayout
{
    using PingPong = QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;
    using Shape    = typename Problem::BlockFmhaShape;

    static constexpr index_t kRegionAlignment = PingPong::kRegionAlignment;
    static constexpr index_t kQBytes          = PingPong::kQBytes;
    static constexpr index_t kKBytes          = PingPong::kKBytes;
    static constexpr index_t kVBytes          = PingPong::kVBytes;
    static constexpr index_t kK0Offset        = PingPong::kK0Offset;
    static constexpr index_t kK1Offset        = PingPong::kK1Offset;
    static constexpr index_t kSOffset         = PingPong::kSOffset;
    static constexpr index_t kV0Offset        = PingPong::kV0Offset;
    static constexpr index_t kV1Offset        = PingPong::kV1Offset;
    static constexpr bool kDoubleV            = PingPong::kDoubleV;
    static constexpr bool kSingleK            = PingPong::kSingleK;
    static constexpr index_t kArenaBudget     = PingPong::kArenaBudget;
    static constexpr bool kQOnBothV           = false;
    static constexpr index_t kQOffset =
        integer_least_multiple(kV1Offset + kVBytes, kRegionAlignment);
    // O staged for the async store (StoreOTile): kM0 rows of kN1 values, a row's bytes padded
    // to 32 past a multiple of 128 (the O tile's 16-byte LDS writes then spread over the
    // banks), in Q's region. A query tile's O is stored after the next Q is read into
    // registers, and the region takes the Q after it once the store is done.
    static constexpr index_t kOVec = 16 / static_cast<index_t>(sizeof(typename Problem::ODataType));
    static constexpr index_t kORowBytes =
        Shape::kN1 * static_cast<index_t>(sizeof(typename Problem::ODataType));
    static constexpr index_t kORowStride =
        (kORowBytes + ((32 - kORowBytes) % 128 + 128) % 128) /
        static_cast<index_t>(sizeof(typename Problem::ODataType));
    static constexpr index_t kOBytes =
        Shape::kM0 * kORowStride * static_cast<index_t>(sizeof(typename Problem::ODataType));
    static constexpr index_t kOOffset = kQOffset;
    // K is issued three trips ahead and V two: two more K buffers and a third V buffer after
    // the Q / O region
    static constexpr index_t kK2Offset =
        integer_least_multiple(kQOffset + max(kQBytes, kOBytes), PingPong::kKAlignment);
    static constexpr index_t kK3Offset =
        integer_least_multiple(kK2Offset + kKBytes, PingPong::kKAlignment);
    static constexpr index_t kV2Offset =
        integer_least_multiple(kK3Offset + kKBytes, kRegionAlignment);
    static constexpr index_t kArenaBytes = integer_least_multiple(kV2Offset + kVBytes, 1024);
    // three workgroups on a WGP (the pipeline's kBlockPerCu): a third of its LDS each, which
    // the pipeline checks (QrTdmGdpaResidentLdsArenaLayout derives from this one)
    static constexpr index_t kStreamArenaBudget = 320 * 1024 / 3;
};

// The resident-K/V arena: one workgroup of kTeams teams (4 waves each) takes a WGP's whole LDS.
// Every key tile of a batch of up to kKSlots * kN0 keys is staged once ([K tiles][V tiles]) and
// read by every team; behind them each team has a Q / O region.
template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmGdpaResidentLdsArenaLayout
    : QrTdmGdpaQLoopLdsArenaLayout<Problem, QPadding, KPadding, VPadding>
{
    using Stream   = QrTdmGdpaQLoopLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;
    using PingPong = typename Stream::PingPong;
    using Shape    = typename Stream::Shape;

    static constexpr index_t kTeams   = 3;   // BlockFmhaPipelineQRKSVSTdmGdpaResidentPolicy's
    // key tiles staged at once: a batch's 384 keys (codegen's GFX125_GDPA_RESIDENT_MAX_KEYS)
    static constexpr index_t kKSlots  = 384 / Shape::kN0;
    static constexpr index_t kKStride =
        integer_least_multiple(Stream::kKBytes, PingPong::kKAlignment);
    static constexpr index_t kVStride =
        integer_least_multiple(Stream::kVBytes, Stream::kRegionAlignment);
    static constexpr index_t kKResOffset = 0;
    static constexpr index_t kVResOffset =
        integer_least_multiple(kKSlots * kKStride, Stream::kRegionAlignment);
    // O staged at Q's row stride (Q's rows with QPadding's pad): a wave stages its O rows over
    // the Q rows it read (the QK and the PV gemm give a wave the same rows), so no wave writes
    // a row another has yet to read and a team needs no barrier
    static constexpr index_t kORowStride =
        (Stream::kORowBytes + (QPadding::kEnabled ? QPadding::kPadBytes : 0)) /
        static_cast<index_t>(sizeof(typename Problem::ODataType));
    static constexpr index_t kOBytes =
        Shape::kM0 * kORowStride * static_cast<index_t>(sizeof(typename Problem::ODataType));
    // (Q's LDS rows: kM0 - 1 strides and a row)
    static_assert(Stream::kQBytes ==
                      ((Shape::kM0 - 1) * kORowStride + Shape::kN1) *
                          static_cast<index_t>(sizeof(typename Problem::ODataType)),
                  "O is staged over Q's rows");
    static constexpr index_t kTeamOffset =
        integer_least_multiple(kVResOffset + kKSlots * kVStride, Stream::kRegionAlignment);
    static constexpr index_t kTeamBytes =
        integer_least_multiple(max(Stream::kQBytes, kOBytes), Stream::kRegionAlignment);
    static constexpr index_t kArenaBudget = 320 * 1024; // a WGP's LDS
    static constexpr index_t kArenaBytes =
        integer_least_multiple(kTeamOffset + kTeams * kTeamBytes, 1024);
    static_assert(kArenaBytes <= kArenaBudget);
};

} // namespace detail

struct BlockFmhaPipelineQRKSVSTdmGdpaQLoopPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    // teams of a workgroup (see BlockFmhaPipelineQRKSVSTdmGdpaResidentPolicy)
    static constexpr index_t kTeams = 1;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmGdpaQLoopLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;
};

// BlockFmhaPipelineQRKSVSTdmGdpaQLoop with K and V resident (batches of up to
// LdsArenaLayout::kKSlots key tiles): a workgroup is kTeams teams of the tile's 4 waves, the
// kernel's translation unit defines CK_TILE_WARP_TEAM_SIZE (get_warp_id is then the wave's index
// in its team).
struct BlockFmhaPipelineQRKSVSTdmGdpaResidentPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    static constexpr index_t kTeams = 3;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmGdpaResidentLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;
};

// GDPA (P = act(scale_s S), no softmax) over several query tiles per workgroup: one software
// pipeline walks the (query tile, key tile) pairs, so the K/V transfers run on across query
// tiles, the next Q tile is staged under the current tile's gemms, and each tile's O is
// stored as its last key tile completes. With few keys per batch (TCX: 256)
// a workgroup per query tile spends a third of its time starting and finishing a tile.
//
// With BlockFmhaPipelineQRKSVSTdmGdpaResidentPolicy (kTeams > 1) a workgroup is kTeams teams of
// the tile's 4 waves on a WGP of its own, and K and V are resident: every key tile of the batch
// is staged once and read by every team, so a trip issues and waits for nothing and has no
// barrier (the K / V transfers of every trip cost 11% at TCX's 256 keys, the gfx1250 at its
// power limit). The workgroup's query tiles are split evenly between the teams; a wave
// transfers the Q rows it reads and stages its O over them. A batch of more keys than the LDS
// holds is staged a chunk at a time. gfx1250 at its power limit, 768 x 256 keys x 3 heads, d128
// bf16: 12288 queries a batch (TCX7/8) and 5252 (TCX6) both -13% against the streaming q-loop.
template <typename Problem_,
          typename Policy_          = BlockFmhaPipelineQRKSVSTdmGdpaQLoopPolicy,
          typename LdsReadSchedule_ = QrTdmPingPongLdsReadSchedule<>>
struct BlockFmhaPipelineQRKSVSTdmGdpaQLoop
    : BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_, LdsReadSchedule_>
{
    using PingPong = BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_, LdsReadSchedule_>;
    using Base     = typename PingPong::Base;

    using typename PingPong::FmhaMask;
    using typename PingPong::KDataType;
    using typename PingPong::Policy;
    using typename PingPong::Problem;
    using typename PingPong::QDataType;
    using typename PingPong::SaccDataType;
    using typename PingPong::SMPLComputeDataType;
    using typename PingPong::VDataType;

    using PingPong::I0;
    using PingPong::I1;
    using PingPong::kK0;
    using PingPong::kK1;
    using PingPong::kM0;
    using PingPong::kN0;
    using PingPong::kN1;
    using PingPong::kNWarp;
    using PingPong::kQKHeaddim;
    using PingPong::kSubQKHeaddim;

    // teams of 4 waves in a workgroup, each with query tiles of its own (resident K / V), or 1
    static constexpr index_t kTeams = Policy::kTeams;
    static constexpr bool kResident = kTeams > 1;
    // the workgroup's threads (FmhaFwdKernel launches kBlockSize)
    static constexpr index_t kBlockSize = PingPong::kBlockSize * kTeams;
    // query tiles a workgroup takes (FmhaFwdKernel sizes the grid by kM0 * kQTilesPerBlock)
    static constexpr bool kQTileLoop         = true;
    // (resident: 32, a (batch, head) of TCX7/8's 12288 queries a workgroup, K and V staged
    // once for it; TCX6/7/8 at 4 .. 32: the more the faster)
    static constexpr index_t kTeamQTiles     = kResident ? 32 : 8;
    static constexpr index_t kQTilesPerBlock = kTeamQTiles * kTeams;
    // three workgroups per WGP (the arena takes a third of the LDS), one wave per SIMD each
    // (resident: one workgroup of three teams): the kernel's launch bound holds the registers
    // to three waves per SIMD (at the default two the compiler took 338 VGPRs, two waves, +12%
    // at 256 keys)
    static constexpr index_t kBlockPerCu = 3;
    // the next Q is issued this many trips before a query tile's last (a trip head waits for
    // the transfers of two heads before it, so at three the Q has landed by the last trip's)
    static constexpr index_t kQIssueAhead = 3;
    // how many WMMAs ahead of its WMMA a B-operand read is issued, QK and PV gemm (measured
    // on gfx1250 at TCX's 256 keys)
    // (resident: every read of a gemm ahead of its WMMAs, -1.3% / -1.9% at TCX7/8 / TCX6)
    static constexpr index_t kQkReadAheadWmma = kResident ? 16 : 8;
    static constexpr index_t kPvReadAheadWmma = kResident ? 16 : 12;

    static_assert(!Base::kIsSoftmax, "qr_tdm_gdpa_qloop: GDPA only");
    static_assert(
        [] {
            if constexpr(kResident)
                return Policy::template LdsArenaLayout<Problem>::kTeams == kTeams;
            else
                return true;
        }(),
        "qr_tdm_gdpa_qloop: the arena holds a Q / O region a team");
    static_assert(kResident || Policy::template LdsArenaLayout<Problem>::kArenaBytes <=
                                   Policy::template LdsArenaLayout<Problem>::kStreamArenaBudget,
                  "qr_tdm_gdpa_qloop: three workgroups on a WGP");
    // its own main loop: of qr_tdm_pingpong's schedule only the two V buffers carry over
    static_assert(PingPong::kDoubleV && kNWarp == 1,
                  "qr_tdm_gdpa_qloop: the double-V 128-row tile");
    static_assert(!FmhaMask::IsMasking, "qr_tdm_gdpa_qloop: no mask");

    // Stores a query tile's O: staged in LDS (o_lds) in the O tile's distribution, then copied
    // to o_dram_window with async_store_tile_from_lds, each wave the rows it staged (so the
    // wave waits on its own LDS writes only). Rows and columns past o_dram_window's tensor are
    // not written (bounds checked only for a tile that crosses them). The copy is in flight on
    // return (ASYNCcnt): the caller waits before o_lds is written again.
    template <typename ODramWindow, typename OAccTile>
    CK_TILE_DEVICE static void
    StoreOTile(ODramWindow& o_dram_window, const OAccTile& o_acc_tile, void* o_lds)
    {
        using ODataType = remove_cvref_t<typename ODramWindow::DataType>;
        using Layout    = typename Policy::template LdsArenaLayout<Problem>;
        // the O tile's rows: m * kMWarp * kWarpM + w * kWarpM + [0, kWarpM), m < kMIter, for
        // wave w
        using OEncoding          = typename OAccTile::StaticTileDistribution::DstrEncode;
        constexpr index_t kMIter = OEncoding::hs_lengthss_[number<0>{}][number<0>{}];
        constexpr index_t kMWarp = OEncoding::hs_lengthss_[number<0>{}][number<1>{}];
        constexpr index_t kWarpM = kM0 / (kMIter * kMWarp);
        static_assert(OEncoding::ps_to_rhss_major_[number<0>{}][number<0>{}] == 1 &&
                          OEncoding::ps_to_rhss_minor_[number<0>{}][number<0>{}] == 1,
                      "the wave index selects the M warp of the O tile");
        constexpr index_t kVec    = Layout::kOVec;
        constexpr index_t kLanesN = kN1 / kVec;
        constexpr index_t kLanesM = get_warp_size() / kLanesN;
        static_assert(kN1 % kVec == 0 && get_warp_size() % kLanesN == 0 && kWarpM % kLanesM == 0);

        auto o_lds_view = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(o_lds),
            make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN1>{}),
                                         make_tuple(number<Layout::kORowStride>{}, number<1>{}),
                                         number<kVec>{},
                                         number<1>{}));
        auto o_lds_write_window = make_tile_window(o_lds_view,
                                                   make_tuple(number<kM0>{}, number<kN1>{}),
                                                   {0, 0},
                                                   o_acc_tile.get_tile_distribution());
        // 16-byte vectors along a row, the wave's rows as in the O tile
        auto o_lds_copy_window = make_tile_window(
            o_lds_view,
            make_tuple(number<kM0>{}, number<kN1>{}),
            {0, 0},
            make_static_tile_distribution(tile_distribution_encoding<
                                          sequence<>,
                                          tuple<sequence<kMIter, kMWarp, kWarpM / kLanesM, kLanesM>,
                                                sequence<kLanesN, kVec>>,
                                          tuple<sequence<1>, sequence<1, 2>>,
                                          tuple<sequence<1>, sequence<3, 0>>,
                                          sequence<1, 1, 2>,
                                          sequence<0, 2, 1>>{}));

        store_tile(o_lds_write_window, cast_tile<ODataType>(o_acc_tile));
        s_wait_dscnt<0>();
        // a tile inside O (all but a batch's last) copies without the per-access bounds check
        const auto o_lengths =
            o_dram_window.get_bottom_tensor_view().get_tensor_descriptor().get_lengths();
        const auto o_origin = o_dram_window.get_window_origin();
        if(o_origin[number<0>{}] + kM0 <= o_lengths[number<0>{}] &&
           o_origin[number<1>{}] + kN1 <= o_lengths[number<1>{}])
            async_store_tile_from_lds(o_dram_window, o_lds_copy_window, bool_constant<false>{});
        else
            async_store_tile_from_lds(o_dram_window, o_lds_copy_window);
    }

    // FmhaFwdKernel runs RunQTiles (on its untransposed-V path, kQTileLoop) and returns; its
    // one-tile code after that, which it still compiles, takes this instead of qr_tdm_pingpong's
    // main loop. Never run: it traps rather than leave O unwritten.
    static_assert(PingPong::kUsesUntransposedVKernelPath);
    template <typename... Args>
    CK_TILE_DEVICE auto operator()(Args&&...) const
    {
        __builtin_trap();
        return Policy::template GetPVBlockGemm<Problem>().MakeCBlockTile();
    }

    // The activation's constant factor (attention_activation_out_factor) goes on O, once per query
    // tile, instead of on every score; P is then 2 / scale_s times larger, which a bf16 P holds
    // (fp16 would overflow at much smaller scores, so it takes the factor on P).
    static constexpr bool kActFactorOnO = std::is_same_v<typename Problem::PDataType, bf16_t>;
    // Resident K / V are scaled as they are staged, once a workgroup: K by
    // attention_activation_prescale (the activation then takes a packed operation less a pair
    // of scores) and V by the output factor (no pass over O at a query tile's end). Each is
    // rounded to its type once more.
    static constexpr bool kPrescaleKV = kResident && kActFactorOnO;

    // The workgroup's query tiles, one trip per (query tile, key tile) pair: a trip's head (a
    // barrier, every wave done with the last trip's buffers) waits for the transfers issued
    // before the last head (its V and K, and the next trip's K: its first slice is read under
    // this trip's PV gemm and the next QK starts on registers) and issues V two trips ahead and
    // K three (three V and four K buffers, rotating), then, kQIssueAhead trips before a query
    // tile's last, the next Q. Waiting for every transfer instead (V and K a trip ahead and two)
    // GDPA took 2.2% longer at TCX's 256 keys.
    // store_o(o_acc, i_tile, o_lds) stores query tile i_tile (of this workgroup) from o_acc
    // through o_lds, Q's region: the next Q is read into registers first, and the copy is
    // waited for before that region takes the Q after it.
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename StoreO>
    CK_TILE_DEVICE void RunQTiles(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                  const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                  const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                  FmhaMask mask,
                                  index_t num_block_q_tiles,
                                  index_t seqlen_k,
                                  float scale_s,
                                  void* smem_arena,
                                  StoreO&& store_o) const
    {
        using Layout = typename Policy::template LdsArenaLayout<Problem>;

        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();
        using Gemm0T          = remove_cvref_t<decltype(gemm_0)>;
        using Gemm1T          = remove_cvref_t<decltype(gemm_1)>;

        auto s_acc = gemm_0.MakeCBlockTile();
        auto o_acc = gemm_1.MakeCBlockTile();

        const TDMConfig tdm_config_q =
            detail::make_tdm_config<typename Policy::template LdsPaddingConfigQ<Problem>>();
        const TDMConfig tdm_config_k =
            detail::make_tdm_config<typename Policy::template LdsPaddingConfigK<Problem>>();
        const TDMConfig tdm_config_v =
            detail::make_tdm_config<typename Policy::template LdsPaddingConfigV<Problem>>();

        char* const arena = static_cast<char*>(smem_arena);
        // this wave's team: its query tiles of the workgroup's and its Q / O region
        // (the workgroup's tiles split evenly, the first teams one more: a batch's last
        // workgroup keeps every team busy)
        const index_t team         = kResident ? get_warp_team_id() : 0;
        const index_t team_share   = num_block_q_tiles / kTeams;
        const index_t team_extra   = num_block_q_tiles - team_share * kTeams;
        const index_t first_q_tile = team * team_share + min(team, team_extra);
        const index_t num_q_tiles  =
            kResident ? team_share + (team < team_extra ? 1 : 0) : num_block_q_tiles;
        char* const q_region = [&]() {
            if constexpr(kResident)
                return arena + Layout::kTeamOffset + team * Layout::kTeamBytes;
            else
                return arena + Layout::kQOffset;
        }();

        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());
        if constexpr(kResident)
            move_tile_window(q_dram_window, {first_q_tile * kM0, 0});
        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QDataType*>(q_region),
            Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {0, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());
        // the K buffers of this trip, the next two and a free one (the last trip's), rotated
        // at the end of a trip: no buffer arithmetic at a trip head. Resident: this trip's key
        // tile and the next trip's (k_buf2 and k_buf3 unused), walking the staged tiles.
        constexpr auto kBufOffsets = [] {
            if constexpr(kResident)
                return array<index_t, 7>{Layout::kKResOffset,
                                         Layout::kKResOffset,
                                         Layout::kKResOffset,
                                         Layout::kKResOffset,
                                         Layout::kVResOffset,
                                         Layout::kVResOffset,
                                         Layout::kVResOffset};
            else
                return array<index_t, 7>{Layout::kK0Offset,
                                         Layout::kK1Offset,
                                         Layout::kK2Offset,
                                         Layout::kK3Offset,
                                         Layout::kV0Offset,
                                         Layout::kV1Offset,
                                         Layout::kV2Offset};
        }();
        // a resident key tile's bytes, K and V
        constexpr auto kResStrides = [] {
            if constexpr(kResident)
                return array<index_t, 2>{Layout::kKStride, Layout::kVStride};
            else
                return array<index_t, 2>{0, 0};
        }();
        KDataType* k_buf0 = reinterpret_cast<KDataType*>(arena + kBufOffsets[0]);
        KDataType* k_buf1 = reinterpret_cast<KDataType*>(arena + kBufOffsets[1]);
        KDataType* k_buf2 = reinterpret_cast<KDataType*>(arena + kBufOffsets[2]);
        KDataType* k_buf3 = reinterpret_cast<KDataType*>(arena + kBufOffsets[3]);
        auto k_lds_view   = make_tensor_view<address_space_enum::lds>(
            k_buf0, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_write_window = make_tile_window(
            k_lds_view, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto k_lds_read_window =
            make_tile_window(k_lds_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             PingPong::VSeqOffset(0),
                             Policy::template MakeVDramTileDistribution<Problem>());
        // the V buffers of this trip, the next one and a free one, rotated as K's
        VDataType* v_buf0 = reinterpret_cast<VDataType*>(arena + kBufOffsets[4]);
        VDataType* v_buf1 = reinterpret_cast<VDataType*>(arena + kBufOffsets[5]);
        VDataType* v_buf2 = reinterpret_cast<VDataType*>(arena + kBufOffsets[6]);
        auto v_lds_view   = make_tensor_view<address_space_enum::lds>(
            v_buf0, Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_write_window = make_tile_window(
            v_lds_view, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto v_lds_read_window =
            make_tile_window(v_lds_view,
                             PingPong::VReadLengths(),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        const index_t nk           = integer_divide_ceil(seqlen_k, kN0);
        const auto q_origin        = q_dram_block_window_tmp.get_window_origin();
        const index_t q_issue_trip = nk > kQIssueAhead ? nk - kQIssueAhead : 0;
        const index_t num_total    = nk * num_q_tiles;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        char* const o_stage        = q_region;
        // the staged key tiles (resident)
        char* const k_res     = arena + kBufOffsets[0];
        char* const v_res     = arena + kBufOffsets[4];
        char* k_res_end = k_res + nk * kResStrides[0];
        char* v_res_end = v_res + nk * kResStrides[1];
        if constexpr(kResident)
            k_buf1 = reinterpret_cast<KDataType*>(nk > 1 ? k_res + kResStrides[0] : k_res);
        // query tile i_q of this team, of the workgroup's
        const auto store_o_tile = [&](const auto& o_tile, index_t i_q) {
            store_o(o_tile, i_q + first_q_tile, static_cast<void*>(o_stage));
        };

        // no keys (a batch's empty key set): O is 0
        if(nk == 0)
        {
            clear_tile(o_acc);
            for(index_t i_q = 0; i_q < num_q_tiles; ++i_q)
            {
                store_o_tile(o_acc, i_q);
                s_wait_asynccnt<0>(); // the next tile stages in the same region
            }
            return;
        }

        // K / V of the next key tile into a buffer: the key tiles in order, round and round (the
        // windows walk them and go back to the first after the last). Past the last trip they go
        // on into free buffers, unread: a branch at every trip head cost more than these few
        // tiles. Each is one TDM instruction per wave (a window has one coordinate).
        index_t k_tile_next = 0;
        index_t v_tile_next = 0;
        const auto issue_k  = [&](KDataType* buf) {
            k_lds_write_window.set_bottom_tensor_view_data_ptr(buf);
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            const bool wrap = ++k_tile_next == nk;
            move_tile_window(k_dram_window, {wrap ? -(nk - 1) * kN0 : kN0, 0});
            k_tile_next = wrap ? 0 : k_tile_next;
        };
        const auto issue_v = [&](VDataType* buf) {
            v_lds_write_window.set_bottom_tensor_view_data_ptr(buf);
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
            const bool wrap = ++v_tile_next == nk;
            move_tile_window(v_dram_window, PingPong::VSeqOffset(wrap ? -(nk - 1) * kN0 : kN0));
            v_tile_next = wrap ? 0 : v_tile_next;
        };
        // a trip's first kK0 slice of K, read ahead of its QK
        using KTile           = decltype(load_tile(k_lds_read_window));
        auto k_tiles          = generate_tuple([](auto) { return KTile{}; }, number<k0_loops>{});
        const auto prefetch_k = [&](KDataType* buf) {
            k_lds_read_window.set_bottom_tensor_view_data_ptr(buf);
            k_tiles(I0) = load_tile(k_lds_read_window);
            if constexpr(k0_loops > 1)
                move_tile_window(k_lds_read_window, {0, kK0});
        };

        // The next query tile's Q. Resident: each wave the rows it reads (the QK gemm's: its
        // rows of each M iteration, one box an iteration), so a wave waits for its own
        // transfers only and its team needs no barrier.
        using QkGemm              = remove_cvref_t<decltype(gemm_0)>;
        constexpr index_t kQBands = QkGemm::MIterPerWarp;
        constexpr index_t kQBand  = kM0 / kQBands; // an M iteration's rows
        constexpr index_t kWaves  = Problem::kBlockSize / get_warp_size();
        auto q_band_window        = make_tile_window(
            q_dram_window.get_bottom_tensor_view(),
            make_tuple(number<kQBand>{}, number<kSubQKHeaddim>{}),
            q_dram_window.get_window_origin(),
            make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<kWaves, kQBand / kWaves>,
                                                 sequence<kSubQKHeaddim>>,
                                           tuple<sequence<1>>,
                                           tuple<sequence<0>>,
                                           sequence<1, 2>,
                                           sequence<1, 0>>{},
                bool_constant<true>{}));
        const auto issue_q = [&]() {
            if constexpr(kResident)
            {
                static_for<0, kQBands, 1>{}([&](auto m) {
                    auto q_band_lds =
                        make_tile_window(q_lds_view,
                                         make_tuple(number<kQBand>{}, number<kSubQKHeaddim>{}),
                                         {m * kQBand, 0});
                    load_tile_tdm(tdm_config_q, q_band_lds, q_band_window);
                    move_tile_window(q_band_window, {kQBand, 0});
                });
            }
            else
            {
                load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
                move_tile_window(q_dram_window, {kM0, 0});
            }
        };

        // A trip: S = Q K_i^T, P = act(S), O += P V_i, the next trip's first K slice read under
        // the PV gemm. Resident: then on to the next staged key tile, the first after the last.
        auto q_tile = decltype(load_tile(q_lds_read_window)){};
        index_t i_k = 0; // key tile of this trip
        const auto trip = [&]() {
            // S = Q K^T: the first slice already in registers
            clear_tile(s_acc);
            static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                if constexpr(i_k0 + 1 < k0_loops)
                {
                    k_tiles(number<i_k0 + 1>{}) = load_tile(k_lds_read_window);
                    if constexpr(i_k0 + 2 < k0_loops)
                        move_tile_window(k_lds_read_window, {0, kK0});
                }
                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      sequence<0, i_k0 * kK0>{},
                                      sequence<kM0, (i_k0 + 1) * kK0>{}),
                       k_tiles[i_k0]);
            });
            PingPong::template SchedReadsAheadOfWmma<Gemm0T::MIterPerWarp,
                                                     Gemm0T::MIterPerWarp *
                                                         Gemm0T::NIterPerWarp * k0_loops,
                                                     kQkReadAheadWmma>();
            move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});

            const index_t k_start = i_k * kN0;
            if(mask.IsEdgeTile(q_origin.at(I0), k_start, number<kM0>{}, number<kN0>{}))
            {
                set_tile_if(
                    s_acc, type_convert<SaccDataType>(Base::kMaskedScore), [&](auto tile_idx) {
                        return mask.IsOutOfBound(q_origin.at(I0) + tile_idx.at(I0),
                                                 k_start + tile_idx.at(I1));
                    });
            }

            // P = act(scale_s S), without its constant factor where O takes it
            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s_acc.get_tile_distribution());
            if constexpr(kPrescaleKV)
                attention_activation_prescaled_x2<Base::Activation>(
                    s_acc.get_thread_buffer(), p_compute.get_thread_buffer());
            else
                attention_activation_x2<Base::Activation, kActFactorOnO>(
                    s_acc.get_thread_buffer(), p_compute.get_thread_buffer(), scale_s);
            [[maybe_unused]] int32_t p_scale = 0;
            auto p_tile = Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);

            // the next trip's first K slice, under this trip's PV gemm (past the last trip:
            // a landed tile, unused)
            prefetch_k(k_buf1);

            // O += P V_i
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_buf0);
            auto v_tile = PingPong::LoadVTile(v_lds_read_window);
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                auto p_slice = get_slice_tile(
                    p_tile, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{});
                if constexpr(i_k1 + 1 < k1_loops)
                {
                    gemm_1(o_acc, p_slice, v_tile);
                    move_tile_window(v_lds_read_window, PingPong::VSeqOffset(kK1));
                    v_tile = PingPong::LoadVTile(v_lds_read_window);
                }
                else
                    gemm_1(o_acc, p_slice, v_tile);
            });
            move_tile_window(v_lds_read_window, PingPong::VSeqOffset(-kK1 * (k1_loops - 1)));
            PingPong::template SchedReadsAheadOfWmma<Gemm1T::MIterPerWarp,
                                                     Gemm1T::MIterPerWarp *
                                                         Gemm1T::NIterPerWarp * k1_loops,
                                                     kPvReadAheadWmma>();
            if constexpr(kResident)
            {
                k_buf0              = k_buf1;
                char* const k_after = reinterpret_cast<char*>(k_buf1) + kResStrides[0];
                k_buf1 = reinterpret_cast<KDataType*>(k_after == k_res_end ? k_res : k_after);
                char* const v_after = reinterpret_cast<char*>(v_buf0) + kResStrides[1];
                v_buf0 = reinterpret_cast<VDataType*>(v_after == v_res_end ? v_res : v_after);
            }
        };

        // Resident: the first team stages `tiles` key tiles (the next ones of the batch, round
        // and round) and, once they landed and every wave is past a barrier, scales them in place
        // (its waves a tile's rows, the lanes a row's columns: kPrescaleKV).
        const auto stage_tiles = [&](index_t tiles) {
            if(team == 0)
            {
                for(index_t t = 0; t < tiles; ++t)
                {
                    issue_k(reinterpret_cast<KDataType*>(k_res + t * kResStrides[0]));
                    issue_v(reinterpret_cast<VDataType*>(v_res + t * kResStrides[1]));
                }
            }
        };
        const auto prescale_tiles = [&](index_t tiles) {
            if constexpr(kPrescaleKV)
            {
                if(team == 0)
                {
                    const auto scale_tiles = [&](auto& lds_view,
                                                 char* base,
                                                 index_t stride,
                                                 float factor,
                                                 auto cols) {
                        constexpr index_t kCols  = decltype(cols)::value;
                        constexpr index_t kLanes = get_warp_size();
                        static_assert(kN0 % kWaves == 0 && kCols % kLanes == 0);
                        auto window = make_tile_window(
                            lds_view,
                            make_tuple(number<kN0>{}, number<kCols>{}),
                            {0, 0},
                            make_static_tile_distribution(tile_distribution_encoding<
                                                          sequence<>,
                                                          tuple<sequence<kWaves, kN0 / kWaves>,
                                                                sequence<kLanes, kCols / kLanes>>,
                                                          tuple<sequence<1>, sequence<2>>,
                                                          tuple<sequence<0>, sequence<0>>,
                                                          sequence<1, 2>,
                                                          sequence<1, 1>>{}));
                        using DataType = remove_cvref_t<typename decltype(window)::DataType>;
                        for(index_t t = 0; t < tiles; ++t)
                        {
                            window.set_bottom_tensor_view_data_ptr(
                                reinterpret_cast<DataType*>(base + t * stride));
                            auto rows = cast_tile<float>(load_tile(window));
                            tile_elementwise_inout([&](auto& x) { x *= factor; }, rows);
                            store_tile(window, cast_tile<DataType>(rows));
                        }
                    };
                    scale_tiles(k_lds_view,
                                k_res,
                                kResStrides[0],
                                attention_activation_prescale<Base::Activation>(scale_s),
                                number<kSubQKHeaddim>{});
                    scale_tiles(v_lds_view,
                                v_res,
                                kResStrides[1],
                                attention_activation_prescaled_out_factor<Base::Activation>(),
                                number<kN1>{});
                }
            }
            else
                ignore = tiles;
        };
        // the staged tiles the trips walk
        const auto walk_tiles = [&](index_t tiles) {
            k_res_end = k_res + tiles * kResStrides[0];
            v_res_end = v_res + tiles * kResStrides[1];
            k_buf0    = reinterpret_cast<KDataType*>(k_res);
            k_buf1    = reinterpret_cast<KDataType*>(tiles > 1 ? k_res + kResStrides[0] : k_res);
            v_buf0    = reinterpret_cast<VDataType*>(v_res);
        };

        if constexpr(kResident)
        {
            static_assert(sizeof(QDataType) == sizeof(typename Problem::ODataType) &&
                              kSubQKHeaddim == kN1,
                          "O is staged over the Q rows");
            if(nk > Layout::kKSlots)
            {
                // More key tiles than the LDS holds: a query tile takes them a chunk of kKSlots at
                // a time, each staged for every team at once. The teams stay in step (three
                // barriers a chunk: every wave done with the last chunk, the chunk landed, the
                // chunk scaled), a team past its last query tile only keeping the barriers.
                const index_t steps = team_share + (team_extra > 0 ? 1 : 0);
                bool k_slice_read   = false;
                for(index_t i_q = 0; i_q < steps; ++i_q)
                {
                    const bool mine = i_q < num_q_tiles;
                    if(mine)
                    {
                        s_wait_asynccnt<0>(); // the last O copy out of this wave's Q rows
                        issue_q();
                        clear_tile(o_acc);
                    }
                    for(index_t t0 = 0; t0 < nk; t0 += Layout::kKSlots)
                    {
                        const index_t tiles = min(index_t{Layout::kKSlots}, nk - t0);
                        block_sync_lds<0>();
                        stage_tiles(tiles);
                        s_wait_tensorcnt_barrier<0>();
                        prescale_tiles(tiles);
                        block_sync_lds<0>();
                        if(mine)
                        {
                            if(t0 == 0)
                                q_tile = load_tile(q_lds_read_window);
                            walk_tiles(tiles);
                            // (the last trip read the first slice of a tile since restaged, and
                            // left the read window a slice on)
                            if constexpr(k0_loops > 1)
                            {
                                if(k_slice_read)
                                    move_tile_window(k_lds_read_window, {0, -kK0});
                            }
                            k_slice_read = true;
                            prefetch_k(k_buf0);
                            for(index_t t = 0; t < tiles; ++t)
                            {
                                i_k = t0 + t;
                                trip();
                            }
                        }
                    }
                    if(mine)
                    {
                        if constexpr(kActFactorOnO && !kPrescaleKV)
                        {
                            const float f =
                                attention_activation_out_factor<Base::Activation>(scale_s);
                            tile_elementwise_inout([&](auto& o) { o *= f; }, o_acc);
                        }
                        s_wait_dscnt<0>(); // Q read: its rows take this wave's O
                        store_o_tile(o_acc, i_q);
                    }
                }
                s_wait_tensorcnt<0>();
                s_wait_asynccnt<0>();
                return;
            }
        }

        if constexpr(kResident)
        {
            // prologue: the first team stages every key tile, each team its Q_0; every transfer
            // landed (the workgroup's only barrier). A team past the batch's last query tile
            // is done.
            stage_tiles(nk);
            if(num_q_tiles > 0)
                issue_q();
            s_wait_tensorcnt_barrier<0>();
            if constexpr(kPrescaleKV)
            {
                prescale_tiles(nk);
                block_sync_lds<0>();
            }
            if(num_q_tiles <= 0)
                return;
        }
        else
        {
            // prologue: Q_0, V of the first two trips, K of the first three; every transfer
            // landed
            issue_q();
            issue_v(v_buf0);
            issue_v(v_buf1);
            issue_k(k_buf0);
            issue_k(k_buf1);
            issue_k(k_buf2);
            s_wait_tensorcnt_barrier<0>();
        }
        q_tile = load_tile(q_lds_read_window);
        prefetch_k(k_buf0);

        clear_tile(o_acc);
        index_t i_q     = 0; // query tile of this trip
        index_t i_total = 0;
        do
        {
            if constexpr(!kResident)
            {
                // every transfer but the last head's V and K landed (they retire in issue
                // order; a Q issued after them is waited for with its V): this trip's V and K
                // and the next trip's K; every wave is done with the last trip's buffers, which
                // take V two trips ahead and K three
                s_wait_tensorcnt<2>();
                block_sync_lds<0>();
                issue_v(v_buf2);
                issue_k(k_buf3);
            }
            if(i_k == q_issue_trip)
            {
                // Q's region takes the next Q: the last O copy out of it done
                s_wait_asynccnt<0>();
                if(i_q + 1 < num_q_tiles)
                    issue_q();
            }

            trip();

            if(i_k + 1 == nk)
            {
                if(i_q + 1 < num_q_tiles)
                {
                    if constexpr(kResident)
                    {
                        // the next Q into registers: the wave's own rows, which then take its O
                        s_wait_tensorcnt<0>();
                        q_tile = load_tile(q_lds_read_window);
                        s_wait_dscnt<0>();
                    }
                    else
                    {
                        // the next Q into registers: a head has waited for it two heads after
                        // its issue, else it is waited for here
                        if(i_k < q_issue_trip + 2)
                            s_wait_tensorcnt_barrier<0>();
                        q_tile = load_tile(q_lds_read_window);
                        block_sync_lds<0>(); // every wave read Q: its region takes the O shuffle
                    }
                }
                if constexpr(kActFactorOnO && !kPrescaleKV)
                {
                    const float f = attention_activation_out_factor<Base::Activation>(scale_s);
                    tile_elementwise_inout([&](auto& o) { o *= f; }, o_acc);
                }
                store_o_tile(o_acc, i_q);
                clear_tile(o_acc);
                ++i_q;
            }
            i_k = i_k + 1 == nk ? 0 : i_k + 1;
            if constexpr(!kResident)
            {
                // the buffers move down a trip, this trip's becoming the free ones
                KDataType* const k_done = k_buf0;
                k_buf0                  = k_buf1;
                k_buf1                  = k_buf2;
                k_buf2                  = k_buf3;
                k_buf3                  = k_done;
                VDataType* const v_done = v_buf0;
                v_buf0                  = v_buf1;
                v_buf1                  = v_buf2;
                v_buf2                  = v_done;
            }
        } while(++i_total < num_total);

        // the transfers past the last trip, before the workgroup's LDS goes to another
        s_wait_tensorcnt<0>();
        s_wait_asynccnt<0>();
    }
};

} // namespace ck_tile
