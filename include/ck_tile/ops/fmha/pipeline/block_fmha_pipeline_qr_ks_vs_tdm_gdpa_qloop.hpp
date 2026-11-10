// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong_policy.hpp"

namespace ck_tile {

namespace detail {

// qr_tdm_pingpong's arena (K0, K1, S, V0, V1) with Q in a region of its own, so the next Q
// tile's transfer runs under the current tile's gemms, O staged in Q's region, and a third K
// buffer.
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
    // a third K buffer (K issued two trips ahead) after the Q / O region
    static constexpr index_t kK2Offset =
        integer_least_multiple(kQOffset + max(kQBytes, kOBytes), PingPong::kKAlignment);
    static constexpr index_t kArenaBytes = integer_least_multiple(kK2Offset + kKBytes, 1024);
    // three resident workgroups per WGP (the pipeline's kBlockPerCu)
    static_assert(kArenaBytes <= 320 * 1024 / 3);
};

} // namespace detail

struct BlockFmhaPipelineQRKSVSTdmGdpaQLoopPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmGdpaQLoopLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;
};

// GDPA (P = act(scale_s S), no softmax) over several query tiles per workgroup: one software
// pipeline walks the (query tile, key tile) pairs, so the K/V transfers run on across query
// tiles, the next Q tile is staged under the current tile's gemms, and each tile's O is
// stored as its last key tile completes. With few keys per batch (TCX: 256)
// a workgroup per query tile spends a third of its time starting and finishing a tile.
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

    // query tiles a workgroup takes (FmhaFwdKernel sizes the grid by kM0 * kQTilesPerBlock)
    static constexpr bool kQTileLoop         = true;
    static constexpr index_t kQTilesPerBlock = 8;
    // three workgroups per WGP (the arena takes a third of the LDS), one wave per SIMD each:
    // the kernel's launch bound holds the registers to three waves per SIMD (at the default
    // two the compiler took 338 VGPRs, two waves, +12% at 256 keys)
    static constexpr index_t kBlockPerCu = 3;
    // the next Q is issued this many trips before a query tile's last
    static constexpr index_t kQIssueAhead = 2;
    // how many WMMAs ahead of its WMMA a B-operand read is issued, QK and PV gemm (measured
    // on gfx1250 at TCX's 256 keys)
    static constexpr index_t kQkReadAheadWmma = 8;
    static constexpr index_t kPvReadAheadWmma = 12;

    static_assert(!Base::kIsSoftmax, "qr_tdm_gdpa_qloop: GDPA only");
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

    // The workgroup's query tiles, one trip per (query tile, key tile) pair: a trip's head (a
    // barrier, every wave done with the last trip's buffers) waits for its V and K and issues
    // V of the next trip, K two trips ahead (three K buffers, so the next trip's K has landed
    // by this head too: its first slice is read under this trip's PV gemm and the next QK
    // starts on registers) and, kQIssueAhead trips before a query tile's last, the next Q
    // (ahead of them).
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
                                  index_t num_q_tiles,
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

        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());
        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QDataType*>(arena + Layout::kQOffset),
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
        KDataType* const k_buf0 = reinterpret_cast<KDataType*>(arena + Layout::kK0Offset);
        KDataType* const k_buf1 = reinterpret_cast<KDataType*>(arena + Layout::kK1Offset);
        KDataType* const k_buf2 = reinterpret_cast<KDataType*>(arena + Layout::kK2Offset);
        // buffers picked with ?: (as qr_tdm_pingpong): an array indexed at run time would lose
        // the LDS address space, and the K reads would go out as flat loads
        const auto k_buf = [&](index_t slot) {
            return slot == 0 ? k_buf0 : (slot == 1 ? k_buf1 : k_buf2);
        };
        auto k_lds_view = make_tensor_view<address_space_enum::lds>(
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
        VDataType* const v_buf0 = reinterpret_cast<VDataType*>(arena + Layout::kV0Offset);
        VDataType* const v_buf1 = reinterpret_cast<VDataType*>(arena + Layout::kV1Offset);
        auto v_lds_view         = make_tensor_view<address_space_enum::lds>(
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
        char* const o_stage        = arena + Layout::kOOffset;

        // no keys (a batch's empty key set): O is 0
        if(nk == 0)
        {
            clear_tile(o_acc);
            for(index_t i_q = 0; i_q < num_q_tiles; ++i_q)
            {
                store_o(o_acc, i_q, static_cast<void*>(o_stage));
                s_wait_asynccnt<0>(); // the next tile stages in the same region
            }
            return;
        }

        // K / V of key tile `tile` into a slot. The windows walk the key tiles in order, back to
        // the first after the last, so they stand at `tile` and only its wrap is needed.
        const auto issue_k = [&](index_t tile, index_t slot) {
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_buf(slot));
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            move_tile_window(k_dram_window, {tile + 1 == nk ? -(nk - 1) * kN0 : kN0, 0});
        };
        const auto issue_v = [&](index_t tile, index_t slot) {
            v_lds_write_window.set_bottom_tensor_view_data_ptr(slot ? v_buf1 : v_buf0);
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
            move_tile_window(v_dram_window,
                             PingPong::VSeqOffset(tile + 1 == nk ? -(nk - 1) * kN0 : kN0));
        };
        // the key tile `ahead` (1 or 2) after key tile i_k
        const auto tile_after = [&](index_t i_k, index_t ahead) {
            index_t t = i_k + ahead;
            t -= t >= nk ? nk : 0;
            return t >= nk ? t - nk : t; // nk == 1
        };
        // a trip's first kK0 slice of K, read ahead of its QK
        using KTile           = decltype(load_tile(k_lds_read_window));
        auto k_tiles          = generate_tuple([](auto) { return KTile{}; }, number<k0_loops>{});
        const auto prefetch_k = [&](index_t slot) {
            k_lds_read_window.set_bottom_tensor_view_data_ptr(k_buf(slot));
            k_tiles(I0) = load_tile(k_lds_read_window);
            if constexpr(k0_loops > 1)
                move_tile_window(k_lds_read_window, {0, kK0});
        };

        // prologue: Q_0, V_0, K_0, K_1; every transfer landed
        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        move_tile_window(q_dram_window, {kM0, 0});
        issue_v(0, 0);
        issue_k(0, 0);
        if(num_total > 1)
            issue_k(tile_after(0, 1), 1);
        s_wait_tensorcnt_barrier<0>();
        auto q_tile = load_tile(q_lds_read_window);
        prefetch_k(0);

        clear_tile(o_acc);
        index_t i_k     = 0;     // key tile of this trip
        index_t i_q     = 0;     // query tile of this trip
        index_t k_slot  = 0;     // the K slot of this trip
        index_t v_slot  = 0;     // the V slot of this trip
        bool q_pending  = false; // a Q was issued at this trip's head
        bool o_pending  = false; // the last query tile's O copy may be in flight
        index_t i_total = 0;
        do
        {
            // Q's region takes the next Q at this head: the last O copy out of it done
            if(o_pending && i_k == q_issue_trip)
            {
                s_wait_asynccnt<0>();
                o_pending = false;
            }
            // V, K (and a Q) of this trip landed (the next trip's K, issued a trip earlier,
            // retired before them); every wave is done with the last trip's buffers. The Q goes
            // out ahead of the head's V and K: behind them (the next head waiting for all but
            // the Q) GDPA at TCX's 256 keys took 1.6% longer.
            s_wait_tensorcnt_barrier<0, 0>();
            const index_t next_slot = k_slot == 2 ? 0 : k_slot + 1;
            q_pending               = i_k == q_issue_trip && i_q + 1 < num_q_tiles;
            if(q_pending)
            {
                load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
                move_tile_window(q_dram_window, {kM0, 0});
            }
            if(i_total + 1 < num_total)
                issue_v(tile_after(i_k, 1), v_slot ^ 1);
            if(i_total + 2 < num_total)
                issue_k(tile_after(i_k, 2), next_slot == 2 ? 0 : next_slot + 1);

            {
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
                                                         0,
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
                attention_activation_x2<Base::Activation, kActFactorOnO>(
                    s_acc.get_thread_buffer(), p_compute.get_thread_buffer(), scale_s);
                [[maybe_unused]] int32_t p_scale = 0;
                auto p_tile = Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);

                // the next trip's first K slice, under this trip's PV gemm
                if(i_total + 1 < num_total)
                    prefetch_k(next_slot);

                // O += P V_i
                v_lds_read_window.set_bottom_tensor_view_data_ptr(v_slot ? v_buf1 : v_buf0);
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
                                                         0,
                                                         kPvReadAheadWmma>();
            }

            if(i_k + 1 == nk)
            {
                if(i_q + 1 < num_q_tiles)
                {
                    // the next Q into registers, once it (not the transfers behind it) landed
                    // the next Q into registers (issued at an earlier head, or at this one)
                    if(q_pending)
                        s_wait_tensorcnt_barrier<0>();
                    q_tile = load_tile(q_lds_read_window);
                    block_sync_lds<0>(); // every wave read Q: its region takes the O shuffle
                }
                if constexpr(kActFactorOnO)
                {
                    const float f = attention_activation_out_factor<Base::Activation>(scale_s);
                    tile_elementwise_inout([&](auto& o) { o *= f; }, o_acc);
                }
                store_o(o_acc, i_q, static_cast<void*>(o_stage));
                o_pending = true;
                clear_tile(o_acc);
                ++i_q;
            }
            i_k    = i_k + 1 == nk ? 0 : i_k + 1;
            k_slot = next_slot;
            v_slot ^= 1;
        } while(++i_total < num_total);

        s_wait_asynccnt<0>();
    }
};

} // namespace ck_tile
