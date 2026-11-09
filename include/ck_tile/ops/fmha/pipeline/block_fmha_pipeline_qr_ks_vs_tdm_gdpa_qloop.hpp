// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong_policy.hpp"

namespace ck_tile {

namespace detail {

// qr_tdm_pingpong's arena (K0, K1, S, V0, V1) with Q in a region of its own, so the next Q
// tile's transfer runs under the current tile's gemms; O is staged in Q's region.
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
    static constexpr index_t kArenaBytes =
        integer_least_multiple(kQOffset + max(kQBytes, kOBytes), 1024);
    // three resident blocks per WGP (the pipeline's kBlockPerCu)
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
// stored as its last key tile completes. With few keys per batch (TCX: 256, four key tiles)
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

    static_assert(!Base::kIsSoftmax, "qr_tdm_gdpa_qloop: GDPA only");
    // its own main loop: of qr_tdm_pingpong's schedule only the two V buffers and the read-ahead
    // distance (kReadAheadWmma) carry over
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

    // store_o(o_acc, i_tile, o_lds) stores query tile i_tile (of this workgroup) from o_acc,
    // through o_lds.
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

        TDMConfig tdm_config_q = PingPong::template MakeTdmConfig<
            typename Policy::template LdsPaddingConfigQ<Problem>>();
        TDMConfig tdm_config_k = PingPong::template MakeTdmConfig<
            typename Policy::template LdsPaddingConfigK<Problem>>();
        TDMConfig tdm_config_v = PingPong::template MakeTdmConfig<
            typename Policy::template LdsPaddingConfigV<Problem>>();

        char* const arena = static_cast<char*>(smem_arena);

        // Q: its own region; read whole into registers at a tile's start
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

        // K and V: two buffers each, trip i in buffer i & 1
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {0, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());
        // buffers picked with ?: (as qr_tdm_pingpong): an array indexed at run time would lose
        // the LDS address space, and the K reads would go out as flat loads
        KDataType* const k_buf0 = reinterpret_cast<KDataType*>(arena + Layout::kK0Offset);
        KDataType* const k_buf1 = reinterpret_cast<KDataType*>(arena + Layout::kK1Offset);
        auto k_lds_view         = make_tensor_view<address_space_enum::lds>(
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

        const index_t num_k_tiles = integer_divide_ceil(seqlen_k, kN0);
        const auto q_origin       = q_dram_block_window_tmp.get_window_origin();
        // the trip of a query tile that issues the next Q: two trips before its end, so the
        // last tile's async O copies (from the region Q goes to) drain first, and Q has two
        // trips to land. The workgroups of a CU run in step, and their O stores come at once.
        const index_t q_issue_trip = num_k_tiles >= 3 ? num_k_tiles - 2 : 0;
        const index_t num_total    = num_k_tiles * num_q_tiles;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        // no keys (a batch's empty key set): O is 0
        if(num_k_tiles == 0)
        {
            clear_tile(o_acc);
            for(index_t i_q = 0; i_q < num_q_tiles; ++i_q)
            {
                store_o(o_acc, i_q, static_cast<void*>(arena + Layout::kOOffset));
                s_wait_asynccnt<0>(); // the next tile stages in the same region
            }
            return;
        }

        // prologue: Q_0, K_0, V_0; Q_0 retires first
        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        move_tile_window(q_dram_window, {kM0, 0});
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        v_lds_write_window.set_bottom_tensor_view_data_ptr(v_buf0);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
        s_wait_tensorcnt_barrier<2>();
        auto q_tile = load_tile(q_lds_read_window);

        clear_tile(o_acc);
        index_t i_k     = 0;     // key tile of this trip
        index_t i_q     = 0;     // query tile of this trip
        bool q_after_kv = false; // the last trip issued the next Q behind its K/V
        bool o_pending  = false; // the last trip started async O copies
        index_t i_total = 0;
        do
        {
            const bool even = (i_total & 1) == 0;

            // the last tile's async O copies read the region the next Q goes to (every wave's,
            // retired before the barrier below)
            if(o_pending && i_k == q_issue_trip)
            {
                s_wait_asynccnt<0>();
                o_pending = false;
            }
            // K_i and V_i have landed, and every read of the buffers trip i + 1 writes is
            // done (dscnt 0 retires them). A Q issued behind the last trip's K/V may stay.
            if(q_after_kv)
                s_wait_tensorcnt_barrier<1, 0>();
            else
                s_wait_tensorcnt_barrier<0, 0>();
            q_after_kv = false;

            // the next trip's K and V, then (at a query tile's q_issue_trip) the next Q
            const bool has_next  = i_total + 1 < num_total;
            const index_t next_k = i_k + 1 == num_k_tiles ? 0 : i_k + 1;
            if(has_next)
            {
                move_tile_window(k_dram_window, {(next_k - i_k) * kN0, 0});
                move_tile_window(v_dram_window, PingPong::VSeqOffset((next_k - i_k) * kN0));
                v_lds_write_window.set_bottom_tensor_view_data_ptr(even ? v_buf1 : v_buf0);
                load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
                k_lds_write_window.set_bottom_tensor_view_data_ptr(even ? k_buf1 : k_buf0);
                load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            }
            if(i_k == q_issue_trip && i_q + 1 < num_q_tiles)
            {
                // every wave read Q_i into registers before this trip's barrier
                load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
                move_tile_window(q_dram_window, {kM0, 0});
                q_after_kv = has_next;
            }

            // S = Q K^T over the kK0 slices of K_i
            clear_tile(s_acc);
            k_lds_read_window.set_bottom_tensor_view_data_ptr(even ? k_buf0 : k_buf1);
            auto k_tile = load_tile(k_lds_read_window);
            if constexpr(1 < k0_loops)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    move_tile_window(k_lds_read_window, {0, kK0});
                    auto k_tile_switch = load_tile(k_lds_read_window);
                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_tile);
                    k_tile = k_tile_switch;
                });
                move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});
            }
            gemm_0(s_acc,
                   get_slice_tile(q_tile,
                                  sequence<0, (k0_loops - 1) * kK0>{},
                                  sequence<kM0, k0_loops * kK0>{}),
                   k_tile);
            PingPong::template SchedReadsAheadOfWmma<Gemm0T::MIterPerWarp,
                                                     Gemm0T::MIterPerWarp * Gemm0T::NIterPerWarp *
                                                         k0_loops,
                                                     0,
                                                     PingPong::kReadAheadWmma>();

            // keys past seqlen_k (the mask's edge tile, the last when seqlen_k is not a
            // whole number of key tiles) take the activation's masked score (0, which it
            // keeps 0)
            const index_t k_start = i_k * kN0;
            if(mask.IsEdgeTile(q_origin.at(I0), k_start, number<kM0>{}, number<kN0>{}))
            {
                set_tile_if(
                    s_acc, type_convert<SaccDataType>(Base::kMaskedScore), [&](auto tile_idx) {
                        return mask.IsOutOfBound(q_origin.at(I0) + tile_idx.at(I0),
                                                 k_start + tile_idx.at(I1));
                    });
            }

            // P = act(scale_s S)
            const auto s_new = cast_tile<SMPLComputeDataType>(s_acc);
            auto p_compute =
                make_static_distributed_tensor<SMPLComputeDataType>(s_new.get_tile_distribution());
            Base::ApplyActivation(s_new, p_compute, scale_s);
            [[maybe_unused]] int32_t p_scale = 0;
            auto p_tile = Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);

            // O += P V_i
            v_lds_read_window.set_bottom_tensor_view_data_ptr(even ? v_buf0 : v_buf1);
            auto v_tile = PingPong::LoadVTile(v_lds_read_window);
            if constexpr(1 < k1_loops)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    gemm_1(o_acc,
                           get_slice_tile(p_tile,
                                          sequence<0, i_k1 * kK1>{},
                                          sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_tile);
                    move_tile_window(v_lds_read_window, PingPong::VSeqOffset(kK1));
                    v_tile = PingPong::LoadVTile(v_lds_read_window);
                });
                move_tile_window(v_lds_read_window, PingPong::VSeqOffset(-kK1 * (k1_loops - 1)));
            }
            gemm_1(o_acc,
                   get_slice_tile(p_tile,
                                  sequence<0, (k1_loops - 1) * kK1>{},
                                  sequence<kM0, k1_loops * kK1>{}),
                   v_tile);
            PingPong::template SchedReadsAheadOfWmma<Gemm1T::MIterPerWarp,
                                                     Gemm1T::MIterPerWarp * Gemm1T::NIterPerWarp *
                                                         k1_loops,
                                                     0,
                                                     PingPong::kReadAheadWmma>();

            // a query tile's last key tile: its O out, the next Q in
            if(i_k + 1 == num_k_tiles)
            {
                if(i_q + 1 < num_q_tiles)
                {
                    // the next Q into registers; its region then takes the O shuffle. Q went
                    // out before the next trip's V and K (which may stay in flight) unless
                    // this trip's head issued Q behind them.
                    if(has_next && q_issue_trip + 1 < num_k_tiles)
                        s_wait_tensorcnt_barrier<2>();
                    else
                        s_wait_tensorcnt_barrier<0>();
                    q_tile     = load_tile(q_lds_read_window);
                    q_after_kv = false;
                    block_sync_lds<0>();
                }
                store_o(o_acc, i_q, static_cast<void*>(arena + Layout::kOOffset));
                o_pending = true;
                clear_tile(o_acc);
                ++i_q;
            }
            i_k = next_k;
        } while(++i_total < num_total);

        // the async O copies read LDS: done before the workgroup releases it
        s_wait_asynccnt<0>();
    }
};

} // namespace ck_tile
