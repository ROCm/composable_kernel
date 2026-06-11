// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

namespace ck_tile {

struct UnifiedAttentionPipelineDefaultPolicy
{
    static constexpr ck_tile::index_t NumWarpPerGroup = 4;
    static constexpr ck_tile::index_t NumThreadPerWarpGroup =
        NumWarpPerGroup * ck_tile::get_warp_size();

    // TODO: GetAlignment*() currently didn't consider if need padding or not
    //       so in pipeline still need check padding requirement
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    // K/V async-load width selection (returns elements/lane, not bytes).
    //
    // On gfx950 the LDS-targeted async-load instructions (buffer_load_dword_lds
    // / global_load_lds) support exactly three widths: dword (4 B/lane),
    // dwordx3 (12 B/lane), and dwordx4 (16 B/lane). 8 B/lane fails the
    // static_assert in amd_buffer_addressing_builtins. dwordx3 needs the
    // head dim to be a multiple of 12 which never holds for d ∈ {64, 128},
    // so the practical choices are 16 B/lane and 4 B/lane.
    //
    // We pick the widest width such that NumIssues >= 1 on the actual tile:
    //
    //   NumIssues = (kPageBlockSize * kHeadDim) / (kBlockSize * KVector_elems)
    //
    // For BF16/FP16 the historical blanket 16 B/lane always satisfies this
    // for every variant we compile, so the BF16/FP16 path is unchanged.
    //
    // For FP8/BF8 the blanket 4 B/lane was set defensively because the
    // 8-warp prefill variants (kBlockSize = 512) push NumIssues to 0.5 at
    // 16 B/lane. But the 1/2/4-warp decode variants all tile cleanly at
    // 16 B/lane — verified at compile-time for decode_d{64,128}_m{16,32,
    // 64,128} (5 of 7 decode tiers can use dwordx4). Forcing 4 B/lane
    // for those decode tiers doubles the async-load issue count for the
    // same byte volume and is the dominant cause of the
    // FP8-slower-than-BF16 regression observed on long-context decode
    // (e.g. b=128 sq=1 sk=128000 d=64: FP8 SQ_INSTS_VMEM 131M vs
    // BF16 65M, GRBM_GUI_ACTIVE 144M vs 116M; see ua-test-scripts/
    // rocprof_analysis/BOTTLENECK_ANALYSIS.md for the full PMC table).
    //
    // The selector below picks dwordx4 whenever it tiles cleanly and falls
    // back to dword (matches the historical FP8 path) on the prefill tier.
    template <typename Problem,
              index_t ElementSizeInBytes,
              index_t NumLoadThreads = Problem::kBlockSize>
    CK_TILE_DEVICE static constexpr index_t GetKVAlignmentBytes()
    {
#if defined(__gfx950__)
        // dwordx4 = 16 B/lane; tile must yield NumIssues >= 1, integer.
        constexpr index_t tile_elems =
            Problem::UnifiedAttentionShape::kPageBlockSize *
            Problem::UnifiedAttentionShape::kHeadDim;
        // Threads that actually cooperate on this tile's load. Default is the
        // whole block (kBlockSize), but the FA4 per-warp-group decoupling has a
        // single 4-warp group fill the tile by itself, so the width budget is
        // that group's thread count -- which lets the small FP8 prefill tile
        // (4 KB / 256 thr = 16 B/thr) finally tile cleanly at dwordx4 instead of
        // falling back to 4x as many dword loads.
        constexpr index_t block_size = NumLoadThreads;
        // KVector_elems for 16 B/lane = 16 / ElementSizeInBytes.
        // NumIssues * KVector_bytes * kBlockSize == tile_bytes,
        // so the divisibility check is tile_elems * ElementSizeInBytes
        // == multiple of (kBlockSize * 16). Equivalent (since both sides
        // share an ElementSizeInBytes factor when KVector_elems is a power
        // of two) to checking tile_elems is a multiple of (kBlockSize *
        // 16 / ElementSizeInBytes), and tile_elems * elem_bytes >=
        // kBlockSize * 16. Just check the byte form directly:
        constexpr index_t tile_bytes  = tile_elems * ElementSizeInBytes;
        constexpr index_t wide_bytes  = block_size * 16;  // dwordx4 needs this much
        if constexpr (tile_bytes >= wide_bytes && (tile_bytes % wide_bytes) == 0)
            return 16;  // dwordx4
        else
            return 4;   // dword (fallback; matches the historical FP8 path)
#else
        return 4;
#endif
    }

    // NumWarps = the waves that cooperate on the K/V load (default = the full
    // block). The FA4 decoupling loads K/V with a single 4-warp group, so the
    // load-path callers pass that group's warp count to widen the load.
    template <typename Problem,
              ck_tile::index_t NumWarps = Problem::UnifiedAttentionShape::NumWarps>
    CK_TILE_DEVICE static constexpr auto GetAlignmentK()
    {
        using namespace ck_tile;
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        constexpr index_t NumLoadThreads = NumWarps * get_warp_size();
        constexpr index_t MaxReadSizeInBytes =
            GetKVAlignmentBytes<Problem, sizeof(KDataType), NumLoadThreads>();
        return MaxReadSizeInBytes / sizeof(KDataType);
    }

    template <typename Problem,
              ck_tile::index_t NumWarps = Problem::UnifiedAttentionShape::NumWarps>
    CK_TILE_DEVICE static constexpr auto GetAlignmentV()
    {
        using namespace ck_tile;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t NumLoadThreads = NumWarps * get_warp_size();
        constexpr index_t MaxReadSizeInBytes =
            GetKVAlignmentBytes<Problem, sizeof(VDataType), NumLoadThreads>();
        return MaxReadSizeInBytes / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        using namespace ck_tile;

        // TODO: this is for 3d layout
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemVPackK()
    {
        using namespace ck_tile;

        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    // NumWarpsOverride mirrors MakeVDramTileDistribution: the FA4 "WG1 loads K"
    // path passes NumThreadPerWarpGroup/WarpSize so warp group 1's waves alone
    // tile the full K buffer (the partner group reads it from shared LDS).
    template <typename Problem,
              ck_tile::index_t NumWarpsOverride = Problem::UnifiedAttentionShape::NumWarps>
    CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using namespace ck_tile;

        constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
        constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = NumWarpsOverride;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        constexpr index_t KVector = GetAlignmentK<Problem, NumWarpsOverride>(); // this is for global load

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = WarpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (NumWarps * WarpSize * KVector));
        static_cast<void>(kBlockSize);

        constexpr index_t N0 = NumIssues;
        constexpr index_t N1 = LaneGroups;
        constexpr index_t N2 = NumWarps;
        constexpr index_t K0 = LanesPerK;
        constexpr index_t K1 = KVector;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<2>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // NumWarpsOverride lets the FA4 per-warp-group ("private V") path request a
    // distribution where only NumWarps waves cooperate on the load (so each
    // warp group loads the FULL V tile by itself, into its own LDS buffer, and
    // its own vmcnt proves residency without waiting on the partner group).
    // Default = the shape's NumWarps (the original block-cooperative load).
    template <typename Problem,
              ck_tile::index_t NumWarpsOverride = Problem::UnifiedAttentionShape::NumWarps>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using namespace ck_tile;

        constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
        constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = NumWarpsOverride;
        constexpr index_t WarpSize   = ck_tile::get_warp_size(); // 64

        constexpr index_t KVector = GetAlignmentV<Problem, NumWarpsOverride>(); // this is for global load
        // 4

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = WarpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        // NumWarps-relative form (NumWarps may be < the full block when the FA4
        // per-warp-group path requests a private-V distribution).
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (NumWarps * WarpSize * KVector));
        static_cast<void>(kBlockSize);

        constexpr index_t N0 = NumIssues;  // 8
        constexpr index_t N1 = LaneGroups; // 2
        constexpr index_t N2 = NumWarps;   // 8
        constexpr index_t K0 = LanesPerK;  // 32
        constexpr index_t K1 = KVector;    // 4

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<2>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakePRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;

        return make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::UnifiedAttentionShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::UnifiedAttentionShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
        constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto v_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = ck_tile::detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        // compute the endcoding before transpose
        constexpr auto v_block_dstr =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(v_block_dstr_encode),
                                          typename Problem::VDataType>::TransposedDstrEncode{});

        return v_block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using namespace ck_tile;

        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::UnifiedAttentionShape::kBlockM,
                                                    Problem::UnifiedAttentionShape::kPageBlockSize,
                                                    Problem::UnifiedAttentionShape::kHeadDim>,
                                           typename Problem::UnifiedAttentionShape::Gemm0BlockWarps,
                                           typename Problem::UnifiedAttentionShape::Gemm0WarpTile>>;

        using WarpGemm =
            WarpGemmDispatcher<typename Problem::QDataType,
                               typename Problem::KDataType,
                               typename Problem::SaccDataType,
                               Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<0>{}),
                               Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<1>{}),
                               Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<2>{}),
                               true,
                               false,
                               false>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV2CustomPolicy<
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::SaccDataType,
            typename Problem::UnifiedAttentionShape::Gemm0BlockWarps,
            WarpGemm,
            GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPVBlockGemm()
    {
        using namespace ck_tile;

        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::UnifiedAttentionShape::kBlockM,
                                                    Problem::UnifiedAttentionShape::kHeadDim,
                                                    Problem::UnifiedAttentionShape::kPageBlockSize>,
                                           typename Problem::UnifiedAttentionShape::Gemm1BlockWarps,
                                           typename Problem::UnifiedAttentionShape::Gemm1WarpTile>>;
        // `load_tile_transpose` is only valid when the tile distribution's inner
        // packing matches the transpose engine's SubtileMinorDimension =
        // 64 bits / sizeof(VDataType_in_bits). The PV warp gemm produces
        // kABKPerLane = WG::K / lanes_in_K elements per lane on the K direction
        // (lanes_in_K = 4 for 16x16x* MFMA, 2 for 32x32x*), so we must pick
        // AttrNumAccess such that kABKPerLane / AttrNumAccess == SubMinDim:
        //   bf16/fp16 16x16x32 -> kABKPerLane=8, SubMinDim=4 -> Double.
        //   bf16/fp16 16x16x16 -> kABKPerLane=4, SubMinDim=4 -> Single.
        //   bf16/fp16 32x32x16 -> kABKPerLane=8, SubMinDim=4 -> Double.
        //   fp8/bf8   16x16x32 -> kABKPerLane=8, SubMinDim=8 -> Single.
        //   fp8/bf8   32x32x16 -> kABKPerLane=8, SubMinDim=8 -> Single.
        //   fp8/bf8   32x32x64 -> kABKPerLane=32, SubMinDim=8 -> Quad.
        // The select (ratio = kABKPerLane / SubMinDim) is a compile-time alias.
        static constexpr index_t kPVWarpGemmM =
            Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<0>{});
        static constexpr index_t kPVWarpGemmK =
            Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<2>{});
        static constexpr index_t kPVLanesInK = (kPVWarpGemmM == 16) ? 4 : 2;
        static constexpr index_t kPVABKPerLane = kPVWarpGemmK / kPVLanesInK;
        static constexpr index_t kPVSubMinDim = 8 / sizeof(typename Problem::VDataType);
        static constexpr index_t kPVNumAccessRatio = kPVABKPerLane / kPVSubMinDim;
        static constexpr WGAttrNumAccessEnum PVAttrNumAccess =
            (kPVNumAccessRatio <= 1)   ? WGAttrNumAccessEnum::Single
            : (kPVNumAccessRatio == 2) ? WGAttrNumAccessEnum::Double
                                       : WGAttrNumAccessEnum::Quad;
        using WarpGemm =
            WarpGemmDispatcher<typename Problem::PDataType,
                               typename Problem::VDataType,
                               typename Problem::OaccDataType,
                               Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<0>{}),
                               Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<1>{}),
                               Problem::UnifiedAttentionShape::Gemm1WarpTile::at(number<2>{}),
                               true,
                               false,
                               false,
                               PVAttrNumAccess>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV2CustomPolicy<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            typename Problem::UnifiedAttentionShape::Gemm1BlockWarps,
            WarpGemm,
            GemmLoopOrder::MNK>;
        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    static constexpr ck_tile::index_t kKLdsPadInBytes = 4 * 4;  // 4 dwords
    static constexpr ck_tile::index_t kVLdsPadInBytes = 4 * 16; // 16 dwords

    // WarpIdShift handles a sub-block load issued by a NON-zero warp group via
    // the raw async path. The raw store derives its LDS offset as
    //   M0 = base + size_per_wave * get_warp_id()   (ABSOLUTE warp id 0..7)
    // so a NumWarps-wide (e.g. 4-wave) layout only tiles correctly for warp ids
    // 0..NumWarps-1. When warp group g (>0) alone fills the tile, its waves have
    // absolute ids [g*NumWarps, (g+1)*NumWarps); shifting the descriptor base by
    // -WarpIdShift*size_per_wave (WarpIdShift = g*NumWarps) maps them back to
    // effective ids 0..NumWarps-1, i.e. the exact physical layout a warp-group-0
    // load would produce -- so the (unshifted) read descriptor reads it directly.
    template <typename Problem,
              ck_tile::index_t NumWarpsOverride = Problem::UnifiedAttentionShape::NumWarps,
              ck_tile::index_t WarpIdShift      = 0,
              ck_tile::index_t IBuf             = 0>
    CK_TILE_DEVICE static constexpr auto
    MakeKLdsStoreBlockDescriptor(ck_tile::number<IBuf> = ck_tile::number<0>{})
    {
        using namespace ck_tile;

        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
        constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = NumWarpsOverride;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        [[maybe_unused]] constexpr index_t KPack = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem, NumWarpsOverride>(); // this is for global load
        constexpr index_t kPad =
            kKLdsPadInBytes /
            sizeof(typename Problem::KDataType); // for async-copy, this pad is between warps.
                                                 // Optimize this for lds_read speed

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK =
            kKPerBlock / KVector; // how many lane (within a wave) to load K
        constexpr index_t LaneGroups =
            WarpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (NumWarps * WarpSize * KVector));
        static_cast<void>(kBlockSize);

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},  // n0
                       number<LaneGroups>{}, // n1
                       number<NumWarps>{},   // n2
                       number<LanesPerK>{},  // k0
                       number<KVector>{}),   // k1
            make_tuple(number<NumWarps*(WarpSize * KVector + kPad)>{},
                       number<kKPerBlock>{},
                       number<WarpSize * KVector + kPad>{},
                       number<KVector>{},
                       number<1>{}),
            number<IBuf * GetSingleSmemElementSpaceSize<Problem>() -
                   WarpIdShift*(WarpSize * KVector + kPad)>{},
            number<KVector>{},
            number<1>{});

        // TODO this layout is hard coded, and will be used in async copy buffer view load
        // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
        constexpr auto k_lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        return k_lds_block_desc_issues_warps_lanes;
    }

    template <typename Problem,
              ck_tile::index_t NumWarpsOverride = Problem::UnifiedAttentionShape::NumWarps>
    CK_TILE_DEVICE static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        using namespace ck_tile;

        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
        constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = NumWarpsOverride;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem, NumWarpsOverride>(); // this is for global load
        constexpr index_t kPad =
            kKLdsPadInBytes /
            sizeof(typename Problem::KDataType); // for async-copy, this pad is between warps

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = WarpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (NumWarps * WarpSize * KVector));
        static_cast<void>(kBlockSize);

        constexpr auto k_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumIssues>{},          // n0
                                                    number<NumWarps>{},           // n2
                                                    number<LaneGroups>{},         // n1
                                                    number<kKPerBlock / KPack>{}, // k0
                                                    number<KPack>{}),             // k1
                                         make_tuple(number<NumWarps*(WarpSize * KVector + kPad)>{},
                                                    number<WarpSize * KVector + kPad>{},
                                                    number<kKPerBlock>{},
                                                    number<KPack>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<NumIssues>{}, number<LaneGroups>{}, number<NumWarps>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 2, 1>{}, sequence<3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        // this function assume K/V can share smem
        constexpr index_t SingleKSize = [&]() {
            constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
            constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
            constexpr index_t NumWarps   = Problem::UnifiedAttentionShape::NumWarps;
            constexpr index_t WarpSize   = ck_tile::get_warp_size();

            constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
            constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
            constexpr index_t kPad    = KPack;

            static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
            constexpr index_t LanesPerK  = kKPerBlock / KVector;
            constexpr index_t LaneGroups = WarpSize / LanesPerK;
            constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

            return NumIssues * NumWarps * (WarpSize * KVector + kPad);
        }();

        constexpr index_t SingleVSize = [&]() {
            using VDataType                = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t Banks        = 32; // TODO: need change based on arch
            constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
            constexpr index_t kKPack       = GetSmemKPackK<Problem>();
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
            constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        // Lower-bound on the actual MakeVLdsLoadBlockDescriptor element
        // span: it allocates a (NumIssues, LaneGroups, NumWarps, LanesPerK,
        // KVector) buffer with the outermost stride NumWarps * (WarpSize *
        // KVector + kPad). For BF16/FP16 the existing banked-layout
        // SingleVSize above is always larger; for FP8 the small per-lane
        // KVector (4 B = 4 fp8 elements) combined with the byte-fixed
        // kVLdsPadInBytes = 64 makes the V descriptor's element span
        // dominate, so we must include it here or the static_asserts in
        // GetSmemSizeKV fire.
        constexpr index_t VLoadDescSize = [&]() {
            constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
            constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
            constexpr index_t NumWarps   = Problem::UnifiedAttentionShape::NumWarps;
            constexpr index_t WarpSize   = ck_tile::get_warp_size();
            constexpr index_t KVector    = GetAlignmentV<Problem>();
            constexpr index_t kPad =
                kVLdsPadInBytes / sizeof(typename Problem::VDataType);

            static_assert(WarpSize * KVector >= kKPerBlock &&
                          WarpSize * KVector % kKPerBlock == 0);
            constexpr index_t LanesPerK  = kKPerBlock / KVector;
            constexpr index_t LaneGroups = WarpSize / LanesPerK;
            constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

            return NumIssues * NumWarps * (WarpSize * KVector + kPad);
        }();

        return max(max(SingleKSize, SingleVSize), VLoadDescSize);
    }

    // NumWarpsOverride mirrors MakeVDramTileDistribution: the FA4 "WG0 loads V"
    // path passes NumThreadPerWarpGroup/WarpSize (== 4) so warp group 0's waves
    // alone tile the full V buffer. Default = the shape's NumWarps (cooperative).
    template <typename Problem,
              ck_tile::index_t NumWarpsOverride = Problem::UnifiedAttentionShape::NumWarps,
              ck_tile::index_t IBuf             = 0>
    CK_TILE_DEVICE static constexpr auto
    MakeVLdsStoreBlockDescriptor(ck_tile::number<IBuf> = ck_tile::number<0>{})
    {
        using namespace ck_tile;

        /// FIXME: rename the kNPerBlock & kKPerBlock since the kN1 is congtigous dimension
        constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
        constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = NumWarpsOverride;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        [[maybe_unused]] constexpr index_t KPack = GetSmemVPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentV<Problem, NumWarpsOverride>(); // this is for global load
        constexpr index_t kPad =
            kVLdsPadInBytes /
            sizeof(typename Problem::VDataType); // for async-copy, this pad is between warps.
                                                 // Optimize this for lds_read speed

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK =
            kKPerBlock / KVector; // how many lane (within a wave) to load K
        constexpr index_t LaneGroups =
            WarpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (NumWarps * WarpSize * KVector));
        static_cast<void>(kBlockSize);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},  // n0
                       number<LaneGroups>{}, // n1
                       number<NumWarps>{},   // n2
                       number<LanesPerK>{},  // k0
                       number<KVector>{}),   // k1
            make_tuple(number<NumWarps*(WarpSize * KVector + kPad)>{},
                       number<kKPerBlock>{},
                       number<WarpSize * KVector + kPad>{},
                       number<KVector>{},
                       number<1>{}),
            number<(IBuf + 2) * GetSingleSmemElementSpaceSize<Problem>()>{},
            number<KVector>{},
            number<1>{});

        // TODO this layout is hard coded, and will be used in async copy buffer view load
        // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
        constexpr auto v_lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        return v_lds_block_desc_issues_warps_lanes;
    }

    template <typename Problem,
              ck_tile::index_t NumWarpsOverride = Problem::UnifiedAttentionShape::NumWarps>
    CK_TILE_DEVICE static constexpr auto MakeVLdsLoadBlockDescriptor()
    {
        using namespace ck_tile;

        /// FIXME: rename the kNPerBlock & kKPerBlock since the kN1 is congtigous dimension
        constexpr index_t kNPerBlock = Problem::UnifiedAttentionShape::kPageBlockSize;
        constexpr index_t kKPerBlock = Problem::UnifiedAttentionShape::kHeadDim;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = NumWarpsOverride;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemVPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem, NumWarpsOverride>(); // this is for global load
        constexpr index_t kPad =
            kVLdsPadInBytes /
            sizeof(typename Problem::VDataType); // for async-copy, this pad is between warps

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = WarpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (NumWarps * WarpSize * KVector));
        static_cast<void>(kBlockSize);

        constexpr auto v_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumIssues>{},          // n0
                                                    number<NumWarps>{},           // n2
                                                    number<LaneGroups>{},         // n1
                                                    number<kKPerBlock / KPack>{}, // k0
                                                    number<KPack>{}),             // k1
                                         make_tuple(number<NumWarps*(WarpSize * KVector + kPad)>{},
                                                    number<WarpSize * KVector + kPad>{},
                                                    number<kKPerBlock>{},
                                                    number<KPack>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<NumIssues>{}, number<LaneGroups>{}, number<NumWarps>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 2, 1>{}, sequence<3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        using namespace ck_tile;

        static_assert(MakeKLdsLoadBlockDescriptor<Problem>().get_element_space_size() ==
                      MakeKLdsStoreBlockDescriptor<Problem>().get_element_space_size());
        constexpr index_t k_element_space_size =
            MakeKLdsLoadBlockDescriptor<Problem>().get_element_space_size();

        static_assert(MakeVLdsLoadBlockDescriptor<Problem>().get_element_space_size() ==
                      MakeVLdsStoreBlockDescriptor<Problem>().get_element_space_size());
        constexpr index_t v_element_space_size =
            MakeVLdsLoadBlockDescriptor<Problem>().get_element_space_size();

        static_assert(ck_tile::max(k_element_space_size, v_element_space_size) <=
                      GetSingleSmemElementSpaceSize<Problem>());

        /// TODO: override GetSingleSmemElementSpaceSize() to align with MakeKLdsBlockDescriptor() &
        /// MakeVLdsBlockDescriptor()
        static_assert(std::is_same_v<typename Problem::KDataType, typename Problem::VDataType>);
        constexpr index_t kv_element_space_size_in_bytes =
            GetSingleSmemElementSpaceSize<Problem>() * sizeof(typename Problem::KDataType);

        return kv_element_space_size_in_bytes;
    }

    // FA4 "WG0 loads V" prototype: when the block runs as two warp groups, have
    // ONLY warp group 0 (waves 0-3) load the full V tile into the shared V LDS
    // buffer (V's DRAM dist + LDS descriptors use NumThreadPerWarpGroup/WarpSize
    // == 4 waves so WG0 alone fills the tile). WG1 skips the V DRAM load
    // entirely. No 2x DRAM, no extra LDS (V stays a shared 2-buffer). This
    // decouples V's residency from the partner group's cooperative-load shard
    // (WG0's own vmcnt proves the load) so the V LDS read can later move into
    // the SOFTMAX phase. K stays block-cooperative across all 8 waves.
    // Toggle to false to restore the block-cooperative (8-wave) V load.
    static constexpr bool kFA4WG0LoadsV = true;

    // Symmetric K decoupling: warp group 1 (waves 4-7) alone loads the full K
    // tile into the shared K LDS buffer; warp group 0 reads it from shared LDS.
    // Together with kFA4WG0LoadsV this balances DRAM-load work (WG0->V, WG1->K)
    // and lets each group issue only one tile's load/address instructions.
    static constexpr bool kFA4WG1LoadsK = true;

    // Number of waves that cooperate on a V DRAM->LDS load. For the 2-warp-group
    // FA4 path with kFA4WG0LoadsV, this is one warp group's waves (so WG0 alone
    // fills the tile); otherwise it's the full block (original cooperative load).
    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetVLoadNumWarps()
    {
        constexpr ck_tile::index_t NumWarpGroups =
            Problem::kBlockSize / NumThreadPerWarpGroup;
        if constexpr(kFA4WG0LoadsV && NumWarpGroups == 2)
            return NumThreadPerWarpGroup / ck_tile::get_warp_size();
        else
            return Problem::UnifiedAttentionShape::NumWarps;
    }

    // K analogue of GetVLoadNumWarps (warp group 1 alone fills the K tile).
    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetKLoadNumWarps()
    {
        constexpr ck_tile::index_t NumWarpGroups =
            Problem::kBlockSize / NumThreadPerWarpGroup;
        if constexpr(kFA4WG1LoadsK && NumWarpGroups == 2)
            return NumThreadPerWarpGroup / ck_tile::get_warp_size();
        else
            return Problem::UnifiedAttentionShape::NumWarps;
    }

    // Raw-async warp-id shift for the K store (see MakeKLdsStoreBlockDescriptor):
    // K is loaded by warp group 1, whose absolute warp ids start at one warp
    // group's worth of waves, so the store base must shift by that many waves.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetKStoreWarpShift()
    {
        constexpr ck_tile::index_t NumWarpGroups =
            Problem::kBlockSize / NumThreadPerWarpGroup;
        if constexpr(kFA4WG1LoadsK && NumWarpGroups == 2)
            return NumThreadPerWarpGroup / ck_tile::get_warp_size(); // WG1's first abs warp id
        else
            return 0;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return 4 * GetSmemSizeKV<Problem>();
    }
};

struct UnifiedAttentionPipelineDecodePolicy : UnifiedAttentionPipelineDefaultPolicy
{
    static constexpr ck_tile::index_t NumWarpPerGroup = 2;
    static constexpr ck_tile::index_t NumThreadPerWarpGroup =
        NumWarpPerGroup * ck_tile::get_warp_size();
};

struct UnifiedAttentionPipelineTinyDecodePolicy : UnifiedAttentionPipelineDefaultPolicy
{
    static constexpr ck_tile::index_t NumWarpPerGroup = 1;
    static constexpr ck_tile::index_t NumThreadPerWarpGroup =
        NumWarpPerGroup * ck_tile::get_warp_size();
};

} // namespace ck_tile
