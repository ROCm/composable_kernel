// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/fmha_bwd_tdm_padding.hpp"

namespace ck_tile {

// The dQ atomic is issued one whole 128 B cache line at a time.
//
// A wave32 wmma C fragment gives lanes 0-15 row r and lanes 16-31 row r+8, each
// 16 columns wide. At fp32 that is two 64 B pieces of two different rows, so an
// unmodified buffer_atomic_add_f32 straddles two cache lines and doubles the dQ
// atomic L2 request count. Two changes fix it, and neither works alone:
//   * gemm_4 packs its N iterations against the warp index rather than across
//     it, so one warp owns 32 adjacent dQ columns instead of two 16-column
//     blocks 64 apart (GetSGradKTBlockGemm, MakeKTRegBlockDescriptor,
//     MakeSGradRegSliceBlockDescriptor);
//   * the pipeline folds each N-adjacent register pair with
//     v_permlane16_swap_b32 and stores through MakeQGradStoreBlockDistribution,
//     which describes the folded layout: one row, 32 columns, 128 B.

// Policy for the bwd pipeline that moves its operands global->LDS with TDM and
// reads them back with ds_load_tr.
//
// Everything not listed here is inherited unchanged; this policy adds the
// operand and dV accumulator descriptors and re-does the smem budget.
struct BlockFmhaBwdPipelineTdmPolicy : BlockFmhaBwdPipelineDefaultPolicy
{
    // Depth of the Q/dO/LSE/D software pipeline, in tiles. 2 issues and waits
    // inside one iteration; 3 leaves one tile in flight across every wait and 4
    // leaves two. The hot loop is unrolled by this factor so the slot index is a
    // compile-time constant, and the wait count is kTdmPerTile * (slots - 2).
    // Each extra slot costs GetQDOSlotStride bytes of LDS.
    static constexpr index_t kQDOSlotsDefault = 3;

    // Masked instances keep the shallow ring: the per-pixel mask VALU already
    // covers the transfer, and on the batch path mirror tile pairing has
    // doubled the body the unroll would duplicate.
    static constexpr index_t kQDOSlotsMasked = 2;
    // Padding, in floats, added to the leading dimension of each accumulator.
    //
    // The gemm C fragment hands lane L the values at rows m = j + 8*(L>>4)
    // (j = 0..7) and column n = L&15, so the two half-waves of a wave32 touch
    // rows that are 8 apart. With an unpadded stride those two rows land on the
    // same LDS banks whenever 8*headdim is a multiple of the 64 banks -- true
    // for every headdim we build (32/64/128/256), giving a 2-way conflict on
    // every access. A 4-float pad makes 8*(headdim+4) mod 64 == 32, which
    // separates the two half-waves by 32 banks, and keeps the row start
    // 16-byte aligned.
    static constexpr index_t kAccLdsPad = 4;

    // Padding, in elements, added to the leading dimension of the four operand
    // boxes that TDM writes and ds_load reads back.
    //
    // At headdim 128 with bf16 a row is 128 * 2 / 4 = 64 dwords, and gfx1250 has
    // 64 LDS banks, so an unpadded row stride puts *every* row on bank 0 and the
    // reader conflicts across the full width. One 128-bit unit of pad per row
    // (8 elements = 16 B = 4 dwords) makes the stride 68 dwords, spreading
    // consecutive rows over 64 / gcd(68, 64) = 16 banks, 2 lanes each -- the
    // floor for a wave32 ds_read_b128, which moves 32 * 4 = 128 dwords -- and a
    // whole number of 16 B units keeps the row start aligned for ds_read_b128 /
    // ds_load_tr16_b128.
    //
    // This is the same failure kAccLdsPad fixes for the accumulators, arrived at
    // from the other direction: there the conflict comes from the C fragment's
    // lane mapping, here from the row stride landing on a multiple of the bank
    // count. The value is derived the way the gemm pipeline derives it (see
    // GetLdsPaddingConfig in gemm_universal_pipeline_ag_bg_cr_policy.hpp).
    //
    // XOR swizzling is not an option for these four: TDM writes a plain box
    // without going through the descriptor, so a reader-side XOR would have
    // nothing to cancel against. dS, which IS written through its descriptor by
    // store_tile, keeps its XOR instead.
    static constexpr index_t kOperandLdsPad = 8;

    // TDM encodes pad_amount as (dwords of padding - 1) and pad_interval as
    // (log2 of the dwords written between pads - 1). One row is exactly one
    // interval, so every row gets kOperandLdsPad elements appended. That must
    // match the descriptor stride: TDM writes the box, the descriptor reads it,
    // and nothing checks the two against each other.
    template <typename T, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto GetOperandLdsPaddingConfig()
    {
        return detail::make_fmha_bwd_tdm_padding_config<T, KPerBlock, kOperandLdsPad>();
    }

    // ---- K: one plain box, K^T read back by ds_load_tr ----------------------
    //
    // One box, read straight for gemm_0 and transposed by ds_load_tr for
    // gemm_4, so no second LDS copy and no register shuffle.
    //
    // The box has to be plain: ds_load_tr reads a hardware-fixed physical
    // pattern, so a descriptor-level XOR has no opportunity to cancel the way it
    // does on the per-element load_tile path. TDM needs the same plain layout.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        using KDataType              = typename Problem::KDataType;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(KDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        return sizeof(typename Problem::KDataType) *
               MakeKLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    }

    // The second K copy is gone, so nothing needs the shuffled staging area.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeKT()
    {
        return 0;
    }

    // gemm_4 with PackMNIter on, which is the block gemm's own supported way of
    // ordering the outer N dimension <NWarp, NIterPerWarp> instead of
    // <NIterPerWarp, NWarp>.
    //
    // Default:  N = n_iter * (NWarp * 16) + w * 16 + nlane
    // Packed:   N = w * (NIterPerWarp * 16) + n_iter * 16 + nlane
    //
    // At hdim 128 (NWarp 4, NIterPerWarp 2) the default hands warp w columns
    // {16w..16w+15} and {64+16w..64+16w+15}: the 128 B dQ line spanning columns
    // [32k, 32k+32) is split across two warps, so no single wave can ever write
    // it whole. Packed, warp w owns [32w, 32w+32) -- exactly one line -- and the
    // two wmma results it holds are the adjacent halves of it.
    //
    // A, B and C all flip together inside the block gemm, so only the two
    // operand descriptors below have to be re-derived.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradKTBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::GemmDataType,
                             typename Problem::KDataType,
                             typename Problem::AccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kQKHeaddim,
                                                    Problem::BlockFmhaShape::kK4>,
                                           typename Problem::BlockFmhaShape::Gemm4BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm4WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::GemmDataType,
                                            typename Problem::KDataType,
                                            typename Problem::AccDataType,
                                            Problem::BlockFmhaShape::Gemm4WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm4WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm4WarpTile::at(number<2>{}),
                                            false>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                typename Problem::KDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm4BlockWarps,
                                                WarpGemm,
                                                1 /*KSubTileNum*/,
                                                true /*PackMNIter*/>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    // The dQ store layout produced by folding the gemm_4 C fragment with
    // v_permlane16_swap_b32; see the dQ atomic note at the top of this file.
    //
    // Before the fold, register (m_iter, n_iter, e) holds
    //     M = (mwarp*MIterPerWarp + m_iter)*16 + mlane*8 + e     mlane = lane>>4
    //     N = (w*NIterPerWarp + n_iter)*16 + (lane & 15)
    // After swapping the n_iter pair (2k, 2k+1), the low register holds row
    // mlane=0 across all 32 lanes and the high register row mlane=1, both over
    // 32 adjacent columns:
    //     M = (mwarp*MIterPerWarp + m_iter)*16 + s*8 + e         s = which of the pair
    //     N = (w*(NIterPerWarp/2) + k)*32 + lane
    // which is what this encoding says. The Y order is (m_iter, k, s, e), i.e.
    // exactly the (m_iter, n_iter, e) order the accumulator already had with
    // n_iter split into (k, s) -- so the fold is in place and no register moves.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQGradStoreBlockDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t MIterPerWarp = Problem::BlockFmhaShape::kM0 / (MWarp * WarpGemm::kM);
        constexpr index_t NIterPerWarp =
            Problem::BlockFmhaShape::kQKHeaddim / (NWarp * WarpGemm::kN);

        constexpr index_t kWarpSize = get_warp_size();
        // Values one lane holds per wmma, and how many lanes the fragment spans
        // down M. For the gfx12 wmma these are 8 and 2: the lane's values are
        // contiguous in M, and the two half-waves are 8 rows apart.
        constexpr index_t kMPerLane = WarpGemm::kM * WarpGemm::kN / kWarpSize;
        constexpr index_t kCMLane   = WarpGemm::kM / kMPerLane;

        static_assert(kCMLane == 2,
                      "the fold assumes the C fragment spans exactly two half-waves down M");
        static_assert(NIterPerWarp % 2 == 0,
                      "the fold needs N-adjacent wmma pairs, so NIterPerWarp must be even");
        static_assert(kWarpSize == kCMLane * (WarpGemm::kN),
                      "the folded row must be exactly one wave wide");

        // Two P dims, like the C tile it replaces: P0 is the warp id and P1 the
        // lane id. This is not cosmetic -- get_partition_index() feeds a
        // single-P distribution get_lane_id() alone, so folding the warp index
        // into one 128-long P dim would land all four warps on the first 32
        // columns.
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<MWarp, MIterPerWarp, kCMLane, kMPerLane>,
                                             sequence<NWarp, NIterPerWarp / 2, kWarpSize>>,
                                       tuple<sequence<1, 2>, sequence<2>>,
                                       tuple<sequence<0, 0>, sequence<2>>,
                                       sequence<1, 2, 1, 1>,
                                       sequence<1, 1, 2, 3>>{});
    }

    // Same encoding the base policy builds for gemm_4's B operand, wrapped so
    // that load_tile_transpose fills it. Identical logical content, different
    // physical arrangement.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTRegBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // PackMNIter ordering -- must stay the exact type MakeBBlockDistribution
        // Encode() returns, the block gemm static_asserts on it.
        constexpr auto kt_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<MWarp>,
            tuple<sequence<NWarp, NIterPerWarp>, sequence<KIterPerWarp>>, // 4 2, 4
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 0>>,
            sequence<1, 2>,
            sequence<1, 0>>{};

        constexpr auto kt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            kt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        auto output =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(kt_block_dstr_encode),
                                          typename Problem::KDataType>::TransposedDstrEncode{});
        return output;
    }

    // K is moved global->LDS by TDM, same three requirements as V: a plain box
    // (above), a trivial tile-major DRAM walk (here) and no writer-side padding.
    // The inherited distribution scatters each row across lanes so load_tile can
    // assemble a register tile; TDM builds no register tile and just needs the
    // box walked in order.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kSeq     = Problem::BlockFmhaShape::kN0;
        constexpr index_t kHeaddim = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t warpNum  = Problem::BlockFmhaShape::NumWarps;

        static_assert(kSeq % warpNum == 0,
                      "K kN0 must be divisible by the warp count for a tile-major K dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kSeq / warpNum>, sequence<kHeaddim>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    // LSE/D are 1-D over seqlen_q and reach LDS by TDM just like K/V/Q/dO.
    // The inherited MakeLSEDDramTileDistribution scatters kM0
    // across the lanes of a warp so load_tile can assemble a register tile; its
    // ys length is therefore kM0/warp_size = 2, which as a TDM box would be 8 B.
    // TDM builds no register tile and just needs the box walked in order, so
    // partition by warp only (IsWarpLevelParallelOnly) and let ys span the whole
    // per-warp run.  This mirrors MakeKDramTileDistribution one rank down.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeLSEDDramTdmDistribution()
    {
        constexpr index_t kSeq    = Problem::BlockFmhaShape::kM0;
        constexpr index_t warpNum = Problem::BlockFmhaShape::NumWarps;

        static_assert(kSeq % warpNum == 0,
                      "LSE/D kM0 must be divisible by the warp count for a tile-major dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kSeq / warpNum>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1>,
                                       sequence<1>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        return GetOperandLdsPaddingConfig<typename Problem::KDataType,
                                          Problem::BlockFmhaShape::kQKHeaddim>();
    }

    // ---- dO: one plain box, dO^T read back by ds_load_tr --------------------
    //
    // Exactly the K story one operand over: dO was materialised twice, the
    // second copy produced by shuffle_tile purely so gemm_1 could read dO^T with
    // a plain load_tile. ds_load_tr16_b128 does it in hardware, so the shuffle,
    // its staging tile and the second copy all go.
    //
    // Unlike K, dO is re-loaded every Q iteration, so the shuffle this removes
    // was in the hot loop.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptor()
    {
        using OGradDataType          = typename Problem::OGradDataType;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(OGradDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad()
    {
        return sizeof(typename Problem::OGradDataType) *
               MakeOGradLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    // the shuffled staging area is gone
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGradT()
    {
        return 0;
    }

    // Same encoding the base policy builds for gemm_1's B operand, wrapped so
    // that load_tile_transpose fills it.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOGradTRegSliceBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        // constexpr index_t kNPerBlock = 32;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto dot_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto dot_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            dot_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});
        // CK_PRINT<typename WarpGemm::BWarpDstrEncoding>();
        // CK_PRINT<decltype(dot_block_dstr_encode)>();

        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                decltype(dot_block_dstr_encode),
                typename Problem::OGradDataType>::TransposedDstrEncode{});
    }

    // ---- Q: one plain box, Q^T read back by ds_load_tr ----------------------
    //
    // Same treatment as K and dO. Q was materialised twice -- once as-is for
    // gemm_0 and once through shuffle_tile into a second LDS copy so gemm_3
    // could read Q^T with a plain load_tile. ds_load_tr16_b128 removes the need
    // for both, and Q's shuffle sits in the Q loop, so it ran once per iteration
    // rather than once per block the way K's did.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        using QDataType              = typename Problem::QDataType;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(QDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        return sizeof(typename Problem::QDataType) *
               MakeQLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    // The shuffled Q copy is gone.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQT()
    {
        return 0;
    }

    // gemm_3's B operand, wrapped so load_tile_transpose fills it -- the base
    // policy returns the same encoding unwrapped, for a plain read off the
    // pre-shuffled copy.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQTRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK3;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto qt_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto qt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            qt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        return make_static_tile_distribution(typename InputTileDistributionTraits<
                                             decltype(qt_block_dstr_encode),
                                             typename Problem::QDataType>::TransposedDstrEncode{});
    }

    // dS box transposed to [kN0][kM0], for THIS pipeline only.
    //
    // v_wmma_f32_16x16x32_bf16 hands each lane 8 C values down a *column* (the M
    // direction). With the box N-contiguous those 8 land in 8 different rows,
    // kN0*sizeof(bf16) = 256 B apart, so the compiler cannot merge them -- 64
    // ds_store_b16/_d16_hi per thread per iteration. M-contiguous instead and the
    // same run collapses to ds_store_b128.
    //
    // This MUST stay an override: the base descriptor is shared by five other bwd
    // pipelines, and only this one reads dS back through load_tile_transpose. A
    // transposed box with a plain reader silently produces wrong dQ -- caught on
    // d=32/64/100/127, which route through the register-resident pipeline.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPack     = GetSmemKPackSGrad<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    // gemm_4's A operand, wrapped so load_tile_transpose fills it -- the same
    // trick the Q^T reader above uses.  Route (b): dS is stored into an
    // M-contiguous [kN0][kM0] box so gemm_2's C fragment (8 values down a
    // column) lands contiguously and the store collapses to ds_store_b128;
    // gemm_4 then reads it back through ds_load_tr16_b128.
    //
    // kKPerBlock is kN0, not kK4: a kK4-wide window cannot be transposed here
    // because kK4 == WarpGemm::kK, so KIterPerWarp collapses to 1 and the
    // encoding comes out a lane-mapping factor short (is_sequence_suffix
    // underflows on <2,1> against the required <2,2,1>).  The whole box is read
    // once and sliced in registers instead, exactly as kt_reg_tensor is.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeSGradRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // PackMNIter ordering. At MWarp == 1 it addresses the same elements as
        // the unpacked <MIterPerWarp, MWarp> spelling, but the block gemm
        // compares encodings by type, so it still has to be spelled packed.
        constexpr auto ds_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MWarp, MIterPerWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{};

        constexpr auto ds_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            ds_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                decltype(ds_block_dstr_encode),
                typename Problem::GemmDataType>::TransposedDstrEncode{});
    }

    // ---- V staged into LDS by TDM -------------------------------------------
    //
    // TDM writes V global -> LDS directly, with no register round trip.
    //
    // The inherited descriptor (MakeXLdsBlockDescriptor) carries an XOR swizzle,
    // which works when store_tile writes it because the reader shares the
    // descriptor and the two XORs cancel. TDM does not go through the descriptor
    // -- it writes a single plain box -- so V gets a plain row-major descriptor
    // instead, the same one the fwd TDM policy uses.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        using VDataType              = typename Problem::VDataType;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(VDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    // V's DRAM-side distribution for TDM. The inherited one is shaped for
    // load_tile, which spreads each row across lanes so a register tile comes
    // out in the gemm's operand order. TDM does not build a register tile at
    // all -- it needs a trivial tile-major walk of the box it is about to write.
    // This mirrors MakeVDramTileDistribution in the fwd TDM policy.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kSeq     = Problem::BlockFmhaShape::kN0;       // V rows
        constexpr index_t kHeaddim = Problem::BlockFmhaShape::kVHeaddim; // V cols
        constexpr index_t warpNum  = Problem::BlockFmhaShape::NumWarps;

        static_assert(kSeq % warpNum == 0,
                      "V kN0 must be divisible by the warp count for a tile-major V dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>, // R: nothing replicated
                                       tuple<sequence<warpNum, kSeq / warpNum>, // X[0]: rows, split
                                                                                // across warps
                                             sequence<kHeaddim>>, // X[1]: one full row per thread
                                       tuple<sequence<1>>,        // PsToRH major
                                       tuple<sequence<0>>,        // PsToRH minor
                                       sequence<1, 2>,            // YsToD major
                                       sequence<1, 0>>{},         // YsToD minor
            bool_constant<true>{});                               // warp-level parallel only
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        return GetOperandLdsPaddingConfig<typename Problem::VDataType,
                                          Problem::BlockFmhaShape::kVHeaddim>();
    }

    // GetSmemSizeV lives in the base and would otherwise call the base's
    // descriptor, not the override above. Same value either way, but keep them
    // from drifting apart.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        return MakeVLdsWriteBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    // The dV accumulator is kN0 x headdim and does not shrink with kM0, so below
    // a kM0 floor the same work pays more LDS round trips than registers save.
    // headdim 256 takes a lower floor because its only tile is kM0 32.
    template <typename Problem>
    static constexpr bool kDVInReg =
        Problem::BlockFmhaShape::kM0 >= (Problem::BlockFmhaShape::kVHeaddim >= 256 ? 32 : 64);

    // dV accumulator: [kN0, kVHeaddim] fp32.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVGradAccLdsBlockDescriptor()
    {
        constexpr index_t kN0       = Problem::BlockFmhaShape::kN0;
        constexpr index_t kVHeaddim = Problem::BlockFmhaShape::kVHeaddim;

        return make_naive_tensor_descriptor(
            make_tuple(number<kN0>{}, number<kVHeaddim>{}),
            make_tuple(number<kVHeaddim + kAccLdsPad>{}, number<1>{}),
            number<1>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeVGradAcc()
    {
        if constexpr(kDVInReg<Problem>)
        {
            return 0;
        }
        else
        {
            return sizeof(typename Problem::AccDataType) *
                   MakeVGradAccLdsBlockDescriptor<Problem>().get_element_space_size();
        }
    }

    // Offset of the accumulator block inside the workgroup's smem.
    //
    // The staged regions (K/KT, V, and the Q/dO/LSE/D/dS set) alias each other
    // and are sized as a max over phases, because each is dead by the time the
    // next phase starts. The accumulators are live for the whole Q loop, so
    // they cannot join that max -- they sit after it, and every existing offset
    // in the pipeline stays exactly as it was.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeStaged()
    {
        // Recomputed from THIS policy's sizes rather than delegating.
        //
        // The base's GetSmemSize resolves GetSmemSizeK/V/Q/OGrad *inside the
        // base class*, so it cannot see the padded descriptors above. Delegating
        // to it would undersize the region and let the padded boxes overrun
        // whatever follows them, with nothing reporting an error. The internal
        // offsets are fine either way -- the pipeline places LSE/D/dS through
        // GetStagedTailOffset, which does use the derived sizes.
        using Base = BlockFmhaBwdPipelineDefaultPolicy;

        constexpr index_t stage0_0 = GetSmemSizeK<Problem>() + GetSmemSizeKT<Problem>();
        constexpr index_t stage0_1 = GetSmemSizeV<Problem>();
        // Q^T and dO^T are read out of the Q and dO boxes, so this policy
        // reports 0 for them and there is nothing to reserve.
        constexpr index_t stage1 = GetSmemSizeQT<Problem>() + GetSmemSizeQ<Problem>() +
                                   GetSmemSizeOGradT<Problem>() + GetSmemSizeOGrad<Problem>() +
                                   Base::template GetSmemSizeLSE<Problem>() +
                                   Base::template GetSmemSizeD<Problem>() +
                                   max(Base::template GetSmemSizeBias<Problem>(),
                                       Base::template GetSmemSizeSGrad<Problem>());

        constexpr index_t total = max(stage0_0, stage0_1, stage1);
        return total;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVGradAccSmemOffset()
    {
        return GetSmemSizeStaged<Problem>();
    }

    // V gets its own region rather than aliasing K/KT.
    //
    // Sharing offset 0 with K/KT would separate them only by time: V could not
    // be written until K and KT had been read back out. That ordering is fatal
    // for TDM -- issuing the load late and waiting on TENSORcnt right afterwards
    // exposes the whole global->LDS latency, and issuing it early lands the V
    // box on top of K. Its own kN0*kVHeaddim*sizeof(V) bytes let the issue sit
    // at the top of the prologue and the wait just before the first read, so the
    // transfer overlaps the K staging that follows it.
    //
    // Cost at headdim 128 is 16 KiB on top of ~102 KiB, well inside the 320 KiB
    // a gfx1250 workgroup may take, and not enough to change occupancy.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVSmemOffset()
    {
        return GetSmemSizeStaged<Problem>() + GetSmemSizeVGradAcc<Problem>();
    }

    // Q and dO DRAM distributions for TDM: trivial tile-major, same reasoning as
    // K and V. The inherited ones scatter each row across lanes so load_tile can
    // assemble a register tile; TDM builds no register tile.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        constexpr index_t kRows   = Problem::BlockFmhaShape::kM0;
        constexpr index_t kCols   = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t warpNum = Problem::BlockFmhaShape::NumWarps;
        static_assert(kRows % warpNum == 0, "kM0 must divide by the warp count");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kRows / warpNum>, sequence<kCols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOGradDramTileDistribution()
    {
        constexpr index_t kRows   = Problem::BlockFmhaShape::kM0;
        constexpr index_t kCols   = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t warpNum = Problem::BlockFmhaShape::NumWarps;
        static_assert(kRows % warpNum == 0, "kM0 must divide by the warp count");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kRows / warpNum>, sequence<kCols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        return GetOperandLdsPaddingConfig<typename Problem::QDataType,
                                          Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigOGrad()
    {
        return GetOperandLdsPaddingConfig<typename Problem::OGradDataType,
                                          Problem::BlockFmhaShape::kVHeaddim>();
    }

    // Double-buffering Q/dO is a trade, not a free win: causal already has the
    // mask VALU covering the transfer, so only the unmasked instance takes it --
    // and only that instance pays the extra LDS.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseQDOPrefetch()
    {
        // Strict superset of `!IsMasking`: masked instances gain the second
        // Q/dO pair too, except storerandval, which already sits at 994-998
        // VGPR and spills into the 1024 ceiling if the buffers are added.
        return !(Problem::FmhaMask::IsMasking && Problem::FmhaDropout::IsStoreRandval);
    }

    // How many Q/dO/LSE/D slots the pipeline rotates through; see
    // kQDOSlotsDefault above.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQDOSlots()
    {
        // Without the prefetch budget there is a single box and the pipeline
        // falls back to issue-and-drain.
        if constexpr(!UseQDOPrefetch<Problem>())
        {
            return 1;
        }
        else
        {
            if constexpr(Problem::FmhaMask::IsMasking)
            {
                return kQDOSlotsMasked;
            }
            else if constexpr(Problem::kQDOSlots != 0)
            {
                // Per-instance override carried by the tile. seqlen_q is a
                // runtime value, so the depth is chosen by dispatching to a
                // separate instance rather than here.
                return Problem::kQDOSlots;
            }
            else
            {
                return kQDOSlotsDefault;
            }
        }
    }

    // One slot is Q + dO + LSE + D. Slot 0 lives inside the staged region; the
    // rest are appended past everything else, so slot 1 lands exactly where the
    // old second Q/dO pair did and the two-slot layout stays byte-identical.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQDOSlotStride()
    {
        return GetSmemSizeQ<Problem>() + GetSmemSizeOGrad<Problem>() + GetSmemSizeLSE<Problem>() +
               GetSmemSizeD<Problem>();
    }

    // Base of slot j, j >= 1. j == 0 is addressed through the staged offsets.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQDOSlotBase(index_t j)
    {
        return GetSmemSizeStaged<Problem>() + GetSmemSizeVGradAcc<Problem>() +
               GetSmemSizeV<Problem>() + (j - 1) * GetQDOSlotStride<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t single =
            GetSmemSizeStaged<Problem>() + GetSmemSizeVGradAcc<Problem>() + GetSmemSizeV<Problem>();
        return single + (GetQDOSlots<Problem>() - 1) * GetQDOSlotStride<Problem>();
    }
};

} // namespace ck_tile
