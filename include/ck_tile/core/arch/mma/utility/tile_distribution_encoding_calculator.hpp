// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"

#include <algorithm>
#include <type_traits>

namespace ck_tile::core::arch::mma {
/**
 * @class TileDistrEncCalc
 * @brief Given an MmaOp and modifiers, provides warp-level tile distribution encodings for mapping
 * ABC matrix fragment coordinates to register coordinates (lane, vector item) and vice versa. Note
 * that in case of compression (sparse intrinsics), we can choose to describe the compressed or
 * uncompressed A matrix, where the former is the default (see UncompressedA). When considering A as
 * compressed, the matrix minor dimension is effectively shrunk by the compression factor. Generally
 * the compressed interpretation is used at the MmaOp level (compression has already taken place),
 * whereas the uncompressed interpretation is used at the WarpGemm / MmaPipeline level (expects to
 * be invoked with uncompressed A matrix, compression handled internally).
 * @tparam MmaOp           Intrinsic (amdgcn_mma).
 * @tparam CTranspose      Whether we are using CTranspose.
 * @tparam SFactor         Swizzle factor: special permutation of the M dimension.
 * @tparam kIter           K composition factor (consecutive intrinsic calls to form larger k dim).
 * @tparam AttrNumAccessAV Requested NumAccess *value* for the A matrix. Must be multiple of
 *                         "fundamental" NumAccess for intrinsic. See details in amdgcn_mma.hpp.
 * @tparam AttrNumAccessBV Requested NumAccess *value* for the B matrix.
 * @tparam UncompressedA   Give an uncompressed (full) layout for A instead. This is used at the
 *                         pipeline level, whereas the MmaOp level deals with pre-compressed A
 *                         matrices.
 */
template <typename MmaOp,
          bool CTranspose         = false,
          index_t SFactor         = 1,
          index_t kIter           = 1,
          index_t AttrNumAccessAV = 1,
          index_t AttrNumAccessBV = 1,
          bool UncompressedA      = false,
          bool UsePackedNumAccess = false>
struct TileDistrEncCalc
{
    private:
    // Silently set NumAccess values to at least the values from the intrinsic. In practice this is
    // only used to turn a requested (1,1) into a (1,2) or (2,1) for a small number of gfx950
    // intrinsics (some mixed precision scale intrinsics and some sparse intrinsics).
    static constexpr index_t NumAccessA = std::max(MmaOp::kAKNumAccess, AttrNumAccessAV);
    static constexpr index_t NumAccessB = std::max(MmaOp::kBKNumAccess, AttrNumAccessBV);

    // We are free to choose any NumAccess value to manipulate the load / store behavior, unless the
    // intrinsic fundamentally requires a base NumAccess factor for the layout to be correct.
    static_assert(NumAccessA % MmaOp::kAKNumAccess == 0, "NumAccessA incompatible with builtin.");
    static_assert(NumAccessB % MmaOp::kBKNumAccess == 0, "NumAccessB incompatible with builtin.");

    static_assert(MmaOp::kABKPerLane % (NumAccessA * MmaOp::kCompressionRatio) == 0);
    static_assert(MmaOp::kABKPerLane % NumAccessB == 0);
    static_assert(MmaOp::kCMNumAccess % SFactor == 0, "kCMNumAccess must be multiple of SFactor");

    // Encoding with Ps2RHssMinor = <1, 0, 0> layout. Lane reads strided K values, i.e. K =
    // {NumAccess, kABKLane, VecPerAccess}
    template <index_t MajorDimSize, index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEncStridedK = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<MajorDimSize>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio * kIter>>,
        tuple<sequence<2, 0, 1>>,
        tuple<sequence<1, 0, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;
    // Encoding without trivial dims (Repeat == 1)
    template <index_t MajorDimSize, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEncStridedKLegacy1 = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MajorDimSize>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio * kIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    // Map-equivalent reshaping of ABWarpDstrEncStridedK that drops the two size-1 dimensions the
    // general encoding carries when Repeat == 1 && NumAccess == 1: the Repeat dim is removed
    // entirely (Rs = <> / NDimR == 0) instead of being threaded through P, and the leading size-1
    // NumAccess dim of H1 is folded away, leaving H1 = <kABKLane, VecPerAccess>. Because only
    // size-1 dims move/drop, this is a PROVABLY IDENTICAL lane/vector register map (traceable
    // through calculate_bottom_index), and it coincides term-for-term with the dense-MFMA legacy
    // WarpGemm A/B tree <<kMNLane>, <kABKLane, kABKPerLane>> -- which is what collapses the
    // thread->data address arithmetic to the fused v_lshl_add_u32 (vs a separate v_add_nc_u32 +
    // v_lshlrev_b32) and 16-bit SDWA index math (vs 32-bit) that the general encoding emits.
    template <index_t MajorDimSize, index_t CompressionRatio = 1>
    using ABWarpDstrEncStridedKLegacy2 =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MajorDimSize>,
                                         sequence<MmaOp::kK / MmaOp::kABKPerLane,
                                                  MmaOp::kABKPerLane / CompressionRatio * kIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 0>>,
                                   sequence<2>,
                                   sequence<1>>;

    // Encoding with Ps2RHssMinor = <0, 0, 0> layout. Lane reads contiguous K values, i.e. K =
    // {kABKLane, NumAccess, VecPerAccess}
    template <index_t MajorDimSize, index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEncContiguousK = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<MajorDimSize>,
              sequence<MmaOp::kK / MmaOp::kABKPerLane,
                       NumAccess,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio * kIter>>,
        tuple<sequence<2, 0, 1>>,
        tuple<sequence<0, 0, 0>>,
        sequence<2, 2>,
        sequence<1, 2>>;

    // Map-equivalent reshaping of ABWarpDstrEncStridedK for the dense, single-block,
    // single-repeat, single-access, uncompressed case. It relocates only size-1 dimensions
    // relative to the general encoding, so it is a PROVABLY IDENTICAL lane/vector register map
    // (verified by tracing calculate_bottom_index): it keeps the (size-1) Repeat dim OUT of the
    // lane (P) decomposition (<2, 1> instead of threading it through P as <2, 0, 1>) and splits
    // H0 from <MajorDimSize> into <1, MajorDimSize>. Correctness therefore rests on equivalence
    // to ABWarpDstrEncStridedK (arch-independent), NOT on reproducing the legacy hand-written
    // tree. It additionally COINCIDES term-for-term with the legacy WarpGemm A/B encoding only
    // in the kAK0PerLane == 1 sub-case (e.g. gfx11/gfx12 16x16x16), which is exactly where
    // matching the legacy merge/unmerge tree collapses the address arithmetic to the fused
    // v_lshl_add_u32 (vs a separate v_add_nc_u32 + v_lshlrev_b32). Guarded to Repeat == 1
    // because the general encoding threads Repeat through the lane mapping; for Repeat > 1 that
    // term is not size-1 and must remain in P.
    template <index_t MajorDimSize, index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEncStridedKLegacy = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<1, MajorDimSize>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio * kIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 2>,
        sequence<0, 0, 2>>;

    // Preserve the legacy gfx11 WMMA tree, including the non-trivial repeat dimension in P and
    // the size-1 block dimension in H0. This is map-equivalent to ABWarpDstrEncStridedK, but its
    // exact merge/unmerge structure is needed to reproduce the legacy address arithmetic.
    template <index_t MajorDimSize, index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEncGfx11WmmaLegacy = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<1, MajorDimSize>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio * kIter>>,
        tuple<sequence<0, 2, 1>>,
        tuple<sequence<0, 1, 1>>,
        sequence<1, 2, 2>,
        sequence<0, 0, 2>>;

    // Select a legacy-matching layout only for the cases where the general strided-K encoding
    // diverges from legacy purely by trivial-dimension placement. For gfx11 WMMA with Repeat == 2,
    // use the specialized gfx11 legacy tree (ABWarpDstrEncGfx11WmmaLegacy). For other WMMA with
    // Repeat == 1 && NumAccess == 1, use the <block, lane> legacy tree
    // (ABWarpDstrEncStridedKLegacy). Every other Repeat == 1 && NumAccess == 1 case uses the
    // simpler no-trivial-dims form (ABWarpDstrEncStridedKLegacy2). Both are provably identical maps
    // to ABWarpDstrEncStridedK (they only relocate/drop size-1 dimensions); all remaining cases
    // (packed, multi-block, Repeat > 1, sub-access) keep the general encoding.
    // kUseLegacyStridedKMfma additionally gates the matching legacy C encoding in
    // get_cwarp_dstr_encoding() below (dense / single-block / uncompressed MFMA only).
    static constexpr bool kUseLegacyStridedKWmma =
        (is_mma_op_wmma_v<MmaOp> && MmaOp::kCMBlocks == 1 && MmaOp::kCNBlocks == 1 &&
         MmaOp::kCompressionRatio == 1);

    static constexpr bool kUseLegacyGfx11Wmma =
        kUseLegacyStridedKWmma &&
        is_target_family_gfx11<typename MmaOpTraits<MmaOp>::CompilerTarget>();

    static constexpr bool kUseLegacyStridedKMfma =
        (is_mma_op_mfma_v<MmaOp> && MmaOp::kCMBlocks == 1 && MmaOp::kCNBlocks == 1 &&
         MmaOp::kCompressionRatio == 1);

    template <index_t MajorDimSize, index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEnc = std::conditional_t<
        (UsePackedNumAccess && NumAccess > 1),
        ABWarpDstrEncContiguousK<MajorDimSize, Repeat, NumAccess, CompressionRatio>,
        std::conditional_t<
            (kUseLegacyGfx11Wmma && Repeat == 2 && CompressionRatio == 1),
            ABWarpDstrEncGfx11WmmaLegacy<MajorDimSize, Repeat, NumAccess, CompressionRatio>,
            std::conditional_t<
                (kUseLegacyStridedKWmma && Repeat == 1 && NumAccess == 1 && CompressionRatio == 1),
                ABWarpDstrEncStridedKLegacy<MajorDimSize, Repeat, NumAccess, CompressionRatio>,
                std::conditional_t<
                    Repeat == 1 && NumAccess == 1,
                    ABWarpDstrEncStridedKLegacy2<MajorDimSize, CompressionRatio>,
                    std::conditional_t<
                        Repeat == 1,
                        ABWarpDstrEncStridedKLegacy1<MajorDimSize, NumAccess, CompressionRatio>,
                        ABWarpDstrEncStridedK<MajorDimSize,
                                              Repeat,
                                              NumAccess,
                                              CompressionRatio>>>>>>;

    // Special A Warp distribution encoding just for swizzle case. This was split out since it
    // specifically deals with the M dimension which would make not sense for B.
    template <index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using AWarpDstrEncSwizzle = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<MmaOp::kCMBlocks * MmaOp::kCMNumAccess / SFactor,
                       MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane,
                       SFactor,
                       MmaOp::kCMPerLane / MmaOp::kCMNumAccess>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio * kIter>>,
        tuple<sequence<2, 0, 1, 1, 1, 1>>,
        tuple<sequence<1, 0, 0, 2, 1, 3>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    static constexpr auto get_cwarp_dstr_encoding()
    {
        if constexpr(kUseLegacyGfx11Wmma && SFactor == 1)
        {
            using MSubDims = sequence<MmaOp::kCMBlocks,
                                      MmaOp::kCMNumAccess,
                                      MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane,
                                      MmaOp::kCMPerLane / MmaOp::kCMNumAccess>;
            using NSubDims = sequence<MmaOp::kCNBlocks, MmaOp::kN / MmaOp::kCNBlocks>;

            using MatDims = std::
                conditional_t<CTranspose, tuple<NSubDims, MSubDims>, tuple<MSubDims, NSubDims>>;
            constexpr int MInx = CTranspose ? 2 : 1;
            constexpr int NInx = CTranspose ? 1 : 2;

            return tile_distribution_encoding<sequence<>,
                                              MatDims,
                                              tuple<sequence<MInx, NInx>>,
                                              tuple<sequence<2, 1>>,
                                              sequence<MInx, MInx>,
                                              sequence<1, 3>>{};
        }
        // TODO: Big kludge: some higher level code can not deal with extra trivial dimensions in
        // the C distribution encoding. In theory this should be fixed there, but in practice the
        // best way to deal with this for now is to provide a simplified C distribution for the
        // cases without blocks.
        else if constexpr(MmaOp::kCMBlocks == 1 && MmaOp::kCNBlocks == 1)
        {
            using MSubDims = sequence<MmaOp::kCMNumAccess / SFactor,
                                      MmaOp::kM / MmaOp::kCMPerLane,
                                      MmaOp::kCMPerLane * SFactor / MmaOp::kCMNumAccess>;
            using NSubDims = sequence<MmaOp::kN>;

            // In case of CTranspose, all we do is swap the M and N dimension.
            using MatDims = std::
                conditional_t<CTranspose, tuple<NSubDims, MSubDims>, tuple<MSubDims, NSubDims>>;
            constexpr int MInx = CTranspose ? 2 : 1;
            constexpr int NInx = CTranspose ? 1 : 2;

            // The general single-block C encoding carries a size-1 Repeat dim (Rs = <1>) that is
            // referenced by neither P (<MInx, NInx>) nor Ys (<MInx, MInx>) -- it is a pure,
            // unreferenced replicate of factor 1. Dropping it (Rs = <>) is therefore a PROVABLY
            // IDENTICAL lane/vector register map (the R space never enters calculate_bottom_index),
            // and it makes this encoding coincide term-for-term with the legacy WarpGemm C encoding
            // (which uses Rs = <>). Gated to kUseLegacyStridedKMfma so only the dense /
            // single-block / uncompressed MFMA case -- the same case that takes the legacy A/B tree
            // above -- is affected; every other case keeps the general Rs = <1> form unchanged.
            using CRepeat = std::conditional_t<kUseLegacyStridedKMfma, sequence<>, sequence<1>>;

            return tile_distribution_encoding<CRepeat,
                                              MatDims,
                                              tuple<sequence<MInx, NInx>>,
                                              tuple<sequence<1, 0>>,
                                              sequence<MInx, MInx>,
                                              sequence<0, 2>>{};
        }
        else
        {
            // We unmerge the M and N dimensions in the same way every time.
            using MSubDims = sequence<MmaOp::kCMBlocks,
                                      MmaOp::kCMNumAccess / SFactor,
                                      MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane,
                                      MmaOp::kCMPerLane * SFactor / MmaOp::kCMNumAccess>;
            using NSubDims = sequence<MmaOp::kCNBlocks, MmaOp::kN / MmaOp::kCNBlocks>;

            // In case of CTranspose, all we do is swap the M and N dimension.
            using MatDims = std::
                conditional_t<CTranspose, tuple<NSubDims, MSubDims>, tuple<MSubDims, NSubDims>>;
            constexpr int MInx = CTranspose ? 2 : 1;
            constexpr int NInx = CTranspose ? 1 : 2;

            // For MFMA intrinsics with blocks, the block dimensions might be in the Lane dim or in
            // the Vec dim, so we get different merge orderings.
            if constexpr(MmaOp::CBlockDimInVecDim)
            {
                return tile_distribution_encoding<sequence<1>,
                                                  MatDims,
                                                  tuple<sequence<MInx, NInx>>,
                                                  tuple<sequence<2, 1>>,
                                                  sequence<MInx, NInx, MInx, MInx>,
                                                  sequence<0, 0, 1, 3>>{};
            }
            else
            {
                return tile_distribution_encoding<sequence<1>,
                                                  MatDims,
                                                  tuple<sequence<MInx, MInx, NInx, NInx>>,
                                                  tuple<sequence<2, 0, 0, 1>>,
                                                  sequence<MInx, MInx>,
                                                  sequence<1, 3>>{};
            }
        }
    }

    static constexpr index_t compressionRatioA = UncompressedA ? 1 : MmaOp::kCompressionRatio;

    using AEnc_ = std::conditional_t<
        (SFactor > 1),
        AWarpDstrEncSwizzle<MmaOp::kARepeat, NumAccessA, compressionRatioA>,
        ABWarpDstrEnc<MmaOp::kM, MmaOp::kARepeat, NumAccessA, compressionRatioA>>;
    using BEnc_ = ABWarpDstrEnc<MmaOp::kN, MmaOp::kBRepeat, NumAccessB>;

    public:
    // When using CTranspose, the A and B matrices are swapped.
    using AWarpDstrEncoding = std::conditional_t<CTranspose, BEnc_, AEnc_>;
    using BWarpDstrEncoding = std::conditional_t<CTranspose, AEnc_, BEnc_>;
    using CWarpDstrEncoding = decltype(get_cwarp_dstr_encoding());

    // Some additional consistency checks
    static_assert(TileDistrEncRegMap<AWarpDstrEncoding>::num_lanes == MmaOp::WaveSize);
    static_assert(TileDistrEncRegMap<BWarpDstrEncoding>::num_lanes == MmaOp::WaveSize);
    static_assert(TileDistrEncRegMap<CWarpDstrEncoding>::num_lanes == MmaOp::WaveSize);

    static_assert(TileDistrEncRegMap<AWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::AVecType>::vector_size * MmaOp::APackedSize *
                      kIter * MmaOp::kCompressionRatio / compressionRatioA);
    static_assert(TileDistrEncRegMap<BWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::BVecType>::vector_size * MmaOp::BPackedSize *
                      kIter);
    static_assert(TileDistrEncRegMap<CWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::CVecType>::vector_size);
};
} // namespace ck_tile::core::arch::mma
