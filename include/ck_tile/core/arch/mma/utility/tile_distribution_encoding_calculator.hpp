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
          bool UncompressedA      = false>
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

    template <index_t MajorDimSize, index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEnc = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<MajorDimSize>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio * kIter>>,
        tuple<sequence<2, 0, 1>>,
        tuple<sequence<1, 0, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

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
        // TODO: Big kludge: some higher level code can not deal with extra trivial dimensions in
        // the C distribution encoding. In theory this should be fixed there, but in practice the
        // best way to deal with this for now is to provide a simplified C distribution for the
        // cases without blocks.
        if constexpr(MmaOp::kCMBlocks == 1 && MmaOp::kCNBlocks == 1)
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

            return tile_distribution_encoding<sequence<1>,
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
