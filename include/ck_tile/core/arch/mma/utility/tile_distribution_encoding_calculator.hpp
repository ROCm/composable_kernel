// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"

namespace ck_tile::core::arch::mma {
/**
 * @class TileDistrEncCalc
 * @brief Given an MmaOp and modifiers, provides warp-level tile distribution encodings for mapping
 * ABC matrix fragment coordinates to register coordinates (lane, vector item) and vice versa. Note
 * that in case of compression or packed data types, the matrix minor dimension is effectively
 * shrunk by that factor. This is because tile distribution encodings always describe compressed /
 * packed *Datatype* elements, not logical / mathematical uncompressed *value* elements.
 * @tparam MmaOp          Intrinsic (amdgcn_mma).
 * @tparam CTranspose     Whether we are using CTranspose.
 * @tparam SFactor        Swizzle factor. Not implemented.
 * @tparam AttrNumAccessA Requested NumAccess for the A matrix. Must be multiple of "fundamental"
 *                        NumAccess for intrinsic. See details in amdgcn_mma.hpp.
 * @tparam AttrNumAccessB Requested NumAccess for the B matrix.
 */
template <typename MmaOp,
          bool CTranspose        = false,
          index_t SFactor        = 1,
          index_t AttrNumAccessA = MmaOp::kAKNumAccess,
          index_t AttrNumAccessB = MmaOp::kBKNumAccess>
struct TileDistrEncCalc
{
    private:
    static constexpr index_t NumAccessA = std::max(MmaOp::kAKNumAccess, AttrNumAccessA);
    static constexpr index_t NumAccessB = std::max(MmaOp::kBKNumAccess, AttrNumAccessB);

    // We are free to choose any NumAccess value to manipulate the load / store behavior, unless the
    // intrinsic fundamentally requires a base NumAccess factor for the layout to be correct.
    static_assert(AttrNumAccessA % MmaOp::kAKNumAccess == 0,
                  "Requesting NumAccessA incompatible with builtin.");
    static_assert(AttrNumAccessB % MmaOp::kBKNumAccess == 0,
                  "Requesting NumAccessB incompatible with builtin.");

    static_assert(MmaOp::kABKPerLane %
                      (NumAccessA * MmaOp::kCompressionRatio * MmaOp::APackedSize) ==
                  0);
    static_assert(MmaOp::kABKPerLane % (NumAccessB * MmaOp::BPackedSize) == 0);
    static_assert(SFactor == 1, "Swizzle not implemented yet."); // TODO: Implement Swizzle.

    template <index_t MajorDimSize,
              index_t Repeat,
              index_t NumAccess,
              index_t PackedSize       = 1,
              index_t CompressionRatio = 1>
    using ABWarpDstrEnc = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<MajorDimSize>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio / PackedSize>>,
        tuple<sequence<0, 2, 1>>,
        tuple<sequence<0, 1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    static constexpr auto get_cwarp_dstr_encoding()
    {
        // We unmerge the M and N dimensions in the same way every time.
        using MSubDims = sequence<MmaOp::kCMBlocks,
                                  MmaOp::kCMNumAccess,
                                  MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane,
                                  MmaOp::kCMPerLane / MmaOp::kCMNumAccess>;
        using NSubDims = sequence<MmaOp::kCNBlocks, MmaOp::kN / MmaOp::kCNBlocks>;

        // In case of CTranspose, all we do is swap the M and N dimension.
        using MatDims =
            std::conditional_t<CTranspose, tuple<NSubDims, MSubDims>, tuple<MSubDims, NSubDims>>;
        constexpr int MInx = CTranspose ? 2 : 1;
        constexpr int NInx = CTranspose ? 1 : 2;

        // For MFMA intrinsics with blocks, the block dimensions might be in the Lane dim or in the
        // Vec dim, so we get different merge orderings.
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
                                              tuple<sequence<MInx, NInx, MInx, NInx>>,
                                              tuple<sequence<0, 0, 2, 1>>,
                                              sequence<MInx, MInx>,
                                              sequence<1, 3>>{};
        }
    }

    using AEnc_ = ABWarpDstrEnc<MmaOp::kM,
                                MmaOp::kARepeat,
                                NumAccessA,
                                MmaOp::APackedSize,
                                MmaOp::kCompressionRatio>;
    using BEnc_ = ABWarpDstrEnc<MmaOp::kN, MmaOp::kBRepeat, NumAccessB, MmaOp::BPackedSize>;

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
                  vector_traits<typename MmaOp::AVecType>::vector_size);
    static_assert(TileDistrEncRegMap<BWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::BVecType>::vector_size);
    static_assert(TileDistrEncRegMap<CWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::CVecType>::vector_size);
};
} // namespace ck_tile::core::arch::mma
