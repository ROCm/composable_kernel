// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include <cstdint>

namespace ck_tile::core::arch::mma {

namespace sparse::detail {
// TODO: c++20: return MmaPipelineOptionFlags directly
constexpr inline int getPipelineFlags()
{
    return static_cast<int>(MmaPipelineOptionFlag::COMPRESS_A);
}
} // namespace sparse::detail

/**
 * @class SparseMmaPipeline
 * @brief Driver for the wave-tile sparse Mma operation. Given a backend MmaOp implementation
 * (e.g., smfmac), this class performs fragment-wise (MmaTile) decomposition to matrix-multiply
 * input WaveTiles of (A: WaveTileM x WaveTileK) x (B: WaveTileK x WaveTileN) and accumulates
 * results into output WaveTile (C: WaveTileM x WaveTileN).
 * Like WaveWiseMmaPipeline, this decomposes WaveTile dimensions into fragments and iterates
 * internally over FragsM × FragsN × FragsK. The A operand is provided in uncompressed form;
 * 2:4 structured sparsity compression (SparseCompressTransform) is applied.
 * @tparam ADataType      Data type of input WaveTile A
 * @tparam BDataType      Data type of input WaveTile B
 * @tparam CDataType      Data type of input/output WaveTile C (accumulator)
 * @tparam WaveTileM      Mma WaveTile M dimension
 * @tparam WaveTileN      Mma WaveTile N dimension
 * @tparam WaveTileK      Mma WaveTile K dimension
 * @tparam AccumPolicy    The fragment order of the accum. registers (row or col major frag order)
 * @tparam CompilerTarget The compiler target
 * @tparam MmaOp_         Backend wrapper class that will perform the mma op
 * @tparam MmaTransforms  The set of transforms to be applied to input/output WaveTiles
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp_ =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          WaveTileM,
                                          WaveTileN,
                                          WaveTileK,
                                          CompilerTarget,
                                          MmaOpFamily::SPARSE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct SparseMmaPipeline : public MmaPipelineBase<sparse::detail::getPipelineFlags(), SparseMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<sparse::detail::getPipelineFlags(), SparseMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on

    static_assert(!Base::template hasFlag<MmaPipelineOptionFlag::ABSwap>(),
                  "Cannot transpose C in sparse intrinsics.");

    using MmaOp = MmaOp_; // Expose the selected MmaOp

    // Fragment dimensions (from the hardware MmaOp)
    constexpr static uint32_t FragM = MmaOp::kM;
    constexpr static uint32_t FragN = MmaOp::kN;
    constexpr static uint32_t FragK = MmaOp::kK;

    // Fragment counts for decomposition
    constexpr static uint32_t FragsM = WaveTileM / FragM;
    constexpr static uint32_t FragsN = WaveTileN / FragN;
    constexpr static uint32_t FragsK = WaveTileK / FragK;

    // Calculate the uncompressed external A per-fragment vector type
    struct ExternalAVecCalculator
    {
        using AVecTraits               = vector_traits<typename MmaOp::AVecType>;
        static constexpr index_t ASize = AVecTraits::vector_size * MmaOp::kCompressionRatio;
        using AVecType                 = ext_vector_t<typename AVecTraits::scalar_type, ASize>;
    };
    using ExternalAFragVecT = typename ExternalAVecCalculator::AVecType;

    // Scalar type of A
    using AScalarT = typename ExternalAVecCalculator::AVecTraits::scalar_type;

    // Per-fragment sizes
    static constexpr uint32_t ExternalAFragSize = ExternalAVecCalculator::ASize;
    static constexpr uint32_t InternalAFragSize =
        vector_traits<typename MmaOp::AVecType>::vector_size;

    // Full wave-tile sizes (all fragments combined)
    static constexpr uint32_t TotalUncompressedElems = FragsM * FragsK * ExternalAFragSize;
    static constexpr uint32_t TotalCompressedElems =
        TotalUncompressedElems / MmaOp::kCompressionRatio;

    // Variable-length idx type for the whole wave-tile (spans multiple int32_t words if needed)
    static constexpr index_t IdxNumWords = sparse::detail::idx_words_needed<TotalCompressedElems>;
    using IdxType                        = sparse::detail::SparseIdxPack<IdxNumWords>;

    // Per-fragment compressed vector type (for individual MmaOp::exec calls)
    using FragAVecT = typename MmaOp::AVecType;

    // Internal vector types used by the base class formatBuffer.
    // InternalAVecT matches the full compressed wave-tile so the base class can
    // format the SparseCompressTransform result via formatBuffer<InternalAVecT>.
    using InternalAVecT = ext_vector_t<AScalarT, TotalCompressedElems>;
    using InternalBVecT = typename MmaOp::BVecType;
    using InternalCVecT = typename MmaOp::CVecType;

    // Buffer types for WaveTiles (caller-facing).
    // A is a single flat uncompressed vector covering the whole wave-tile.
    // The base class compresses it in one pass via ATransform.
    using AVecType = ext_vector_t<AScalarT, TotalUncompressedElems>;
    using BVecType = InternalBVecT[FragsN][FragsK];
    using CVecType = InternalCVecT[FragsM][FragsN];

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    // Sanity checks
    static_assert(WaveTileM >= FragM, "WaveTileM must be >= FragM");
    static_assert(WaveTileN >= FragN, "WaveTileN must be >= FragN");
    static_assert(WaveTileK >= FragK, "WaveTileK must be >= FragK");
    static_assert(WaveTileM % FragM == 0u, "WaveTileM must be a multiple of FragM");
    static_assert(WaveTileN % FragN == 0u, "WaveTileN must be a multiple of FragN");
    static_assert(WaveTileK % FragK == 0u, "WaveTileK must be a multiple of FragK");

    template <typename ATransformResult, typename BTransformResult, typename CTransformResult>
    CK_TILE_DEVICE static void
    execImpl(std::tuple<ATransformResult, BTransformResult, CTransformResult>& transformedInputs)
    {
        auto& [a, b_frag, c_frag]       = transformedInputs;
        auto& [a_compressed_whole, idx] = a;

        // Validate that the ATransform result and per-fragment reinterpretation are correct
        checkATransformResult<ATransformResult>();

        // Reinterpret the full compressed vector as per-fragment arrays
        auto* a_frags = ck_tile::bit_cast<FragAVecT(*)[FragsK]>(&a_compressed_whole);

        // Accumulation loop with per-fragment idx extraction
        if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            for(uint32_t bm = 0u; bm < FragsM; ++bm)
            {
                for(uint32_t bn = 0u; bn < FragsN; ++bn)
                {
                    for(uint32_t bk = 0u; bk < FragsK; ++bk)
                    {
                        c_frag[bm][bn] = MmaOp::exec(
                            a_frags[bm][bk],
                            b_frag[bn][bk],
                            c_frag[bm][bn],
                            sparse::detail::extract_fragment_idx<InternalAFragSize, FragsK>(
                                idx, bm, bk));
                    }
                }
            }
        }
        else if constexpr(AccumPolicy == MmaAccumPolicy::COL_MAJOR)
        {
            for(uint32_t bn = 0u; bn < FragsN; ++bn)
            {
                for(uint32_t bm = 0u; bm < FragsM; ++bm)
                {
                    for(uint32_t bk = 0u; bk < FragsK; ++bk)
                    {
                        c_frag[bm][bn] = MmaOp::exec(
                            a_frags[bm][bk],
                            b_frag[bn][bk],
                            c_frag[bm][bn],
                            sparse::detail::extract_fragment_idx<InternalAFragSize, FragsK>(
                                idx, bm, bk));
                    }
                }
            }
        }
        else
        {
            static_assert(false, "Invalid accumulation policy");
        }
    }

    private:
    // Compile-time validation of ATransform result and per-fragment reinterpretation.
    // Ensures the compressed vector returned by ATransform::exec can be safely
    // reinterpreted as FragAVecT[FragsM][FragsK] for per-fragment MmaOp dispatch.
    template <typename ATransformResult>
    static constexpr void checkATransformResult()
    {
        using ExternalAvecRef = std::add_lvalue_reference_t<AVecType>;
        static_assert(std::is_same_v<ATransformResult,
                                     decltype(ATransform::exec(std::declval<ExternalAvecRef>()))>,
                      "ATransformResult must match the return type of ATransform::exec");

        using CompressedVecType =
            std::remove_reference_t<std::tuple_element_t<0, ATransformResult>>;
        static_assert(sizeof(CompressedVecType) == sizeof(FragAVecT) * FragsM * FragsK,
                      "Compressed A vector size must equal sizeof(FragAVecT[FragsM][FragsK])");

        static_assert(alignof(CompressedVecType) >= alignof(FragAVecT),
                      "Compressed vector alignment must be >= FragAVecT alignment "
                      "for safe reinterpret_cast to per-fragment array");

        using ActualIdxType = std::tuple_element_t<1, ATransformResult>;
        static_assert(std::is_same_v<ActualIdxType, IdxType>,
                      "Sparsity index type must match SparseIdxPack<IdxNumWords>");
    }
};

} // namespace ck_tile::core::arch::mma
