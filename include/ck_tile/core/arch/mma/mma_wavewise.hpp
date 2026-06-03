// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include "amdgcn_mma.hpp"
#include "mma_pipeline.hpp"
#include "mma_selector.hpp"
#include "mma_transforms.hpp"

#include "mfma/mfma.hpp"
#include "wmma/wmma.hpp"
#include <tuple>

namespace ck_tile::core::arch::mma {

/*! @enum MmaAccumPolicy
 * @brief Accumulation order for Mma decomposition
 */
enum struct MmaAccumPolicy
{
    // Decomposition and accumulation in row-major fragment order
    ROW_MAJOR,
    // Decomposition and accumulation in col-major fragment order
    COL_MAJOR
};

namespace dense::wavewise::detail {
// TODO: c++20: return MmaPipelineOptionFlags directly
template <bool SwapAB>
constexpr inline int getPipelineFlags()
{
    return static_cast<int>(SwapAB ? MmaPipelineOptionFlag::ABSwap : MmaPipelineOptionFlag::NONE);
}
} // namespace dense::wavewise::detail

/**
 * @class Mma
 * @brief Driver for the wave-tile Mma operation. Given a backend MmaOp implementation
 * (e.g., mfma or wmma), this class performs fragment-wise (MmaTile) decomposition to
 * matrix-multiply input WaveTiles of (A: WaveTileM x WaveTileK) x (B: WaveTileK x WaveTileN) and
 * accumulates results into output WaveTile (C: WaveTileM x WaveTileN).
 * @tparam ADataType      Data type of input WaveTile A
 * @tparam BDataType      Data type of input WaveTile B
 * @tparam CDataType      Data type of input/output WaveTile C (accumulator)
 * @tparam WaveTileM      Mma WaveTile M dimension
 * @tparam WaveTileN      Mma WaveTile K dimension
 * @tparam WaveTileK      Mma WaveTile M dimension
 * @tparam AccumPolicy    The fragment order of the accum. registers (row or col major frag order)
 * @tparam SwapAB     Swaps A and B input vectors
 * @tparam CompilerTarget The compiler target
 * @tparam MmaOp_         Backend wrapper class that will perform the mma op (e.g., mfma or wmma)
 * @tparam MmaTransforms  The set of transforms to be applied to input/output WaveTiles
 * @par This is an example of an Mma decomposition driver class that can be used in a wave-tile
 * context. Given a WaveTile size, we can decompose the WaveTile into smaller mma op fragments
 * that are natively supported by the hardware (e.g., mfma or wmma). The class also supports
 * applying transforms to the input/output frags as needed (e.g., layout conversions, data type
 * conversions, etc.). We may also specify the accumulation order (row-major or col-major) for the
 * output WaveTile. This is a powerful example of how to build a flexible and reusable mma driver
 * that can adapt to different hardware capabilities and requirements.
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaOpFamily OpFamily,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          bool SwapAB                = false,
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
                                          OpFamily>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct WaveWiseMmaPipeline : public MmaPipelineBase<dense::wavewise::detail::getPipelineFlags<SwapAB>(),
                                            WaveWiseMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, OpFamily, AccumPolicy, SwapAB, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<dense::wavewise::detail::getPipelineFlags<SwapAB>(),
                                 WaveWiseMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, OpFamily, AccumPolicy, SwapAB, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on
    using MmaOp = MmaOp_;

    // Fragment dimensions
    constexpr static uint32_t FragM = MmaOp::kM;
    constexpr static uint32_t FragN = MmaOp::kN;
    constexpr static uint32_t FragK = MmaOp::kK;

    // Fragment counts for decomposition
    constexpr static uint32_t FragsM = WaveTileM / FragM;
    constexpr static uint32_t FragsN = WaveTileN / FragN;
    constexpr static uint32_t FragsK = WaveTileK / FragK;
    constexpr static uint32_t FragsC = FragsM * FragsN;

    // Vector types for packed registers in each fragment
    using InternalAVecT = typename MmaOp::AVecType;
    using InternalBVecT = typename MmaOp::BVecType;
    using InternalCVecT = typename MmaOp::CVecType;

    // Buffer types for WaveTiles
    using AVecType = InternalAVecT[FragsM][FragsK];
    using BVecType = InternalBVecT[FragsN][FragsK];
    using CVecType = InternalCVecT[FragsM][FragsN];

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    // Sanity checks
    static_assert(WaveTileM >= FragM, "WaveTileM must be larger than FragM");
    static_assert(WaveTileN >= FragN, "WaveTileN must be larger than FragN");
    static_assert(WaveTileK >= FragK, "WaveTileK must be larger than FragK");
    static_assert(WaveTileM % FragM == 0u, "WaveTileM must be a multiple of FragM");
    static_assert(WaveTileN % FragN == 0u, "WaveTileN must be a multiple of FragN");
    static_assert(WaveTileK % FragK == 0u, "WaveTileK must be a multiple of FragK");

    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static void execImpl(std::tuple<VecTA, VecTB, VecTC>& vecs)
    {
        auto& [a_frag, b_frag, c_frag] = vecs;

        if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            // "Row-major" accumulation over the N-dimension fragments first.
            // Pseudo code here, but we would basically iterate over the fragments in row-major
            // order. We also have to ensure that the incoming vector WaveTiles are converted to
            // native vector types before passing to the FragWiseMma exec function.
            for(uint32_t bm = 0u; bm < FragsM; ++bm)
            {
                for(uint32_t bn = 0u; bn < FragsN; ++bn)
                {
                    for(uint32_t bk = 0u; bk < FragsK; ++bk)
                    {
                        c_frag[bm][bn] =
                            MmaOp::exec(a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn]);
                    }
                }
            }
        }
        else if constexpr(AccumPolicy == MmaAccumPolicy::COL_MAJOR)
        {
            // "Col-major" accumulation over the M-dimension fragments first.
            // Pseudo code here, but we would basically iterate over the blocks in col-major order
            for(uint32_t bn = 0u; bn < FragsN; ++bn)
            {
                for(uint32_t bm = 0u; bm < FragsM; ++bm)
                {
                    for(uint32_t bk = 0u; bk < FragsK; ++bk)
                    {
                        c_frag[bm][bn] =
                            MmaOp::exec(a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn]);
                    }
                }
            }
        }
        else
        {
            static_assert(false, "Invalid accumulation policy");
        }
    }
};

} // namespace ck_tile::core::arch::mma
