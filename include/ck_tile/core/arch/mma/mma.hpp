// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include "amdgcn_mma.hpp"
#include "mma_selector.hpp"
#include "mma_transforms.hpp"

#include "mfma/mfma.hpp"
#include "wmma/wmma.hpp"

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
 * @tparam CompilerTarget The compiler target
 * @tparam MmaOp          Backend wrapper class that will perform the mma op (e.g., mfma or wmma)
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
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp =
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
          typename MmaTransformsDefaultSelector<MmaOp, CompilerTarget>::SelectedTransforms>
struct WaveWiseMma
{
    using FragWiseMmaOp = MmaOp;

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
    using AVecType = typename MmaOp::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    // Buffer types for WaveTiles
    using ABufferType = AVecType[FragsM][FragsK];
    using BBufferType = BVecType[FragsN][FragsK];
    using CBufferType = CVecType[FragsM][FragsN];

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

    private:
    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT const& inputBuffer)
    {
        // TODO: Implement formatting logic as needed.
        // This is intended to convert input WaveTiles to the native vector types
        // required by the FragWiseMma operation for iteration
        static_assert(sizeof(DstT) == sizeof(SrcT), "Size mismatch in formatBuffer");
        return reinterpret_cast<DstT const&>(inputBuffer);
    }

    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT& inputBuffer)
    {
        // TODO: Implement formatting logic as needed.
        // This is intended to convert input WaveTiles to the native vector types
        // required by the FragWiseMma operation for iteration
        static_assert(sizeof(DstT) == sizeof(SrcT), "Size mismatch in formatBuffer");
        return reinterpret_cast<DstT&>(inputBuffer);
    }

    /*! @brief Execute Mma in row-major accumulation order.
     * @tparam VecTA The input WaveTile A vector type
     * @tparam VecTB The input WaveTile B vector type
     * @tparam VecTC The input/output WaveTile C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec_col_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        // We implement an example wave-tile pipeline here.
        // First, we apply the necessary transforms to the input fragments,
        // then we convert the result into buffers of native vector formats
        // that we can easily index. Native vector formats are necessary inputs
        // to the given MmaOp exec function.
        auto a_frag = formatBuffer<ABufferType>(ATransform::exec(a));
        auto b_frag = formatBuffer<BBufferType>(BTransform::exec(b));
        auto c_frag = formatBuffer<CBufferType>(CTransform::exec(accum));

        // "Col-major" accumulation over the M-dimension fragments first.
        // Pseudo code here, but we would basically iterate over the fragments in col-major order
        for(uint32_t bn = 0u; bn < FragsN; ++bn)
        {
            for(uint32_t bm = 0u; bm < FragsM; ++bm)
            {
                for(uint32_t bk = 0u; bk < FragsK; ++bk)
                {
                    c_frag[bm][bn] =
                        FragWiseMmaOp::exec(a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn]);
                }
            }
        }

        // Convert native vector results back to the output WaveTile format
        // and then return after we apply the final output transform.
        return DTransform::exec(formatBuffer<std::decay_t<VecTC>>(c_frag));
    }

    /*! @brief Execute Mma in row-major accumulation order.
     * @tparam VecTA The input WaveTile A vector type
     * @tparam VecTB The input WaveTile B vector type
     * @tparam VecTC The input/output WaveTile C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec_row_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        // We implement an example wave-tile pipeline here.
        // First, we apply the necessary transforms to the input WaveTiles,
        // then we convert the result into buffers of native vector formats
        // that we can easily index. Native vector formats are necessary inputs
        // to the given MmaOp exec function.
        auto a_frag = formatBuffer<ABufferType>(ATransform::exec(a));
        auto b_frag = formatBuffer<BBufferType>(BTransform::exec(b));
        auto c_frag = formatBuffer<CBufferType>(CTransform::exec(accum));

        // "Row-major" accumulation over the N-dimension fragments first.
        // Pseudo code here, but we would basically iterate over the fragments in row-major order.
        // We also have to ensure that the incoming vector WaveTiles are converted to native vector
        // types before passing to the FragWiseMma exec function.
        for(uint32_t bm = 0u; bm < FragsM; ++bm)
        {
            for(uint32_t bn = 0u; bn < FragsN; ++bn)
            {
                for(uint32_t bk = 0u; bk < FragsK; ++bk)
                {
                    c_frag[bm][bn] =
                        FragWiseMmaOp::exec(a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn]);
                }
            }
        }

        // Convert native vector results back to the output WaveTile format
        // and then return after we apply the final output transform.
        return DTransform::exec(formatBuffer<std::decay_t<VecTC>>(c_frag));
    }

    public:
    /*! @brief Forward to Mma operation with specified accumulation order.
     * @tparam VecTA The input WaveTile A vector type
     * @tparam VecTB The input WaveTile B vector type
     * @tparam VecTC The input/output WaveTile C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            return exec_row_major(
                std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
        }
        else // if constexpr(AccumPolicy == MmaAccumPolicy::COL_MAJOR)
        {
            return exec_col_major(
                std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
        }
    }
};

} // namespace ck_tile::core::arch::mma
