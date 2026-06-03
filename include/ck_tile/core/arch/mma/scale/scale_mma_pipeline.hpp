// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"
#include "ck_tile/core/arch/mma/scale/scale_selector.hpp"
#include "ck_tile/core/arch/mma/scale/scale_transforms.hpp"
#include "ck_tile/core/config.hpp"

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ck_tile::core::arch::mma {

/**
 * @class ScaleMmaPipeline
 * @brief Driver for the wave-tile scale Mma operation. Given a backend MmaOp implementation
 * (e.g., scale MFMA), this class performs fragment-wise (MmaTile) decomposition to
 * matrix-multiply input WaveTiles of (A: WaveTileM x WaveTileK) x (B: WaveTileK x WaveTileN) and
 * accumulates results into output WaveTile (C: WaveTileM x WaveTileN).
 * Like WaveWiseMmaPipeline, this decomposes WaveTile dimensions into fragments and iterates
 * internally over FragsM x FragsN x FragsK, passing per-wave scale factors to each fragment call.
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
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp_ =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp_ = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          WaveTileM,
                                          WaveTileN,
                                          WaveTileK,
                                          CompilerTarget,
                                          MmaOpFamily::SCALE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct ScaleMmaPipeline : public MmaPipelineBase<static_cast<int>(MmaPipelineOptionFlag::NONE), ScaleMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<static_cast<int>(MmaPipelineOptionFlag::NONE), ScaleMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on

    using MmaOp = MmaOp_; // Expose the selected MmaOp

    // Fragment dimensions (from the hardware MmaOp)
    constexpr static uint32_t FragM = MmaOp::kM;
    constexpr static uint32_t FragN = MmaOp::kN;
    constexpr static uint32_t FragK = MmaOp::kK;

    // Fragment counts for decomposition
    constexpr static uint32_t FragsM = WaveTileM / FragM;
    constexpr static uint32_t FragsN = WaveTileN / FragN;
    constexpr static uint32_t FragsK = WaveTileK / FragK;

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
    static_assert(WaveTileM >= FragM, "WaveTileM must be >= FragM");
    static_assert(WaveTileN >= FragN, "WaveTileN must be >= FragN");
    static_assert(WaveTileK >= FragK, "WaveTileK must be >= FragK");
    static_assert(WaveTileM % FragM == 0u, "WaveTileM must be a multiple of FragM");
    static_assert(WaveTileN % FragN == 0u, "WaveTileN must be a multiple of FragN");
    static_assert(WaveTileK % FragK == 0u, "WaveTileK must be a multiple of FragK");

    template <typename VecTA,
              typename VecTB,
              typename VecTC,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static void
    execImpl(std::tuple<VecTA, VecTB, VecTC, ScaleADataType, ScaleBDataType>& vecs)
    {
        auto& [a_frag, b_frag, c_frag, scale_A, scale_B] = vecs;

        if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            for(uint32_t bm = 0u; bm < FragsM; ++bm)
            {
                for(uint32_t bn = 0u; bn < FragsN; ++bn)
                {
                    for(uint32_t bk = 0u; bk < FragsK; ++bk)
                    {
                        c_frag[bm][bn] = MmaOp::exec(
                            a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn], scale_A, scale_B);
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
                            a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn], scale_A, scale_B);
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
