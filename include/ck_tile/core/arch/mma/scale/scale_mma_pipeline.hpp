// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"
#include "ck_tile/core/arch/mma/scale/scale_selector.hpp"
#include "ck_tile/core/arch/mma/scale/scale_transforms.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include <type_traits>

namespace ck_tile::core::arch::mma {

/**
 * @class ScaleMmaPipeline
 * @brief Driver for the wave-tile scale Mma operation. Given a backend MmaOp implementation
 * (e.g., scale MFMA), this class performs fragment-wise (MmaTile) decomposition to
 * matrix-multiply input WaveTiles of (A: WaveTileM x WaveTileK) x (B: WaveTileK x WaveTileN) and
 * accumulates results into output WaveTile (C: WaveTileM x WaveTileN).
 * Like WaveWiseMmaPipeline, this decomposes WaveTile dimensions into fragments and iterates
 * internally over FragsM x FragsN x FragsK, passing per-wave scale factors to each fragment call.
 * @tparam ADataType_      Data type of input WaveTile A
 * @tparam BDataType_      Data type of input WaveTile B
 * @tparam CDataType_      Data type of input/output WaveTile C (accumulator)
 * @tparam WaveTileM       Mma WaveTile M dimension
 * @tparam WaveTileN       Mma WaveTile N dimension
 * @tparam WaveTileK       Mma WaveTile K dimension
 * @tparam AccumPolicy     The fragment order of the accum. registers (row or col major frag order)
 * @tparam CTranspose_     Swaps A and B input vectors and interprets C with transposed layout.
 * @tparam SwizzleFactor   Swizzlefactor for Tile Distribution Encoding calculation.
 * @tparam AttrNumAccessAV Extra unmerge factor for vector dimension for A vec, see amdgcn_mma.hpp.
 * @tparam AttrNumAccessBV Extra unmerge factor for vector dimension for B vec, see amdgcn_mma.hpp.
 * @tparam CompilerTarget  The compiler target
 * @tparam MmaOp_          Backend wrapper class that will perform the mma op
 * @tparam MmaTransforms   The set of transforms to be applied to input/output WaveTiles
 */
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          bool CTranspose_           = false,
          index_t SwizzleFactor      = 1,
          index_t AttrNumAccessAV    = 1,
          index_t AttrNumAccessBV    = AttrNumAccessAV,
          typename CompilerTarget =
              decltype(getCMakeCompilerTarget()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                                  // get_compiler_target(),
          typename MmaOp_ =
              typename MmaDefaultSelector<ADataType_, // TODO: c++20 MmaOpI MmaOp_ = typename
                                                      // MmaDefaultSelector<ADataType,
                                          BDataType_,
                                          CDataType_,
                                          WaveTileM,
                                          WaveTileN,
                                          WaveTileK,
                                          CompilerTarget,
                                          MmaOpFamily::SCALE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct ScaleMmaPipeline : public MmaPipelineBase<ScaleMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose_, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<ScaleMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose_, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on

    using MmaOp                      = MmaOp_; // Expose the selected MmaOp
    static constexpr bool CTranspose = CTranspose_;

    using ADataType = typename MmaOp::ADataType;
    using BDataType = typename MmaOp::BDataType;
    using CDataType = typename MmaOp::CDataType;

    static_assert(!MmaOpTraits<MmaOp>::IsSupported || std::is_same_v<ADataType, ADataType_>);
    static_assert(!MmaOpTraits<MmaOp>::IsSupported || std::is_same_v<BDataType, BDataType_>);
    static_assert(!MmaOpTraits<MmaOp>::IsSupported || std::is_same_v<CDataType, CDataType_>);

    // WaveTile dimensions (Used to be fragment dims but higher level expects these to include k
    // iteration!)
    constexpr static index_t kM = WaveTileM;
    constexpr static index_t kN = WaveTileN;
    constexpr static index_t kK = WaveTileK;

    // Fragment counts for composition
    constexpr static uint32_t FragsM = WaveTileM / MmaOp::kM;
    constexpr static uint32_t FragsN = WaveTileN / MmaOp::kN;
    constexpr static uint32_t FragsK = WaveTileK / MmaOp::kK;

    // No MN composition for now! Only K composition (kIter).
    static_assert(!MmaOpTraits<MmaOp>::IsSupported || FragsM == 1);
    static_assert(!MmaOpTraits<MmaOp>::IsSupported || FragsN == 1);

    // TODO: k iteration with scales makes no sense with current exec implementation...
    static_assert(!MmaOpTraits<MmaOp>::IsSupported || FragsK == 1);

    // K0 or kABKPerLane (plus MmaPipeline k iter!)
    // TODO: Check if this makes sense with numAccess and Compression.
    static constexpr index_t kKPerThread = MmaOp::kABKPerLane * FragsK;

    // These values seem to indicate some sort of canonical "k elements per thread" value before
    // potential further splitting with attrNumAccess. For MFMA it seems to be just kKPerThread, but
    // for WMMA it is meant to be what used to be known as kABK1PerLane. See LayoutFromDataType<>.
    // TODO: Check this in WMMA pipelines / gfx1250.
    static constexpr index_t kAKPack = MmaOp::kABKPerLane * FragsK;
    static constexpr index_t kBKPack = MmaOp::kABKPerLane * FragsK;

    // CK Tile expects this structure with some old-style layout params. Added for compatibility.
    struct WarpGemmAttribute
    {
        struct Impl
        {
            static constexpr index_t kCNLane = MmaOp::kN / MmaOp::kCNBlocks;
            static constexpr index_t kK      = MmaOp::kK;
        };
    };

    // Unsupported MmaOps with nonTrivial AttrNumAccess lead to issues in calculator.
    static constexpr index_t AttrNumAccessAV_support =
        MmaOpTraits<MmaOp>::IsSupported ? AttrNumAccessAV : 1;
    static constexpr index_t AttrNumAccessBV_support =
        MmaOpTraits<MmaOp>::IsSupported ? AttrNumAccessBV : 1;

    // TODO: TileDistrEncCalc only supports K composition (kIter) and always gives post-compression
    // A layout. No Swizzle support yet.
    using EncCalc           = TileDistrEncCalc<MmaOp,
                                               CTranspose,
                                               SwizzleFactor,
                                               FragsK,
                                               AttrNumAccessAV_support,
                                               AttrNumAccessBV_support>;
    using AWarpDstrEncoding = typename EncCalc::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename EncCalc::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename EncCalc::CWarpDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;

    // Full static distributed tensor types including composition. This is the baseline input and
    // output format for all exec and transform functions.
    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;

    // We use these thread_buffer types internally in a number of places, because it allows us to
    // directly select the ext_vectors for individual MmaOp calls.
    using AThreadBufType = thread_buffer<typename MmaOp::AVecType, FragsM * FragsK>;
    using BThreadBufType = thread_buffer<typename MmaOp::BVecType, FragsN * FragsK>;
    using CThreadBufType = thread_buffer<typename MmaOp::CVecType, FragsM * FragsN>;

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    // Sanity checks
    static_assert(WaveTileM >= MmaOp::kM, "WaveTileM must be larger than MmaOp::kM");
    static_assert(WaveTileN >= MmaOp::kN, "WaveTileN must be larger than MmaOp::kN");
    static_assert(WaveTileK >= MmaOp::kK, "WaveTileK must be larger than MmaOp::kK");
    static_assert(WaveTileM % MmaOp::kM == 0u, "WaveTileM must be a multiple of MmaOp::kM");
    static_assert(WaveTileN % MmaOp::kN == 0u, "WaveTileN must be a multiple of MmaOp::kN");
    static_assert(WaveTileK % MmaOp::kK == 0u, "WaveTileK must be a multiple of MmaOp::kK");

    // TODO: Why does this even need to be a template? The types should be known.
    template <typename... Params,
              typename ATensor,
              typename BTensor,
              typename CTensor,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static void
    execImpl(ATensor& a, BTensor& b, CTensor& c, ScaleADataType& scale_A, ScaleBDataType& scale_B)
    {
        static_assert(
            detail::is_similiar_distributed_tensor_v<remove_cvref_t<CTensor>, CWarpTensor> &&
            detail::is_similiar_distributed_tensor_v<remove_cvref_t<ATensor>, AWarpTensor> &&
            detail::is_similiar_distributed_tensor_v<remove_cvref_t<BTensor>, BWarpTensor>);

        auto& a_buf = reinterpret_cast<const AThreadBufType&>(a);
        auto& b_buf = reinterpret_cast<const BThreadBufType&>(b);
        auto& c_buf = reinterpret_cast<CThreadBufType&>(c);

        if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            for(uint32_t bm = 0u; bm < FragsM; ++bm)
            {
                for(uint32_t bn = 0u; bn < FragsN; ++bn)
                {
                    for(uint32_t bk = 0u; bk < FragsK; ++bk)
                    {
                        c_buf.at(bm * FragsN + bn) =
                            MmaOp::template exec<Params...>(a_buf.at(bm * FragsK + bk),
                                                            b_buf.at(bn * FragsK + bk),
                                                            c_buf.at(bm * FragsN + bn),
                                                            scale_A,
                                                            scale_B);
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
                        c_buf.at(bm * FragsN + bn) =
                            MmaOp::template exec<Params...>(a_buf.at(bm * FragsK + bk),
                                                            b_buf.at(bn * FragsK + bk),
                                                            c_buf.at(bm * FragsN + bn),
                                                            scale_A,
                                                            scale_B);
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
