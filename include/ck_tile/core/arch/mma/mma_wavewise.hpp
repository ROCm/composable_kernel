// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
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

/**
 * @class Mma
 * @brief Driver for the wave-tile Mma operation. Given a backend MmaOp implementation
 * (e.g., mfma or wmma), this class performs fragment-wise (MmaTile) decomposition to
 * matrix-multiply input WaveTiles of (A: WaveTileM x WaveTileK) x (B: WaveTileK x WaveTileN) and
 * accumulates results into output WaveTile (C: WaveTileM x WaveTileN).
 * @tparam ADataType_      Data type of input WaveTile A
 * @tparam BDataType_      Data type of input WaveTile B
 * @tparam CDataType_      Data type of input/output WaveTile C (accumulator)
 * @tparam WaveTileM       Mma WaveTile M dimension
 * @tparam WaveTileN       Mma WaveTile N dimension
 * @tparam WaveTileK       Mma WaveTile K dimension
 * @tparam AccumPolicy     The fragment order of the accum. registers (row or col major frag order)
 * @tparam CTranspose_     Swaps A and B input vectors and interprets C with transposed layout.
 * @tparam SwizzleFactor   SwizzleFactor for Tile Distribution Encoding calculation.
 * @tparam AttrNumAccessAV Extra unmerge factor for vector dimension for A vec, see amdgcn_mma.hpp.
 * @tparam AttrNumAccessBV Extra unmerge factor for vector dimension for B vec, see amdgcn_mma.hpp.
 * @tparam CompilerTarget  The compiler target
 * @tparam MmaOp_          Backend wrapper class that will perform the mma op (e.g., mfma or wmma)
 * @tparam MmaTransforms   The set of transforms to be applied to input/output WaveTiles
 * @par This is an example of an Mma decomposition driver class that can be used in a wave-tile
 * context. Given a WaveTile size, we can decompose the WaveTile into smaller mma op fragments
 * that are natively supported by the hardware (e.g., mfma or wmma). The class also supports
 * applying transforms to the input/output frags as needed (e.g., layout conversions, data type
 * conversions, etc.). We may also specify the accumulation order (row-major or col-major) for the
 * output WaveTile. This is a powerful example of how to build a flexible and reusable mma driver
 * that can adapt to different hardware capabilities and requirements.
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
              typename MmaDefaultSelector<ADataType_, // TODO: c++20 MmaOpI MmaOp = typename
                                                      // MmaDefaultSelector<ADataType_,
                                          BDataType_,
                                          CDataType_,
                                          WaveTileM,
                                          WaveTileN,
                                          WaveTileK,
                                          CompilerTarget,
                                          MmaOpFamily::DENSE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct WaveWiseMmaPipeline : public MmaPipelineBase<WaveWiseMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose_, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<WaveWiseMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose_, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on
    using MmaOp                      = MmaOp_;
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

    // TODO: TileDistrEncCalc only supports K composition (kIter) and always gives post-compression
    // A layout. No Swizzle support yet.
    using EncCalc           = TileDistrEncCalc<MmaOp,
                                               CTranspose,
                                               SwizzleFactor,
                                               FragsK,
                                               AttrNumAccessAV,
                                               AttrNumAccessBV>;
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
    template <typename... Params, typename ATensor, typename BTensor, typename CTensor>
    CK_TILE_DEVICE static void execImpl(ATensor& a, BTensor& b, CTensor& c)
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
                                                            c_buf.at(bm * FragsN + bn));
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
                                                            c_buf.at(bm * FragsN + bn));
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
