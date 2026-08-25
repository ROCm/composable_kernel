// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_selector.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_transforms.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_selector.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_transforms.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include <type_traits>

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
          bool UsePackedNumAccess    = false,
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
struct WaveWiseMmaPipeline : public MmaPipelineBase<WaveWiseMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose_, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, UsePackedNumAccess, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<WaveWiseMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose_, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, UsePackedNumAccess, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on
    using MmaOp                      = MmaOp_;
    static constexpr bool CTranspose = CTranspose_;

    static_assert(!MmaOpTraits<MmaOp>::IsSupported ||
                  std::is_same_v<typename MmaOp::ADataType, ADataType_>);
    static_assert(!MmaOpTraits<MmaOp>::IsSupported ||
                  std::is_same_v<typename MmaOp::BDataType, BDataType_>);
    static_assert(!MmaOpTraits<MmaOp>::IsSupported ||
                  std::is_same_v<typename MmaOp::CDataType, CDataType_>);

    // In the old WarpGemm system, CTranspose swaps ADataType and BDataType at the Attribute and
    // WarpGemm level, but not at the Impl level.
    using ADataType =
        std::conditional_t<CTranspose, typename MmaOp::BDataType, typename MmaOp::ADataType>;
    using BDataType =
        std::conditional_t<CTranspose, typename MmaOp::ADataType, typename MmaOp::BDataType>;
    using CDataType = typename MmaOp::CDataType;

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
            static constexpr index_t kM = MmaOp::kM;
            static constexpr index_t kN = MmaOp::kN;
            static constexpr index_t kK = MmaOp::kK;

            // M size excluding blocks. Dubious for gfx1250, needs attention.
            static constexpr index_t kAMLane =
                is_target_id_any_of<CompilerTarget, amdgcn_target_id::GFX1250>()
                    ? 16
                    : MmaOp::kM / MmaOp::kCMBlocks;

            // N size excluding blocks.
            static constexpr index_t kBNLane = MmaOp::kN / MmaOp::kCNBlocks;

            // This value is the size of the middle K dimension, i.e. the second-fastest changing K
            // dimension of the layout unmerge operations.
            static constexpr index_t kABKLane = MmaOp::kK / MmaOp::kABKPerLane;

            // Seems like identical definition for MFMA, and does not exist for WMMA.
            static constexpr index_t kABKPerLane = MmaOp::kABKPerLane;

            static constexpr index_t kCMLane     = MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane;
            static constexpr index_t kCNLane     = MmaOp::kN / MmaOp::kCNBlocks;
            static constexpr index_t kCM0PerLane = MmaOp::kCMNumAccess;
            static constexpr index_t kCM1PerLane = MmaOp::kCMPerLane / MmaOp::kCMNumAccess;

            // TODO: This might be wrong for gfx1250 M=32 intrinsics.
            static constexpr index_t kAMBlock = MmaOp::kCMBlocks;
            static constexpr index_t kBNBlock = MmaOp::kCNBlocks;
        };

        // Overall handling of AttrNumAccess in CK Tile is a big mess. This definition will probably
        // work for most MFMA intrinsics but not for WMMA, which in the CK Tile system has a sort of
        // "canonical" unmerge of the K dimension which happens *before* the "true" attrNumAccess.
        // Further complicating factor are packNumAccess, differing A/B numAccess values, and the
        // recent complication of AttrNumAccess by for some reason adding the datatype packedness
        // into it.
        static constexpr index_t AttrNumAccessV = AttrNumAccessAV;
    };

    // Expose kCMLane for some callers (e.g. gemm_quant block policies)
    static constexpr index_t kCMLane = WarpGemmAttribute::Impl::kCMLane;

    // Unsupported MmaOps with nonTrivial AttrNumAccess / Swizzle lead to issues in calculator.
    static constexpr index_t AttrNumAccessAV_support =
        MmaOpTraits<MmaOp>::IsSupported ? AttrNumAccessAV : 1;
    static constexpr index_t AttrNumAccessBV_support =
        MmaOpTraits<MmaOp>::IsSupported ? AttrNumAccessBV : 1;
    static constexpr index_t SwizzleFactor_support =
        MmaOpTraits<MmaOp>::IsSupported ? SwizzleFactor : 1;

    // TODO: TileDistrEncCalc only supports K composition (kIter) and always gives post-compression
    // A layout.
    // NOTE: TileDistrEncCalc swaps the A and B tile distribution encodings internally in case of
    // CTranspose!
    using EncCalc           = TileDistrEncCalc<MmaOp,
                                               CTranspose,
                                               SwizzleFactor_support,
                                               FragsK,
                                               AttrNumAccessAV_support,
                                               AttrNumAccessBV_support,
                                               false,
                                               UsePackedNumAccess>;
    using AWarpDstrEncoding = typename EncCalc::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename EncCalc::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename EncCalc::CWarpDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;

    // Full static distributed tensor types including composition. This is the baseline input and
    // output format for all exec and transform functions.
    // NOTE: ADataType AND AWarpDstr are already swapped here in case of CTranspose!
    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;

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
    // NOTE: Here we have arrived at the Impl level. We know nothing about CTranspose here, we just
    // perform the intrinsic, potentially multiple times for K composition.
    template <typename... Params, typename ATensor, typename BTensor, typename CTensor>
    CK_TILE_DEVICE static void execImpl(const ATensor& a, const BTensor& b, CTensor& c)
    {
        auto& c_buf = c.get_thread_buffer().template get_as<typename MmaOp::CVecType>();

        if constexpr(FragsM == 1 && FragsN == 1)
        {
            // Replicate the legacy WarpGemmImpl::operator() + WarpGemmAttributeMfmaIterateK
            // accumulation pattern so the new framework reproduces the legacy WarpGemm's gfx9
            // assembly on its own. The legacy WarpGemm path is being deprecated, so we cannot
            // route to it; instead we mimic it here. Load the A/B thread buffers into local
            // value copies and accumulate every K fragment into a LOCAL C accumulator, then
            // write it back to the C thread buffer once. Using a local accumulator instead of
            // read-modify-writing c_buf.at(0) through the buffer reference each iteration
            // reproduces the legacy ACC-VGPR allocation (this covers both the single-fragment
            // FragsK == 1 case and the K-composed FragsK > 1 / IterateK case).
            // For some unknown reason the outer lambda with parameters is important to get the
            // same assembly even though it does nothing (a separate function also works).
            using AVec1 = ext_vector_t<ADataType, ATensor::get_thread_buffer_size()>;
            using BVec1 = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;

            const auto a_buf1 = a.get_thread_buffer().template get_as<AVec1>();
            const auto b_buf1 = b.get_thread_buffer().template get_as<BVec1>();

            auto c_vec = c_buf.at(0);
            [](const auto& a_buf2, const auto& b_buf2, auto& c_vec2) {
                if constexpr(FragsK == 1)
                {
                    c_vec2 = MmaOp::template exec<Params...>(
                        a_buf2.template get_as<typename MmaOp::AVecType>().at(0),
                        b_buf2.template get_as<typename MmaOp::BVecType>().at(0),
                        c_vec2);
                }
                else
                {
                    static_for<0, FragsK, 1>{}([&](auto bk) {
                        c_vec2 = MmaOp::template exec<Params...>(
                            a_buf2.template get_as<typename MmaOp::AVecType>().at(bk),
                            b_buf2.template get_as<typename MmaOp::BVecType>().at(bk),
                            c_vec2);
                    });
                }
            }(a_buf1, b_buf1, c_vec);
            c_buf.at(0) = c_vec;
        }
        else
        {
            const auto& a_buf = a.get_thread_buffer().template get_as<typename MmaOp::AVecType>();
            const auto& b_buf = b.get_thread_buffer().template get_as<typename MmaOp::BVecType>();

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
    }
};

} // namespace ck_tile::core::arch::mma
