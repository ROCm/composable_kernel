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
template <bool SwapAB>
constexpr inline int getPipelineFlags()
{
    return static_cast<int>(MmaPipelineOptionFlag::COMPRESS_A) |
           static_cast<int>(SwapAB ? MmaPipelineOptionFlag::ABSwap : MmaPipelineOptionFlag::NONE);
}
} // namespace sparse::detail

/**
 * @class SparseMmaPipeline
 * @brief Driver for the wave-tile sparse Mma operation. Given a backend MmaOp implementation
 * (e.g., smfmac), this class performs fragment-wise (MmaTile) decomposition to matrix-multiply
 * input WaveTiles of (A: WaveTileM x WaveTileK) x (B: WaveTileK x WaveTileN) and accumulates
 * results into output WaveTile (C: WaveTileM x WaveTileN).
 * Like WaveWiseMmaPipeline, this decomposes WaveTile dimensions into fragments and iterates
 * internally over FragsM x FragsN x FragsK. The A operand is provided in uncompressed form;
 * 2:4 structured sparsity compression (SparseCompressTransform) is applied.
 * @tparam ADataType_      Data type of input WaveTile A
 * @tparam BDataType_      Data type of input WaveTile B
 * @tparam CDataType_      Data type of input/output WaveTile C (accumulator)
 * @tparam WaveTileM       Mma WaveTile M dimension
 * @tparam WaveTileN       Mma WaveTile N dimension
 * @tparam WaveTileK       Mma WaveTile K dimension
 * @tparam AccumPolicy     The fragment order of the accum. registers (row or col major frag order)
 * @tparam CTranspose      Swaps A and B input vectors and interprets C with transposed layout.
 * @tparam SwizzleFactor   SwizzleFactor for Tile Distribution Encoding calculation.
 * @tparam AttrNumAccessAV Extra unmerge factor for vector dimension for A vec, see amdgcn_mma.hpp.
 * @tparam AttrNumAccessBV Extra unmerge factor for vector dimension for B vec, see amdgcn_mma.hpp.
 * @tparam CompilerTarget  The compiler target
 * @tparam MmaOp_          Backend class that will perform the mma op (e.g., smfmac or swmmac)
 * @tparam MmaTransforms   The set of transforms to be applied to input/output WaveTiles
 */
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          bool CTranspose            = false,
          index_t SwizzleFactor      = 1,
          index_t AttrNumAccessAV    = 1,
          index_t AttrNumAccessBV    = AttrNumAccessAV,
          typename CompilerTarget =
              decltype(getCMakeCompilerTarget()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                                  // get_compiler_target(),
          typename MmaOp_ =
              typename MmaDefaultSelector<ADataType_, // TODO: c++20 MmaOpI MmaOp = typename
                                                      // MmaDefaultSelector<ADataType,
                                          BDataType_,
                                          CDataType_,
                                          WaveTileM,
                                          WaveTileN,
                                          WaveTileK,
                                          CompilerTarget,
                                          MmaOpFamily::SPARSE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct SparseMmaPipeline : public MmaPipelineBase<sparse::detail::getPipelineFlags<CTranspose>(), SparseMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<sparse::detail::getPipelineFlags<CTranspose>(), SparseMmaPipeline<ADataType_, BDataType_, CDataType_, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CTranspose, SwizzleFactor, AttrNumAccessAV, AttrNumAccessBV, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on
    using MmaOp = MmaOp_;

    using ADataType = typename MmaOp::ADataType;
    using BDataType = typename MmaOp::BDataType;
    using CDataType = typename MmaOp::CDataType;

    static_assert(!MmaOpTraits<MmaOp>::IsSupported || std::is_same_v<ADataType, ADataType_>);
    static_assert(!MmaOpTraits<MmaOp>::IsSupported || std::is_same_v<BDataType, BDataType_>);
    static_assert(!MmaOpTraits<MmaOp>::IsSupported || std::is_same_v<CDataType, CDataType_>);
    static_assert(!(Base::Flags & MmaPipelineOptionFlag::ABSwap),
                  "Cannot transpose C in sparse intrinsics.");

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
    // TODO: TileDistrEncCalc only supports K composition (kIter). Setting UncompressedA to true
    // ensures that we get a tile distribution for the uncompressed A matrix, which is what the
    // higher level caller will show up with (external).
    using EncCalc           = TileDistrEncCalc<MmaOp,
                                               CTranspose,
                                               SwizzleFactor,
                                               FragsK,
                                               AttrNumAccessAV,
                                               AttrNumAccessBV,
                                               true>; // UncompressedA
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
    using BThreadBufType = thread_buffer<typename MmaOp::BVecType, FragsN * FragsK>;
    using CThreadBufType = thread_buffer<typename MmaOp::CVecType, FragsM * FragsN>;

    // Sizes of the A vectors, Fragment-level and WaveTile-level, compressed and uncompressed.
    using FragAVecT                                 = typename MmaOp::AVecType;
    static constexpr index_t InternalAVecSize       = vector_traits<FragAVecT>::vector_size;
    static constexpr index_t ExternalAVecSize       = InternalAVecSize * MmaOp::kCompressionRatio;
    static constexpr index_t TotalUncompressedElems = FragsM * FragsK * ExternalAVecSize;
    static constexpr index_t TotalCompressedElems   = FragsM * FragsK * InternalAVecSize;

    // Variable-length idx type for the whole wave-tile (spans multiple int32_t words if needed)
    static constexpr index_t IdxNumWords = sparse::detail::idx_words_needed<TotalCompressedElems>;
    using IdxType                        = sparse::detail::SparseIdxPack<IdxNumWords>;

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    // Sanity checks
    static_assert(WaveTileM >= MmaOp::kM, "WaveTileM must be >= MmaOp::kM");
    static_assert(WaveTileN >= MmaOp::kN, "WaveTileN must be >= MmaOp::kN");
    static_assert(WaveTileK >= MmaOp::kK, "WaveTileK must be >= MmaOp::kK");
    static_assert(WaveTileM % MmaOp::kM == 0u, "WaveTileM must be a multiple of MmaOp::kM");
    static_assert(WaveTileN % MmaOp::kN == 0u, "WaveTileN must be a multiple of MmaOp::kN");
    static_assert(WaveTileK % MmaOp::kK == 0u, "WaveTileK must be a multiple of MmaOp::kK");

    // ATransformResult is a big ext_vector plus idx, B and C are static_distributed tensors. Fix
    // later TODO.
    template <typename ATransformResult, typename BTensor, typename CTensor>
    CK_TILE_DEVICE static void execImpl(ATransformResult& a, BTensor& b, CTensor& c)
    {
        static_assert(
            detail::is_similiar_distributed_tensor_v<remove_cvref_t<CTensor>, CWarpTensor> &&
            detail::is_similiar_distributed_tensor_v<remove_cvref_t<BTensor>, BWarpTensor>);

        // Validate that the ATransform result and per-fragment reinterpretation are correct
        checkATransformResult<ATransformResult>();

        auto& [a_compressed_whole, idx] = a;

        // Reinterpret the full compressed vector as per-fragment arrays
        auto* a_frags = ck_tile::bit_cast<FragAVecT(*)[FragsK]>(&a_compressed_whole);

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
                        c_buf.at(bm * FragsN + bn) = MmaOp::exec(
                            a_frags[bm][bk],
                            b_buf.at(bn * FragsK + bk),
                            c_buf.at(bm * FragsN + bn),
                            sparse::detail::extract_fragment_idx<InternalAVecSize, FragsK>(
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
                        c_buf.at(bm * FragsN + bn) = MmaOp::exec(
                            a_frags[bm][bk],
                            b_buf.at(bn * FragsK + bk),
                            c_buf.at(bm * FragsN + bn),
                            sparse::detail::extract_fragment_idx<InternalAVecSize, FragsK>(
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
        using AVecType        = ext_vector_t<ADataType, TotalUncompressedElems>;
        using ExternalAvecRef = std::add_lvalue_reference_t<AVecType>;
        static_assert(
            std::is_same_v<ATransformResult,
                           decltype(ATransform::execExtVec(std::declval<ExternalAvecRef>()))>,
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
