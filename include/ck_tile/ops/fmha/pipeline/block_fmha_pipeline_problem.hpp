// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"

namespace ck_tile {

namespace detail {

template <typename DataType, index_t ElemPerThread>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxVectorSize()
{
    if constexpr(std::is_same_v<DataType, half_t> || std::is_same_v<DataType, bf16_t>)
    {
        // ToDo: need support in ck_tile for using buffer_load_dwordx3
        // if constexpr(ElemPerThread % 6 == 0)
        //    return 6;
        if constexpr(ElemPerThread % 8 == 0)
            return 8;
        else if constexpr(ElemPerThread % 4 == 0)
            return 4;
        else if constexpr(ElemPerThread % 2 == 0)
            return 2;
        return 1;
    }
    else if constexpr(std::is_same_v<DataType, float>)
    {
        // ToDo: need support in ck_tile for using buffer_load_dwordx3
        // if constexpr(ElemPerThread % 3 == 0)
        //    return 3;
        if constexpr(ElemPerThread % 4 == 0)
            return 4;
        else if constexpr(ElemPerThread % 2 == 0)
            return 2;
        return 1;
    }
    else
        return 1;
};

template <typename DataType,
          index_t kThreadBlockSize,
          index_t kHigherDimSize,
          index_t kLowerDimSize>
CK_TILE_HOST_DEVICE static constexpr auto GetDramTileAccessMaxVectorSize()
{
    constexpr index_t ElemPerThread = (kHigherDimSize * kLowerDimSize) / kThreadBlockSize;

    return GetMaxVectorSize<DataType, ElemPerThread>();
}

} // namespace detail

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename RandValOutputDataType_,
          typename LSEDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename AttentionVariant_,
          typename FmhaMask_,
          bool kUseTrLoad_,
          typename Traits_>
struct BlockFmhaPipelineProblem
{
    using QDataType             = remove_cvref_t<QDataType_>;
    using KDataType             = remove_cvref_t<KDataType_>;
    using VDataType             = remove_cvref_t<VDataType_>;
    using SaccDataType          = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType   = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType          = remove_cvref_t<BiasDataType_>;
    using RandValOutputDataType = remove_cvref_t<RandValOutputDataType_>;
    using LSEDataType           = remove_cvref_t<LSEDataType_>;
    using PDataType             = remove_cvref_t<PDataType_>;
    using OaccDataType          = remove_cvref_t<OaccDataType_>;
    using ODataType             = remove_cvref_t<ODataType_>;
    using BlockFmhaShape        = remove_cvref_t<BlockFmhaShape_>;
    using AttentionVariant      = remove_cvref_t<AttentionVariant_>;
    using FmhaMask              = remove_cvref_t<FmhaMask_>;
    using Traits                = remove_cvref_t<Traits_>;

    // TODO: Pass scale types and granularity from FmhaFwdTypeConfig
    using QScaleDataType = ck_tile::e8m0_t;
    using KScaleDataType = ck_tile::e8m0_t;
    using VScaleDataType = ck_tile::e8m0_t;
    using PScaleDataType = ck_tile::e8m0_t;

    static constexpr ck_tile::index_t kQKScaleGranularity = 32;
    static constexpr ck_tile::index_t kVScaleGranularity  = 32;

    static constexpr index_t kNumGemm0Warps = BlockFmhaShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = BlockFmhaShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = BlockFmhaShape::NumWarps * get_warp_size();

    static constexpr bool kIsGroupMode = kIsGroupMode_;
    static constexpr bool kUseTrLoad   = kUseTrLoad_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = Traits::kHasLogitsSoftCap;
    static constexpr bool kSkipMinSeqlenQ   = Traits::kSkipMinSeqlenQ;
    static constexpr auto BiasEnum          = Traits::BiasEnum;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kHasDropout       = Traits::kHasDropout;
    static constexpr auto QScaleEnum        = Traits::QScaleEnum;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
    static constexpr bool kHasSink          = Traits::kHasSink;
};

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename RandValOutputDataType_,
          typename LSEDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename AttentionVariant_,
          typename FmhaMask_,
          bool kUseTrLoad_,
          int kPageBlockSize_,
          typename Traits_>
struct BlockFmhaBatchPrefillPipelineProblem
    : public BlockFmhaPipelineProblem<QDataType_,
                                      KDataType_,
                                      VDataType_,
                                      SaccDataType_,
                                      SMPLComputeDataType_,
                                      BiasDataType_,
                                      RandValOutputDataType_,
                                      LSEDataType_,
                                      PDataType_,
                                      OaccDataType_,
                                      ODataType_,
                                      BlockFmhaShape_,
                                      kIsGroupMode_,
                                      AttentionVariant_,
                                      FmhaMask_,
                                      kUseTrLoad_,
                                      Traits_>
{
    static constexpr index_t kPageBlockSize = kPageBlockSize_;
    static_assert(kPageBlockSize > 0, "kPageBlockSize must be positive");
    static_assert((kPageBlockSize & (kPageBlockSize - 1)) == 0,
                  "kPageBlockSize must be power of two");

    // KV cache load addressing mode. GLOBAL_LOAD_LDS handles >2GB pools via
    // 64-bit addressing; BUFFER_LOAD (default) uses SRD buffer_load for the
    // <2GB fast path. The 2GB bound = INT32_MAX byte offset, matching CK's
    // existing TwoGB convention.
    static constexpr auto kKVLoadMode = Traits_::kKVLoadMode;

    static constexpr index_t kVectorSize  = 16 / sizeof(KDataType_); // Dwordx4
    static constexpr auto kKVMemoryLayout = Traits_::kKVMemoryLayout;
    static constexpr auto kKVLookupTable  = Traits_::kKVLookupTable;
    static constexpr bool kIsVectorizedLayout =
        kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;

    static_assert(BlockFmhaShape_::kQKHeaddim % kVectorSize == 0,
                  "kQKHeaddim must be divisible by kVectorSize");
    static_assert(!(kPageBlockSize == 1 && kIsVectorizedLayout),
                  "page_size=1 only supports linear KV cache layout");
    static_assert(!kIsVectorizedLayout || kPageBlockSize % kVectorSize == 0,
                  "kPageBlockSize must be divisible by kVectorSize for vectorized layout");
    static_assert(kIsGroupMode_, "Batch prefill requires group mode");

    static_assert(BlockFmhaShape_::IsVLayoutRowMajor,
                  "Batch prefill kernel requires RowMajor VLayout");
};

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename LSEDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename AttentionVariant_,
          typename FmhaMask_,
          typename Traits_>
struct BlockFmhaFwdPagedKVPipelineProblem
{
    using QDataType           = remove_cvref_t<QDataType_>;
    using KDataType           = remove_cvref_t<KDataType_>;
    using VDataType           = remove_cvref_t<VDataType_>;
    using SaccDataType        = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType        = remove_cvref_t<BiasDataType_>;
    using LSEDataType         = remove_cvref_t<LSEDataType_>;
    using PDataType           = remove_cvref_t<PDataType_>;
    using OaccDataType        = remove_cvref_t<OaccDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockFmhaShape      = remove_cvref_t<BlockFmhaShape_>;
    using AttentionVariant    = remove_cvref_t<AttentionVariant_>;
    using FmhaMask            = remove_cvref_t<FmhaMask_>;
    using Traits              = remove_cvref_t<Traits_>;

    static constexpr index_t kNumGemm0Warps = BlockFmhaShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = BlockFmhaShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = BlockFmhaShape::NumWarps * get_warp_size();

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = Traits::kHasLogitsSoftCap;
    static constexpr bool kSkipMinSeqlenQ   = Traits::kSkipMinSeqlenQ;
    static constexpr auto BiasEnum          = Traits::BiasEnum;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr bool kIsPagedKV        = Traits::kIsPagedKV;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
    static constexpr bool kHasSink          = Traits::kHasSink;
};

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename LSEDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename AttentionVariant_,
          typename FmhaMask_,
          typename Traits_>
struct BlockFmhaFwdSplitKVPipelineProblem
{
    using QDataType           = remove_cvref_t<QDataType_>;
    using KDataType           = remove_cvref_t<KDataType_>;
    using VDataType           = remove_cvref_t<VDataType_>;
    using SaccDataType        = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType        = remove_cvref_t<BiasDataType_>;
    using LSEDataType         = remove_cvref_t<LSEDataType_>;
    using PDataType           = remove_cvref_t<PDataType_>;
    using OaccDataType        = remove_cvref_t<OaccDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockFmhaShape      = remove_cvref_t<BlockFmhaShape_>;
    using AttentionVariant    = remove_cvref_t<AttentionVariant_>;
    using FmhaMask            = remove_cvref_t<FmhaMask_>;
    using Traits              = remove_cvref_t<Traits_>;

    static constexpr index_t kNumGemm0Warps = BlockFmhaShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = BlockFmhaShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = BlockFmhaShape::NumWarps * get_warp_size();

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ                = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK                = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ               = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV               = Traits::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap          = Traits::kHasLogitsSoftCap;
    static constexpr auto BiasEnum                   = Traits::BiasEnum;
    static constexpr bool kStoreLSE                  = Traits::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant          = Traits::kDoFp8StaticQuant;
    static constexpr bool kIsPagedKV                 = Traits::kIsPagedKV;
    static constexpr bool kHasUnevenSplits           = kIsGroupMode || Traits::kHasUnevenSplits;
    static constexpr bool kMergeNumHeadGroupsSeqLenQ = Traits::kMergeNumHeadGroupsSeqLenQ;
    static constexpr index_t kBlockPerCu             = Traits::kBlockPerCu;
    static constexpr bool kHasSink                   = Traits::kHasSink;
};

// extract tile size attributes to remove dependency on traits
template <typename OaccDataType_, ck_tile::index_t kN1_>
struct BlockFmhaSplitKVCombinePipelineTileSizes
{
    static constexpr index_t MaxVectorSize = 16 / sizeof(OaccDataType_);

    static constexpr index_t kN1      = kN1_;
    static constexpr index_t NThreads = kN1 / MaxVectorSize;
    static constexpr index_t kM0      = get_warp_size() / NThreads; // MThreadPerWarp
};

template <typename LSEDataType_,
          typename OaccDataType_,
          typename ODataType_,
          index_t HeadDimV_,
          bool kIsGroupMode_,
          ck_tile::index_t kN1_,
          typename Traits_>
struct BlockFmhaSplitKVCombinePipelineProblem
    : BlockFmhaSplitKVCombinePipelineTileSizes<OaccDataType_, kN1_>
{
    using BaseType = BlockFmhaSplitKVCombinePipelineTileSizes<OaccDataType_, kN1_>;

    using LSEDataType  = remove_cvref_t<LSEDataType_>;
    using OaccDataType = remove_cvref_t<OaccDataType_>;
    using ODataType    = remove_cvref_t<ODataType_>;
    using Traits       = remove_cvref_t<Traits_>;

    static_assert(std::is_same_v<LSEDataType, OaccDataType>);

    static constexpr index_t kHeadDimV = HeadDimV_;
    static constexpr bool kIsGroupMode = kIsGroupMode_;

    using BaseType::kM0;
    using BaseType::kN1;
    using BaseType::NThreads;

    static_assert(kN1 <= kHeadDimV && kHeadDimV % kN1 == 0);

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
    static constexpr index_t kMaxSplits     = Traits::kMaxSplits;
    static_assert(8 <= kMaxSplits);

    static constexpr index_t kNumWarps  = 4;
    static constexpr index_t kBlockSize = kNumWarps * get_warp_size();

    static_assert(get_warp_size() <= (kM0 * kMaxSplits) &&
                  (kM0 * kMaxSplits) % get_warp_size() == 0);
};

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          index_t kM0_,
          index_t kN0_,
          index_t kK0_,
          index_t kN1_,
          bool kIsVLayoutRowMajor_,
          RotaryEmbeddingEnum RotaryEnum_,
          bool kIsPagedKV_,
          typename Traits_>
struct BlockFmhaFwdAppendKVPipelineProblem
{
    using QDataType = remove_cvref_t<QDataType_>;
    using KDataType = remove_cvref_t<KDataType_>;
    using VDataType = remove_cvref_t<VDataType_>;
    using Traits    = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = 256;

    static constexpr index_t kM0 = kM0_;
    static constexpr index_t kN0 = kN0_;
    static constexpr index_t kK0 = kK0_;
    static constexpr index_t kN1 = kN1_;

    using VLayout = std::conditional_t<kIsVLayoutRowMajor_,
                                       ck_tile::tensor_layout::gemm::RowMajor,
                                       ck_tile::tensor_layout::gemm::ColumnMajor>;

    static constexpr auto RotaryEnum = RotaryEnum_;
    static constexpr bool kIsPagedKV = kIsPagedKV_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK    = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

} // namespace ck_tile
