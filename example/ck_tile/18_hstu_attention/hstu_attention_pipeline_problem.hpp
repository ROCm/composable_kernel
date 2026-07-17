// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include <ck_tile/core/config.hpp>
#include <ck_tile/core/numeric/integer.hpp>

#include "hstu_attention_tile_setting_define.hpp"

namespace ck_tile {

namespace detail {

template <typename T>
struct is_hstu_attention_fwd_tile_setting : std::false_type
{
};

template <typename BlockTile_,
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_>
struct is_hstu_attention_fwd_tile_setting<HstuAttentionFwdTileSettingClass<BlockTile_,
                                                                           Gemm0BlockWarps_,
                                                                           Gemm0WarpTile_,
                                                                           Gemm1BlockWarps_,
                                                                           Gemm1WarpTile_>>
    : std::true_type
{
};

template <typename T>
struct is_hstu_attention_fwd_splitkv_combine_tile_setting : std::false_type
{
};

template <index_t kM_, index_t NumWarps_, index_t kOHeaddim_>
struct is_hstu_attention_fwd_splitkv_combine_tile_setting<
    HstuAttentionFwdSplitKVCombineTileSettingClass<kM_, NumWarps_, kOHeaddim_>> : std::true_type
{
};

template <typename T>
struct is_hstu_attention_bwd_tile_setting_for_kernel1 : std::false_type
{
};

template <typename BlockTile_,
          typename Gemm0Gemm2BlockWarps_,
          typename Gemm0Gemm2WarpTile_,
          typename Gemm4BlockWarps_,
          typename Gemm4WarpTile_>
struct is_hstu_attention_bwd_tile_setting_for_kernel1<
    HstuAttentionBwdTileSettingClassForKernel1<BlockTile_,
                                               Gemm0Gemm2BlockWarps_,
                                               Gemm0Gemm2WarpTile_,
                                               Gemm4BlockWarps_,
                                               Gemm4WarpTile_>> : std::true_type
{
};

template <typename T>
struct is_hstu_attention_bwd_tile_setting_for_kernel2 : std::false_type
{
};

template <typename BlockTile_,
          typename Gemm0Gemm2BlockWarps_,
          typename Gemm0Gemm2WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          typename Gemm3BlockWarps_,
          typename Gemm3WarpTile_>
struct is_hstu_attention_bwd_tile_setting_for_kernel2<
    HstuAttentionBwdTileSettingClassForKernel2<BlockTile_,
                                               Gemm0Gemm2BlockWarps_,
                                               Gemm0Gemm2WarpTile_,
                                               Gemm1BlockWarps_,
                                               Gemm1WarpTile_,
                                               Gemm3BlockWarps_,
                                               Gemm3WarpTile_>> : std::true_type
{
};

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
        static_assert(false, "The data type is not supported!");
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

}; // namespace detail

// PipelineProblem encodes information not only from the original user-problem,
// but it also contains other information needed by the pipeline, which includes
// TileShape -- which determines how block-layer calculation is done in tiles and
//              how warps are allocated on dimensions
template <typename InOutDataType_,
          typename GemmAccDataType_,
          typename CompDataType_, // data type for SiLU and other non-linear calculation
          typename BiasDataType_,
          bool kIsCrossAttention_,
          bool kUseGroup_,
          bool kIsJagged_,
          bool kHasBias_,
          bool kHasDropout_,
          bool kHasCausal_,
          bool kUseSoftmax_,
          bool kStoreLSE_,
          typename TileSetting_>
struct HstuAttentionFwdPipelineProblem
{
    using InOutDataType   = remove_cvref_t<InOutDataType_>;
    using QKVDataType     = InOutDataType;
    using ODataType       = InOutDataType;
    using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;

    // DataType used when siLU calculation
    using CompDataType = remove_cvref_t<CompDataType_>;
    using BiasDataType = remove_cvref_t<BiasDataType_>;

    using OaccDataType = GemmAccDataType;
    using PDataType    = QKVDataType;

    static constexpr bool kIsCrossAttention = kIsCrossAttention_;
    static constexpr bool kUseGroup         = kUseGroup_;
    static constexpr bool kIsJagged         = kIsJagged_;
    static constexpr bool kHasBias          = kHasBias_;
    static constexpr bool kHasDropout       = kHasDropout_;
    static constexpr bool kHasCausal        = kHasCausal_;
    static constexpr bool kUseSoftmax       = kUseSoftmax_;
    static constexpr bool kStoreLSE         = kStoreLSE_;

    static_assert(detail::is_hstu_attention_fwd_tile_setting<remove_cvref_t<TileSetting_>>::value,
                  "TileSetting_ must be an instance of HstuAttentionFwdTileSettingClass!");
    static_assert(!kUseGroup || (kUseGroup && kIsJagged),
                  "Group HSTU is only used with jagged mode!");
    static_assert(!kStoreLSE || (kStoreLSE && kUseSoftmax),
                  "Storing Lse is only necessary when softmax is used!");

    using HstuAttentionTileSetting = remove_cvref_t<TileSetting_>;

    static constexpr index_t kNumGemm0Warps = TileSetting_::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = TileSetting_::NumGemm1Warps;
    static constexpr index_t kBlockSize     = TileSetting_::NumWarps * get_warp_size();

    CK_TILE_HOST_DEVICE static constexpr auto GetQDramTileAccessMaxVectorSize()
    {
        constexpr index_t kMPerBlock = HstuAttentionTileSetting::kM0;
        constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

        return detail::
            GetDramTileAccessMaxVectorSize<QKVDataType, kBlockSize, kMPerBlock, kKPerBlock>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetKDramTileAccessMaxVectorSize()
    {
        constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

        return detail::
            GetDramTileAccessMaxVectorSize<QKVDataType, kBlockSize, kNPerBlock, kKPerBlock>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetVDramTileAccessMaxVectorSize()
    {
        constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = HstuAttentionTileSetting::kK1;

        return detail::
            GetDramTileAccessMaxVectorSize<QKVDataType, kBlockSize, kNPerBlock, kKPerBlock>();
    };
};

template <typename OaccDataType_,
          typename LSEDataType_,
          typename ODataType_,
          bool kIsJagged_,
          bool kUseSoftmax_,
          bool kStoreLSE_,
          typename TileSetting_,
          index_t kMaxSplits_ = 0>
struct HstuAttentionFwdSplitKVCombinePipelineProblem
{
    using OaccDataType = remove_cvref_t<OaccDataType_>;
    using ODataType    = remove_cvref_t<ODataType_>;
    using LSEDataType  = remove_cvref_t<LSEDataType_>;

    static_assert(
        detail::is_hstu_attention_fwd_splitkv_combine_tile_setting<
            remove_cvref_t<TileSetting_>>::value,
        "TileSetting_ must be an instance of HstuAttentionFwdSplitKVCombineTileSettingClass!");

    static constexpr bool kIsJagged   = kIsJagged_;
    static constexpr bool kUseSoftmax = kUseSoftmax_;
    static constexpr bool kStoreLSE   = kStoreLSE_;

    static constexpr index_t kM           = TileSetting_::kM;
    static constexpr index_t NumWarps     = TileSetting_::NumWarps;
    static constexpr index_t kOHeaddim    = TileSetting_::kOHeaddim;
    static constexpr index_t kSubOHeaddim = TileSetting_::kSubOHeaddim;
    static constexpr index_t kBlockSize   = TileSetting_::NumWarps * get_warp_size();
    static constexpr index_t kMaxSplits   = kMaxSplits_;

    static_assert((kMaxSplits == 0) || (kM * kMaxSplits >= kBlockSize), "Check failed!");

    CK_TILE_HOST_DEVICE static constexpr auto GetOaccDramTileAccessMaxVectorSize()
    {
        constexpr index_t kMPerBlock = kM;
        constexpr index_t kKPerBlock = kOHeaddim;

        return detail::
            GetDramTileAccessMaxVectorSize<OaccDataType, kBlockSize, kMPerBlock, kKPerBlock>();
    };

    CK_TILE_HOST_DEVICE static constexpr auto GetODramTileAccessMaxVectorSize()
    {
        constexpr index_t kMPerBlock = kM;
        constexpr index_t kKPerBlock = kOHeaddim;

        return detail::
            GetDramTileAccessMaxVectorSize<ODataType, kBlockSize, kMPerBlock, kKPerBlock>();
    };

    CK_TILE_HOST_DEVICE static constexpr auto GetLSEaccDramTileAccessMaxVectorSize()
    {
        constexpr index_t kMPerBlock = kM;
        constexpr index_t kKPerBlock = kMaxSplits;

        return detail::
            GetDramTileAccessMaxVectorSize<LSEDataType, kBlockSize, kMPerBlock, kKPerBlock>();
    };
};

// Shared template parameters for both backward kernels, excluding tile-sizing policy.
// Analogous to HstuAttentionFwdPipelineProblem but without TileSetting.
template <typename InOutDataType_,   // fp16 or bf16 -- Q, K, V, O, dO, dQ, dK, dV
          typename GemmAccDataType_, // float -- GEMM accumulator
          typename CompDataType_,    // float -- SiLU / softmax intermediate
          bool kIsCrossAttention_,
          bool kUseGroup_,
          bool kIsJagged_,
          bool kHasBias_, // bias added to S (same semantics as forward)
          bool kHasCausal_,
          bool kUseSoftmax_,
          bool kHasDropout_>
struct HstuAttentionBwdPipelineBaseProblem
{
    using InOutDataType   = remove_cvref_t<InOutDataType_>;
    using QKVDataType     = InOutDataType;
    using BiasDataType    = InOutDataType;
    using ODataType       = InOutDataType;
    using OGradDataType   = InOutDataType;
    using QGradDataType   = InOutDataType;
    using KGradDataType   = InOutDataType;
    using VGradDataType   = InOutDataType;
    using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;
    using CompDataType    = remove_cvref_t<CompDataType_>;

    using PDataType    = QKVDataType;
    using OaccDataType = GemmAccDataType;

    static constexpr bool kIsCrossAttention = kIsCrossAttention_;
    static constexpr bool kUseGroup         = kUseGroup_;
    static constexpr bool kIsJagged         = kIsJagged_;
    static constexpr bool kHasBias          = kHasBias_;
    static constexpr bool kHasCausal        = kHasCausal_;
    static constexpr bool kUseSoftmax       = kUseSoftmax_;
    static constexpr bool kHasDropout       = kHasDropout_;

    static_assert(!kUseGroup || (kUseGroup && kIsJagged),
                  "Group HSTU is only used with jagged mode!");
};

// Kernel 1: computes dQ (and D[sq] = dO row(.) O for the softmax path).
// Iterates over K/V dimension blocks; one block per (batch, head, sq_tile).
// TileSetting must be an HstuAttentionTileSettingClassForKernel1 instance.
template <typename PipelineBaseProblem_, // HstuAttentionBwdPipelineBaseProblem instance
          typename TileSetting_>         // HstuAttentionTileSettingClassForKernel1 instance
struct HstuAttentionBwdPipelineProblemForKernel1
{
    static_assert(
        detail::is_hstu_attention_bwd_tile_setting_for_kernel1<remove_cvref_t<TileSetting_>>::value,
        "TileSetting_ must be an instance of HstuAttentionBwdTileSettingClassForKernel1");

    using BaseProblem = remove_cvref_t<PipelineBaseProblem_>;

    using InOutDataType    = typename BaseProblem::InOutDataType;
    using QKVDataType      = typename BaseProblem::QKVDataType;
    using ODataType        = typename BaseProblem::ODataType;
    using BiasDataType     = typename BaseProblem::BiasDataType;
    using OGradDataType    = typename BaseProblem::OGradDataType;
    using QGradDataType    = typename BaseProblem::QGradDataType;
    using KGradDataType    = typename BaseProblem::KGradDataType;
    using VGradDataType    = typename BaseProblem::VGradDataType;
    using GemmAccDataType  = typename BaseProblem::GemmAccDataType;
    using CompDataType     = typename BaseProblem::CompDataType;
    using PDataType        = typename BaseProblem::PDataType;
    using QGradAccDataType = typename BaseProblem::GemmAccDataType;

    static constexpr bool kIsCrossAttention = BaseProblem::kIsCrossAttention;
    static constexpr bool kUseGroup         = BaseProblem::kUseGroup;
    static constexpr bool kIsJagged         = BaseProblem::kIsJagged;
    static constexpr bool kHasBias          = BaseProblem::kHasBias;
    static constexpr bool kHasCausal        = BaseProblem::kHasCausal;
    static constexpr bool kUseSoftmax       = BaseProblem::kUseSoftmax;
    static constexpr bool kHasDropout       = BaseProblem::kHasDropout;

    using HstuAttentionTileSetting = remove_cvref_t<TileSetting_>;

    static constexpr bool IsWarpGemm32 = HstuAttentionTileSetting::IsWarpGemm32;

    static constexpr index_t kNumGemm0Gemm2Warps = TileSetting_::NumGemm0Gemm2Warps;
    static constexpr index_t kNumGemm4Warps      = TileSetting_::NumGemm4Warps;
    static constexpr index_t kBlockSize          = TileSetting_::NumWarps * get_warp_size();

    // K tile: [kN0Sub, kQKHeaddim]
    CK_TILE_HOST_DEVICE static constexpr auto GetKDramTileAccessMaxVectorSize()
    {
        constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

        return detail::
            GetDramTileAccessMaxVectorSize<QKVDataType, kBlockSize, kNPerBlock, kKPerBlock>();
    }

    // V tile: [kN0Sub, kQKHeaddim]
    CK_TILE_HOST_DEVICE static constexpr auto GetVDramTileAccessMaxVectorSize()
    {
        constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

        return detail::
            GetDramTileAccessMaxVectorSize<QKVDataType, kBlockSize, kNPerBlock, kKPerBlock>();
    }
};

// Kernel 2: computes dK and dV.
// Iterates over Q dimension blocks; one block per (batch, head, sk_tile).
// TileSetting must be an HstuAttentionTileSettingClassForKernel2 instance.
template <typename PipelineBaseProblem_, // HstuAttentionBwdPipelineBaseProblem instance
          typename TileSetting_>         // HstuAttentionTileSettingClassForKernel2 instance
struct HstuAttentionBwdPipelineProblemForKernel2
{
    static_assert(
        detail::is_hstu_attention_bwd_tile_setting_for_kernel2<remove_cvref_t<TileSetting_>>::value,
        "TileSetting_ must be an instance of HstuAttentionBwdTileSettingClassForKernel2");

    using BaseProblem = remove_cvref_t<PipelineBaseProblem_>;

    using InOutDataType    = typename BaseProblem::InOutDataType;
    using QKVDataType      = typename BaseProblem::QKVDataType;
    using ODataType        = typename BaseProblem::ODataType;
    using BiasDataType     = typename BaseProblem::BiasDataType;
    using OGradDataType    = typename BaseProblem::OGradDataType;
    using QGradDataType    = typename BaseProblem::QGradDataType;
    using KGradDataType    = typename BaseProblem::KGradDataType;
    using VGradDataType    = typename BaseProblem::VGradDataType;
    using GemmAccDataType  = typename BaseProblem::GemmAccDataType;
    using CompDataType     = typename BaseProblem::CompDataType;
    using PDataType        = typename BaseProblem::PDataType;
    using KGradAccDataType = typename BaseProblem::GemmAccDataType;
    using VGradAccDataType = typename BaseProblem::GemmAccDataType;

    static constexpr bool kIsCrossAttention = BaseProblem::kIsCrossAttention;
    static constexpr bool kUseGroup         = BaseProblem::kUseGroup;
    static constexpr bool kIsJagged         = BaseProblem::kIsJagged;
    static constexpr bool kHasBias          = BaseProblem::kHasBias;
    static constexpr bool kHasCausal        = BaseProblem::kHasCausal;
    static constexpr bool kUseSoftmax       = BaseProblem::kUseSoftmax;
    static constexpr bool kHasDropout       = BaseProblem::kHasDropout;

    using HstuAttentionTileSetting = remove_cvref_t<TileSetting_>;

    static constexpr bool IsWarpGemm32 = HstuAttentionTileSetting::IsWarpGemm32;

    static constexpr index_t kNumGemm0Gemm2Warps = TileSetting_::NumGemm0Gemm2Warps;
    static constexpr index_t kNumGemm1Warps      = TileSetting_::NumGemm1Warps;
    static constexpr index_t kNumGemm3Warps      = TileSetting_::NumGemm3Warps;
    static constexpr index_t kBlockSize          = TileSetting_::NumWarps * get_warp_size();

    // Q tile: [kM0, kK0]
    CK_TILE_HOST_DEVICE static constexpr auto GetQDramTileAccessMaxVectorSize()
    {
        constexpr index_t kMPerBlock = HstuAttentionTileSetting::kM0;
        constexpr index_t kKPerBlock = HstuAttentionTileSetting::kK0;

        return detail::
            GetDramTileAccessMaxVectorSize<QKVDataType, kBlockSize, kMPerBlock, kKPerBlock>();
    }
};

} // namespace ck_tile
