// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

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
// Traits -- other information required for running the kernel and pipeline

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
          bool kUseTrLoad_, // use transposed loading to load V tile from lds to vgprs
          typename AttentionTileSetting_>
struct HstuAttentionFwdPipelineProblem
{
    using InOutDataType   = remove_cvref_t<InOutDataType_>;
    using QKVDataType     = InOutDataType;
    using ODataType       = InOutDataType;
    using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;

    // DataType used when siLU calculation
    using CompDataType = remove_cvref_t<CompDataType_>;
    using BiasDataType = remove_cvref_t<BiasDataType_>;

    // to be compatible with ck_tile existing policy codes
    using OaccDataType = GemmAccDataType;
    using PDataType    = QKVDataType;

    static constexpr bool kIsCrossAttention = kIsCrossAttention_;
    static constexpr bool kUseGroup         = kUseGroup_;
    static constexpr bool kIsJagged         = kIsJagged_;
    static constexpr bool kHasBias          = kHasBias_;
    static constexpr bool kHasDropout       = kHasDropout_;
    static constexpr bool kHasCausal        = kHasCausal_;
    static constexpr bool kUseSoftmax       = kUseSoftmax_;
    static constexpr bool kUseTrLoad        = kUseTrLoad_;

    static_assert(!kUseGroup || (kUseGroup && kIsJagged),
                  "Group HSTU is only used with jagged mode!");

    using HstuAttentionTileSetting = remove_cvref_t<AttentionTileSetting_>;

    static constexpr index_t kNumGemm0Warps = AttentionTileSetting_::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = AttentionTileSetting_::NumGemm1Warps;
    static constexpr index_t kBlockSize     = AttentionTileSetting_::NumWarps * get_warp_size();

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
          typename ODataType_,
          bool kIsJagged_,
          bool kUseSoftmax_,
          typename CombineTileSetting_>
struct HstuAttentionFwdSplitKVCombinePipelineProblem
{
    using OaccDataType = remove_cvref_t<OaccDataType_>;
    using ODataType    = remove_cvref_t<ODataType_>;

    static constexpr bool kIsJagged   = kIsJagged_;
    static constexpr bool kUseSoftmax = kUseSoftmax_;

    static constexpr index_t kM           = CombineTileSetting_::kM;
    static constexpr index_t NumWarps     = CombineTileSetting_::NumWarps;
    static constexpr index_t kOHeaddim    = CombineTileSetting_::kOHeaddim;
    static constexpr index_t kSubOHeaddim = CombineTileSetting_::kSubOHeaddim;
    static constexpr index_t kBlockSize   = CombineTileSetting_::NumWarps * get_warp_size();

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
};

} // namespace ck_tile
