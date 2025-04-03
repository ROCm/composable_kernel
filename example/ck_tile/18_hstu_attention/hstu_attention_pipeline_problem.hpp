// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

// PipelineProblem encodes information not only from the original user-problem,
// but it also contains other information needed by the pipeline, which includes
// TileShape -- which determines how block-layer calculation is done in tiles and
//              how warps are allocated on dimensions
// Traits -- other information required for running the kernel and pipeline

template <typename InOutDataType_,
          typename GemmAccDataType_,
          typename CompDataType_, // data type for SiLU and other non-linear calculation
          typename BiasDataType_,
          bool kIsJagged_,
          bool kHasBias_,
          bool kHasDropout_,
          typename HstuMask_, // encoding Causal and Local, contextual masking
          typename AttentionTileShape_,
          typename Traits_>
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
    using QDataType    = QKVDataType;
    using KDataType    = QKVDataType;
    using VDataType    = QKVDataType;
    using SaccDataType = GemmAccDataType;
    using OaccDataType = GemmAccDataType;
    using PDataType    = QKVDataType;

    static constexpr bool kIsJagged   = kIsJagged_;
    static constexpr bool kHasBias    = kHasBias_;
    static constexpr bool kHasDropout = kHasDropout_;

    using HstuMask = remove_cvref_t<HstuMask_>;

    using HstuAttentionTileShape = remove_cvref_t<AttentionTileShape_>;

    // Keep the name compatible with ck_tile existing policy codes, to be changed
    using BlockFmhaShape = HstuAttentionTileShape;
    using Traits         = remove_cvref_t<Traits_>;

    static constexpr index_t kNumGemm0Warps = AttentionTileShape_::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = AttentionTileShape_::NumGemm1Warps;
    static constexpr index_t kBlockSize     = AttentionTileShape_::NumWarps * get_warp_size();
};

} // namespace ck_tile
