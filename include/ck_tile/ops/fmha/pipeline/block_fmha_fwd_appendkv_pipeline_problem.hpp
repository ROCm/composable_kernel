// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          index_t kTileSizeS_,
          index_t kTileSizeSk_,
          index_t kTileSizeD_,
          index_t kTileSizeDv_,
          bool IsVLayoutRowMajor_,
          bool kIsGroupMode_,
          typename Traits_>
struct BlockFmhaFwdAppendKVPipelineProblem
{
    using QDataType = remove_cvref_t<QDataType_>;
    using KDataType = remove_cvref_t<KDataType_>;
    using VDataType = remove_cvref_t<VDataType_>;
    using Traits    = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = 256;
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    static constexpr index_t kTileSizeS  = kTileSizeS_;
    static constexpr index_t kTileSizeSk = kTileSizeSk_;
    static constexpr index_t kTileSizeD  = kTileSizeD_;
    static constexpr index_t kTileSizeDv = kTileSizeDv_;

    using VLayout = std::conditional_t<IsVLayoutRowMajor_,
                                       ck_tile::tensor_layout::gemm::RowMajor,
                                       ck_tile::tensor_layout::gemm::ColumnMajor>;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK    = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr bool kApplyRoPE     = Traits::kApplyRoPE;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

} // namespace ck_tile
