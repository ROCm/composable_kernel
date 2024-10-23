// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename XDataType_,
          typename GammaDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename InvRmsDataType_,
          typename BlockShape_,
          bool kPadN_,
          bool kSaveInvRms_,
          bool kTwoPass_>
struct Rmsnorm2dFwdPipelineProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using GammaDataType   = remove_cvref_t<GammaDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using InvRmsDataType  = remove_cvref_t<InvRmsDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;

    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;

    static constexpr bool kPadN       = kPadN_;
    static constexpr bool kSaveInvRms = kSaveInvRms_;
    static constexpr bool kTwoPass    = kTwoPass_;
};

} // namespace ck_tile
