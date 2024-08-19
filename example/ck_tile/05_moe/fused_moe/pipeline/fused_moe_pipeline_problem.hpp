// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename GDataType_,
          typename UDataType_,
          typename DDataType_,
          typename ODataType_,
          typename AccDataType_,
          typename ScaleDataType_,
          typename GateActivation_, // = ck_tile::element_wise::Silu,
          typename FusedMoeTileShape_,
          typename Traits_>
struct FusedMoePipelineProblem
{
    using ADataType         = remove_cvref_t<ADataType_>;
    using YDataType         = ADataType;
    using GDataType         = remove_cvref_t<GDataType_>;
    using UDataType         = remove_cvref_t<UDataType_>;
    using DDataType         = remove_cvref_t<DDataType_>;
    using ODataType         = remove_cvref_t<ODataType_>;
    using AccDataType       = remove_cvref_t<AccDataType_>;
    using ScaleDataType     = remove_cvref_t<ScaleDataType_>;
    using FusedMoeTileShape = remove_cvref_t<FusedMoeTileShape_>;

    using Traits = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = FusedMoeTileShape::NumWarps * get_warp_size();

    // attributes from traits
    // static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    // static constexpr bool kPadSeqLenK       = Traits::kPadSeqLenK;
    // static constexpr bool kPadHeadDimQ      = Traits::kPadHeadDimQ;
    // static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    // static constexpr auto BiasEnum          = Traits::BiasEnum;
    // static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    // static constexpr bool kHasDropout       = Traits::kHasDropout;
    // static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;

    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
    using GateActivation                 = remove_cvref_t<typename Traits::GateActivation_>;
};
} // namespace ck_tile
