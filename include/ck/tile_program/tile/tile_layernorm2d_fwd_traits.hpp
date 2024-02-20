// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <bool HasGamma_, bool HasBeta_, bool SaveMeanInvStd_>
struct TileLayernorm2dFwdTraits
{
    static constexpr bool HasGamma       = HasGamma_;
    static constexpr bool HasBeta        = HasBeta_;
    static constexpr bool SaveMeanInvStd = SaveMeanInvStd_;
};

} // namespace tile_program
} // namespace ck
