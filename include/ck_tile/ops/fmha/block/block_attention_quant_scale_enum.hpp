// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockAttentionQuantScaleEnum
{
    NO_SCALE  = 0,
    PERTENSOR = 1,
};

template <BlockAttentionQuantScaleEnum>
struct BlockAttentionQuantScaleEnumToStr;

template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::NO_SCALE>
{
    static constexpr const char* name = "";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::PERTENSOR>
{
    static constexpr const char* name = "pertensor";
};

} // namespace ck_tile
