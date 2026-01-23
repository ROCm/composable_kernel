// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockAttentionQuantScaleEnum
{
    NO_SCALE  = 0,
    PERTENSOR = 1,
    BLOCKSCALE,
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
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::BLOCKSCALE>
{
    static constexpr const char* name = "blockscale";
};

} // namespace ck_tile
