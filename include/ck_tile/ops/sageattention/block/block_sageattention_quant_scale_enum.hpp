// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockSageAttentionQuantScaleEnum
{
    NO_SCALE   = 0,
    PERTENSOR  = 1,
    BLOCKSCALE = 2,
    PERWARP    = 3,
    PERTHREAD  = 4,
};

template <BlockSageAttentionQuantScaleEnum>
struct BlockSageAttentionQuantScaleEnumToStr;

template <>
struct BlockSageAttentionQuantScaleEnumToStr<BlockSageAttentionQuantScaleEnum::NO_SCALE>
{
    static constexpr const char* name = "";
};
template <>
struct BlockSageAttentionQuantScaleEnumToStr<BlockSageAttentionQuantScaleEnum::PERTENSOR>
{
    static constexpr const char* name = "pertensor";
};
template <>
struct BlockSageAttentionQuantScaleEnumToStr<BlockSageAttentionQuantScaleEnum::BLOCKSCALE>
{
    static constexpr const char* name = "blockscale";
};
template <>
struct BlockSageAttentionQuantScaleEnumToStr<BlockSageAttentionQuantScaleEnum::PERWARP>
{
    static constexpr const char* name = "perwarp";
};
template <>
struct BlockSageAttentionQuantScaleEnumToStr<BlockSageAttentionQuantScaleEnum::PERTHREAD>
{
    static constexpr const char* name = "perthread";
};

} // namespace ck_tile
