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
    Q_PERTENSOR_KV_BLOCKSCALE,
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
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::Q_PERTENSOR_KV_BLOCKSCALE>
{
    static constexpr const char* name = "q_pertensor_kv_blockscale";
};

} // namespace ck_tile
