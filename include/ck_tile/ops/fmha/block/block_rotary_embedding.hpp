// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockRotaryEmbeddingEnum
{
    NONE         = 0,
    INTERLEAVED  = 1, // combine dimensions 0 & 1, 2 & 3, etc
    HALF_ROTATED = 2, // combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1, etc
};

template <BlockRotaryEmbeddingEnum>
struct BlockRotaryEmbeddingEnumToStr;

template <>
struct BlockRotaryEmbeddingEnumToStr<BlockRotaryEmbeddingEnum::NONE>
{
    static constexpr const char* name = "";
};
template <>
struct BlockRotaryEmbeddingEnumToStr<BlockRotaryEmbeddingEnum::INTERLEAVED>
{
    static constexpr const char* name = "inter";
};
template <>
struct BlockRotaryEmbeddingEnumToStr<BlockRotaryEmbeddingEnum::HALF_ROTATED>
{
    static constexpr const char* name = "half";
};

} // namespace ck_tile
