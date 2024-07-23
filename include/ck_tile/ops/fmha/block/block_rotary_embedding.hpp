// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class RotaryEmbeddingEnum
{
    NONE         = 0,
    INTERLEAVED  = 1, // combine dimensions 0 & 1, 2 & 3, etc
    HALF_ROTATED = 2, // combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1, etc
};

template <RotaryEmbeddingEnum>
struct RotaryEmbeddingEnumToStr;

template <>
struct RotaryEmbeddingEnumToStr<RotaryEmbeddingEnum::NONE>
{
    static constexpr const char* name = "";
};
template <>
struct RotaryEmbeddingEnumToStr<RotaryEmbeddingEnum::INTERLEAVED>
{
    static constexpr const char* name = "inter";
};
template <>
struct RotaryEmbeddingEnumToStr<RotaryEmbeddingEnum::HALF_ROTATED>
{
    static constexpr const char* name = "half";
};

} // namespace ck_tile
