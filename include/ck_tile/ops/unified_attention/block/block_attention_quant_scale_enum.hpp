// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class UnifiedAttentionQuantScaleEnum
{
    NO_SCALE = 0,
    FP8      = 2,
};

template <UnifiedAttentionQuantScaleEnum>
struct UnifiedAttentionQuantScaleEnumToStr;

template <>
struct UnifiedAttentionQuantScaleEnumToStr<UnifiedAttentionQuantScaleEnum::NO_SCALE>
{
    static constexpr const char* name = "";
};

template <>
struct UnifiedAttentionQuantScaleEnumToStr<UnifiedAttentionQuantScaleEnum::FP8>
{
    static constexpr const char* name = "fp8";
};

} // namespace ck_tile
