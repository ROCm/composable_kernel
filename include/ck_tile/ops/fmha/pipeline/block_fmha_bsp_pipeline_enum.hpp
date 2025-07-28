// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaBspPipelineEnum
{
    QRKSVS_ASYNC=0,
};

template <BlockFmhaBspPipelineEnum>
struct BlockFmhaBspPipelineEnumToStr;

template <>
struct BlockFmhaBspPipelineEnumToStr<BlockFmhaBspPipelineEnum::QRKSVS_ASYNC>
{
    static constexpr const char* name = "qr_async";
};

} // namespace ck_tile
