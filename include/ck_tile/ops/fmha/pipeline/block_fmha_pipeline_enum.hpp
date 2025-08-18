// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaPipelineEnum
{
    QRKSVS = 0,
    QRKSVS_ASYNC,
    QSKSVS,
    QRKSVS_ASYNC_TRLOAD,
};

template <BlockFmhaPipelineEnum>
struct BlockFmhaPipelineEnumToStr;

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS>
{
    static constexpr const char* name = "qr";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_ASYNC>
{
    static constexpr const char* name = "qr_async";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QSKSVS>
{
    static constexpr const char* name = "qs";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD>
{
    static constexpr const char* name = "qr_async_trload";
};

} // namespace ck_tile
