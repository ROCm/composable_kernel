// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

// This class is used for codegen pattern matching
enum class BlockFmhaPipelineEnum
{
    QRKSVS = 0,
    QRKSVS_ASYNC,
    QRKSVS_FP8,
    QSKSVS,
};

} // namespace ck
