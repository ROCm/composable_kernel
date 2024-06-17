// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaBwdPipelineEnum
{
    KSKTSVR = 0,
    QSKSVROGradS,
    KSVR,
};

} // namespace ck_tile
