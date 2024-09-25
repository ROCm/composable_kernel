// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

enum struct BlockGemmPipelineScheduler
{
    Intrawave,
    Interwave,
};

enum struct TailNumber
{
    // Single / Double buffer pipeline
    Odd,
    Even,

    // Long prefetch pipeline, up to 8
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,

    // Unroll stages > Prefetch stages, number of loop is multiple of unroll stages
    Empty,
    // Unroll stages <= Prefetch stages, number of loop is multiple of unroll stages add
    // prefetchstages
    Full,
};

} // namespace ck_tile
