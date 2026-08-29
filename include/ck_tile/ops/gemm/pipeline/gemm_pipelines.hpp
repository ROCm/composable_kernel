// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

enum struct GemmPipeline
{
    COMPUTE_ASYNC,
    COMPUTE_V3,
    COMPUTE_V4,
    COMPUTE_V5,
    COMPUTE_V6,
    MEMORY,
    BASIC_V1,
    BASIC_V2,
    PRESHUFFLE_V2,
    BASIC_ASYNC_V1,
    COMPUTE_TDM_V1,
    COMPUTE_TDM_V2,
    COMPUTE_ASYNC_V2,
    PRESHUFFLE_FLATMM,
    PRESHUFFLE_TDM,
    PRESHUFFLE_MX_TDM,
    COMPUTE_MX_TDM,
    WAVELET
};

} // namespace ck_tile
