// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

// These templates are not used here.
using BlockFmhaBwdOGradDotODefaultPolicy =
    BlockFmhaBwdPipelineDefaultPolicy</* QLoadOnce_      = */ false,
                                      /* QTLoadOnce_     = */ false,
                                      /* KLoadOnce_      = */ false,
                                      /* KTLoadOnce_     = */ false,
                                      /* VLoadOnce_      = */ false,
                                      /* OGradLoadOnce_  = */ false,
                                      /* OGradTLoadOnce_ = */ false>;

} // namespace ck_tile
