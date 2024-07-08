// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

// This pipeline is v located in regs, k & k^t located in lds.
using BlockFmhaBwdDQDKDVPipelineKSKTSVRDefaultPolicy =
    BlockFmhaBwdPipelineDefaultPolicy</* QLoadOnce_      = */ false,
                                      /* QTLoadOnce_     = */ false,
                                      /* KLoadOnce_      = */ true,
                                      /* KTLoadOnce_     = */ true,
                                      /* VLoadOnce_      = */ true,
                                      /* OGradLoadOnce_  = */ false,
                                      /* OGradTLoadOnce_ = */ false>;

} // namespace ck_tile
