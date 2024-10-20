// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
using BlockFmhaFwdSplitKVPipelineQRKSVSDefaultPolicy =
    BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                        /* AsyncCopyK = */ false,
                                        /* AsyncCopyV = */ false,
                                        /* NumPrefetchK = */ 1,
                                        /* NumPrefetchV = */ 1>;

} // namespace ck_tile
