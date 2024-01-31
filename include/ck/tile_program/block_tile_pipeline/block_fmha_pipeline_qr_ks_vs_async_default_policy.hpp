// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

// This pipeline is qkv all located in LDS
using BlockFmhaPipelineQRKSVSAsyncDefaultPolicy =
    BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                        /* AsyncCopyK = */ true,
                                        /* AsyncCopyV = */ false,
                                        /* NumPrefetchK = */ 3,
                                        /* NumPrefetchV = */ 3>;

} // namespace block
} // namespace tile_program
} // namespace ck
