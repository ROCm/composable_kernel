// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattention/pipeline/block_sageattn_pipeline_qr_ks_vs_custom_policy.hpp"

namespace ck_tile {

using BlockSageAttentionPipelineQRKSVSDefaultPolicy =
    BlockSageAttnPipelineQRKSVSCustomPolicy</* QLoadOnce = */ true,
                                            /* AsyncCopy = */ false,
                                            /* NumPrefetchK = */ 1,
                                            /* NumPrefetchV = */ 1>;

} // namespace ck_tile
