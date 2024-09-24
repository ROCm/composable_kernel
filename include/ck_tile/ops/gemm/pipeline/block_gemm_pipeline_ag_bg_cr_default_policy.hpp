// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_ag_bg_cr_mem_custom_policy.hpp"

namespace ck_tile {

// Default policy for BlockGemmPipelineAGmemBGmemCRegV1
// Default policy class should not be templated, put template on member functions instead
using BlockGemmPipelineAgBgCrDefaultPolicy = BlockGemmPipelineAgBgCrMemCustomPolicy;

} // namespace ck_tile
