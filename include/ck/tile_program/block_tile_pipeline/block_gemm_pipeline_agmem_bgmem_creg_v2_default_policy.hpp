// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmPipelineAGmemBGmemCRegV2
// Default policy class should not be templated, put template on member functions instead
// NOTE: policy should be binded to its corresponding operation. It's just a coincidence that
//   BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy is the same as
//   BlockGemmPipelineAGmemBGmemCRegV1DefaultPolicy
using BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy =
    BlockGemmPipelineAGmemBGmemCRegV1DefaultPolicy;

} // namespace block
} // namespace tile_program
} // namespace ck
