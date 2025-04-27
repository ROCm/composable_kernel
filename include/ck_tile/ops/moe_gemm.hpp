// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/flatmm/pipeline/tile_flatmm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"

#include "ck_tile/ops/moe_gemm/kernel/moe_gemm_kernel.hpp"
#include "ck_tile/ops/moe_gemm/pipeline/moe_gemm_pipeline_agmem_bgmem_creg_flatmm.hpp"
#include "ck_tile/ops/moe_gemm/pipeline/moe_gemm_pipeline_agmem_bgmem_creg_flatmm_policy.hpp"
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/common/utils.hpp"
