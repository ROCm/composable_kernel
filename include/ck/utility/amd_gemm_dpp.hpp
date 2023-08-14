// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_gemm_dpp.hpp"

namespace ck {

namespace dpp8 {

/// Number of lanes that can share data using DPP8 modifiers.
constexpr index_t lane_group_size = 8;

__device__ index_t get_lane_group_local_idx() { return threadIdx.x / lane_group_size; }
__device__ index_t get_thread_idx_in_lane_group() { return threadIdx.x % lane_group_size; }

} // namespace dpp8

} // namespace ck
