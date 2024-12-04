// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"
#include "fused_moegemm_api_traits.hpp"
#include "fused_moegemm_api_internal.hpp"

// clang-format off
template float fused_moegemm_<
    fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 128, 32, 32>, S<1, 4, 1>, S<32, 32, 8>, 1, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

// clang-format on
