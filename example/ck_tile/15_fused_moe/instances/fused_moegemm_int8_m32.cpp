// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"
#include "fused_moegemm_api_traits.hpp"
#include "fused_moegemm_api_internal.hpp"

// clang-format off
template float fused_moegemm_<
    fmoe_<ck_tile::int8_t, ck_tile::int8_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 256, 256>, S<1, 4, 1>, S<16, 16, 64>, 1, 1>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

// clang-format on
