
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "layernorm_dispatch.hpp"

// clang-format off
// template float run_layernorm<ck_tile::fp16_t, 1, 16, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 1, 32, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 1, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// template float run_layernorm<ck_tile::fp16_t, 1, 32, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 1, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 2, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// template float run_layernorm<ck_tile::fp16_t, 1, 64, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 2, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 4, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// template float run_layernorm<ck_tile::fp16_t, 2, 64, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 4, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 8, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// template float run_layernorm<ck_tile::fp16_t, 4, 64, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 8, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 16, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// template float run_layernorm<ck_tile::fp16_t, 4, 64, 8, false, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 8, 64, 4, false, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<ck_tile::fp16_t, 16, 64, 2, false, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// clang-format on
