
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "layernorm_dispatch.hpp"

// clang-format off
#ifdef CK_TILE_LAYERNORM2D_FWD_FP32_DEFAULT
template float run_layernorm<float, 1, 32, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 1, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 1, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 2, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 2, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 4, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 4, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 8, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 8, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 16, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 8, 64, 4, false, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 16, 64, 2, false, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 1, 32, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 1, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 1, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 2, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 2, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 4, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 4, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 8, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 8, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 16, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 8, 64, 4, true, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
template float run_layernorm<float, 16, 64, 2, true, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
#endif
// clang-format on
