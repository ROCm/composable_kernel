// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Fwd instances
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"

// Bwd weight instances
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_wmma_cshuffle.hpp"

// Bwd data instances
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp"
#include "ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp"
