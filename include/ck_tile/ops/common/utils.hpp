// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
namespace ck_tile {
// clang-format off
template <typename T> struct t2s;
template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
template <> struct t2s<fp16_t> { static constexpr const char * name = "fp16"; };
template <> struct t2s<bf16_t> { static constexpr const char * name = "bf16"; };
template <> struct t2s<fp8_t> { static constexpr const char * name = "fp8"; };
template <> struct t2s<bf8_t> { static constexpr const char * name = "bf8"; };
template <> struct t2s<int8_t> { static constexpr const char * name = "int8"; };
// clang-format on
} // namespace ck_tile
