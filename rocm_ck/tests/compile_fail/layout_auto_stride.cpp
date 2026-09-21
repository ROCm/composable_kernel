// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: Layout::Auto cannot be used for stride computation.
// Expected error: "leadingDimStride requires Row or Col layout"

#include <rocm_ck/layout.hpp>

#include <array>

using namespace rocm_ck;

constexpr auto bad = leadingDimStride(Layout::Auto, std::array{1, 2});
