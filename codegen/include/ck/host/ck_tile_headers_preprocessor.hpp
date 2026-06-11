// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>

namespace ck {
namespace host {

// Preprocesses a ck_tile header for HIPRTC compilation by replacing
// the bodies of CK_TILE_HOST-only functions with stubs. This prevents
// host-only code (which references APIs unavailable in HIPRTC) from
// being type-checked by the device compiler.
//
// For non-constexpr functions:  { __builtin_unreachable(); }
// For constexpr functions:      { return {}; }
// For constexpr auto functions: { return 0; }
std::string strip_host_bodies(std::string_view content);

} // namespace host
} // namespace ck
