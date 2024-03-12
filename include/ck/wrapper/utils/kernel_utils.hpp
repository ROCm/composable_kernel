// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

#define __CK_WRAPPER_LAUNCH_BOUNDS__ __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)

} // namespace wrapper
} // namespace ck
