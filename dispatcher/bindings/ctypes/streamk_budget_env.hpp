// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * The Stream-K workgroup budget the ctypes libraries that launch Stream-K kernels
 * (streamk_gemm_ctypes_lib.cpp, batched_gemm_ctypes_lib.cpp) read at each launch, in
 * percent of the kernel's default (CUs x occupancy). Python mirror:
 * gemm_utils.STREAMK_BUDGET_ENV.
 */

#pragma once

#include <cerrno>
#include <cstdlib>

// Parse CK_TILE_STREAMK_BUDGET_PCT strictly: unset/empty => 100; otherwise the
// entire value must be a base-10 integer in [1, 100], so typos or out-of-range
// budgets fail fast instead of silently running at 100%.
inline bool parse_streamk_budget_pct(int& out_budget_pct)
{
    const char* v = std::getenv("CK_TILE_STREAMK_BUDGET_PCT");
    if(v == nullptr || *v == '\0')
    {
        out_budget_pct = 100;
        return true;
    }
    errno          = 0;
    char* end      = nullptr;
    const long val = std::strtol(v, &end, 10);
    if(errno == ERANGE || end == v || *end != '\0' || val < 1 || val > 100)
        return false;
    out_budget_pct = static_cast<int>(val);
    return true;
}
