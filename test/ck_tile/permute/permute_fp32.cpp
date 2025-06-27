// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "permute.hpp"
#include "ck_tile/host.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef PERMUTE_USE_ALTERNATIVE_IMPL
#include "alternative_impl/matrix_core_swizzle.hpp"
#endif

#include "permute_utils.inc"

int main()
{
    std::vector<std::vector<std::string>> test_cases = create_test_cases("fp32");

    return !run_test_cases<float>(test_cases);
}
