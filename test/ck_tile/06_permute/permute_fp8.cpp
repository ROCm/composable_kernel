// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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
    std::vector<std::vector<std::string>> test_cases
    {
    {"-prec=fp8", "-shape=3,8", "-perm=1,0", "-v=1", "-warmup=0", "-repeat=1"},
    {"-prec=fp8", "-shape=48,6,8", "-perm=2,1,0",  "-v=1", "-warmup=0", "-repeat=1"},
    {"-prec=fp8", "-shape=24,128,3", "-perm=0,2,1",  "-v=1", "-warmup=0", "-repeat=1"},
    {"-prec=fp8", "-shape=4,10,7,6", "-perm=0,2,3,1",  "-v=1", "-warmup=0", "-repeat=1"},
    {"-prec=fp8", "-shape=8,24,36,10", "-perm=3,1,2,0",  "-v=1", "-warmup=0", "-repeat=1"},
    {"-prec=fp8", "-shape=8,1,36,4", "-perm=2,1,0,3",  "-v=1", "-warmup=0", "-repeat=1"},
    {"-prec=fp8", "-shape=5,10,16,2,36,4", "-perm=4,5,2,1,0,3",  "-v=1", "-warmup=0", "-repeat=1"},
    {"-prec=fp8", "-shape=2,32,8,3,6,2,5,4", "-perm=5,2,4,7,1,6,3,0",  "-v=1", "-warmup=0", "-repeat=1"}
    };

    return !run_test_cases<ck_tile::fp8_t>(test_cases);
}
