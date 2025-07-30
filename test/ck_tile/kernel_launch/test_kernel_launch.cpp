// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include <hip/hip_runtime.h>
#include "ck_tile/host.hpp"

using ck_tile::stream_config;
TEST(Kernel, old)
{
    stream_config s1;
    s1.time_kernel_ = true;
    s1.cold_niters_ = 1;
    s1.nrepeat_ = 2;
    ck_tile::launch_kernel(s1,              // timer for all kernel
        [=](const stream_config& s[[maybe_unused]]) { printf("0, "); },
        [=](const stream_config& s[[maybe_unused]]) { printf("1, "); },
        [=](const stream_config& s[[maybe_unused]]) { printf("2, "); }
    );
}

TEST(Kernel, new_1)
{
    stream_config s1;
    s1.time_kernel_ = true;
    s1.cold_niters_ = 2;
    s1.nrepeat_ = 2;
    ck_tile::launch_kernel<1, 2>(s1,        // timer for kernel [1, 2)
        [=](const stream_config& s[[maybe_unused]]) { printf("0, "); },
        [=](const stream_config& s[[maybe_unused]]) { printf("1, "); },
        [=](const stream_config& s[[maybe_unused]]) { printf("2, "); }
    );
}

TEST(Kernel, new_2)
{
    stream_config s1;
    s1.time_kernel_ = true;
    s1.cold_niters_ = 2;
    s1.nrepeat_ = 2;
    ck_tile::launch_kernel<1>(s1,        // timer for kernel [1, N)
        [=](const stream_config& s[[maybe_unused]]) { printf("0, "); },
        [=](const stream_config& s[[maybe_unused]]) { printf("1, "); },
        [=](const stream_config& s[[maybe_unused]]) { printf("2, "); }
    );
}
