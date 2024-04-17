// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

namespace ck {
static __global__ void flush_icache()
{
    static_for<0, Number<1000>{}, 1>{}([](auto) {
        asm __volatile__("s_icache_inv \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t"
                         "s_nop 0 \n\t" ::
                             :);
    });
}
} // namespace ck
