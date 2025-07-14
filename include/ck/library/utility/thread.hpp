// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifdef __linux__
#include <sched.h>
#endif
#include <thread>
namespace ck {
inline unsigned int get_available_cpu_cores()
{
#if defined(__linux__)
    cpu_set_t cpu_set;
    if(sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set) == 0)
    {
        unsigned int cpu_count = CPU_COUNT(&cpu_set);
        if(cpu_count > 0)
            return cpu_count;
    }
#endif
    // Fallback if sched_getaffinity unavailable or fails
    return std::thread::hardware_concurrency();
}
} // namespace ck
