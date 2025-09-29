// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

struct StreamConfig
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 5;
    int nrepeat_           = 50;

    bool flush_cache   = false;
    int rotating_count = 1;

    bool use_time_based_benchmark_ = false;
    float cold_bench_time_secs_ = 1.0f;  // Default 1 second for warm-up
    float hot_bench_time_secs_ = 2.0f;   // Default 2 seconds for measurement
};
