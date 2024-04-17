// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

struct StreamConfig
{
    hipStream_t stream_id_    = nullptr;
    bool time_kernel_         = false;
    int log_level_            = 0;
    mutable int cold_niters_  = 5;
    mutable int nrepeat_      = 50;
    mutable int time_limit_ms = 1000; // for timing MI300 runs
};
