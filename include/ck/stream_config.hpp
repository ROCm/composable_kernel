// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef CK_NOGPU
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

struct StreamConfig
{
#ifndef CK_NOGPU
    hipStream_t stream_id_ = nullptr;
#endif
    bool time_kernel_      = false;
};
