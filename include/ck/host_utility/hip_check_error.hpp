// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

inline void hip_check_error(hipError_t x)
{
    if(x != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << __FILE__ << ": " << __LINE__
           << "in function: " << __func__;
        throw std::runtime_error(ss.str());
    }
}

#define HIP_CHECK_ERROR(flag)                                                      \
    do                                                                             \
    {                                                                              \
        hipError_t _tmpVal;                                                        \
        if((_tmpVal = flag) != hipSuccess)                                         \
        {                                                                          \
            std::ostringstream ostr;                                               \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
                 << hipGetErrorString(_tmpVal);                                    \
            throw std::runtime_error(ostr.str());                                  \
        }                                                                          \
    } while(0)
