// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <sstream>
#include <hip/hip_runtime.h>

#define HIP_CHECK_ERROR(retval_or_funcall)                                                   \
    do                                                                                       \
    {                                                                                        \
        hipError_t _tmpVal = retval_or_funcall;                                              \
        if(_tmpVal != hipSuccess)                                                            \
        {                                                                                    \
            std::ostringstream ostr;                                                         \
            ostr << "HIP Function Failed ("                                                  \
                 << "file: " << __FILE__ << ":" << __LINE__ << " func: " << __func__ << ") " \
                 << hipGetErrorString(_tmpVal);                                              \
            throw std::runtime_error(ostr.str());                                            \
        }                                                                                    \
    } while(0)

#define hip_check_error(retval) HIP_CHECK_ERROR(retval)
