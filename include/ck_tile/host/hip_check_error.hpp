// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include <sstream>
#include <stdexcept>
#include <hip/hip_runtime.h>

#if defined(__clang__) && defined(__HIP__)
namespace ck_tile {
// To be removed, which really does not tell the location of failed HIP functional call
CK_TILE_HOST void hip_check_error(hipError_t x)
{
    if(x != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << __FILE__ << ": " << __LINE__
           << "in function: " << __func__;
        throw std::runtime_error(ss.str());
    }
}
} // namespace ck_tile

#define HIP_CHECK_ERROR(retval_or_funcall)                                         \
    do                                                                             \
    {                                                                              \
        hipError_t _tmpVal = retval_or_funcall;                                    \
        if(_tmpVal != hipSuccess)                                                  \
        {                                                                          \
            std::ostringstream ostr;                                               \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
                 << hipGetErrorString(_tmpVal);                                    \
            throw std::runtime_error(ostr.str());                                  \
        }                                                                          \
    } while(0)

#else

#ifdef KL_MODEL
#include "cm9_kernel_launch.hpp"
std::string KlGetErrorString(MI_KERNEL::K_Status klerr)
{
    switch(klerr)
    {
        case MI_KERNEL::SUCCEEDED:          return "SUCCEEDED";         break;
        case MI_KERNEL::FAILED:             return "FAILED";            break;
        case MI_KERNEL::INITIALIZED:        return "INITIALIZED";       break;
        case MI_KERNEL::RELEASED:           return "RELEASED";          break;
        case MI_KERNEL::COMPILE_FAILED:     return "COMPILE_FAILED";    break;
        case MI_KERNEL::COMPILE_SUCCEEDED:  return "COMPILE_SUCCEEDED"; break;
        default:                            return "UNKNOW";            break;
    }
}

#define KL_CHECK_ERROR(retval_or_funcall)                                          \
    do                                                                             \
    {                                                                              \
        if(retval_or_funcall != MI_KERNEL::SUCCEEDED)                              \
        {                                                                          \
            std::ostringstream ostr;                                               \
            ostr << "KL Function Failed (" << __FILE__ << "," << __LINE__ << ") "  \
                 << KlGetErrorString(retval_or_funcall);                           \
            throw std::runtime_error(ostr.str());                                  \
        }                                                                          \
    } while(0)

#define HIP_CHECK_ERROR KL_CHECK_ERROR   

#endif // KL_MODEL
#endif // defined(__clang__) && defined(__HIP__)
