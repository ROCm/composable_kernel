
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "ck_tile/host/hip_check_error.hpp"

#define HSTU_CHECK(COND, ERR)                  \
    if(!(COND))                                \
    {                                          \
        std::ostringstream ostr;               \
        ostr << "'" #COND "' failed: " << ERR; \
        throw std::runtime_error(ostr.str());  \
    }

static inline int get_number_of_cu()
{
    int device;

    HIP_CHECK_ERROR(hipGetDevice(&device));

    hipDeviceProp_t props;

    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));

    return props.multiProcessorCount;
}
