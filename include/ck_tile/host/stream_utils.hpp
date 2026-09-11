// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime_api.h>

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/hip_check_error.hpp"

namespace ck_tile {

static inline index_t get_available_compute_units(const stream_config&)
{
    index_t num_cu;
    hipError_t rtn;
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    rtn = hipGetDevice(&dev);
    hip_check_error(rtn);
    rtn = hipGetDeviceProperties(&dev_prop, dev);
    hip_check_error(rtn);
    num_cu = dev_prop.multiProcessorCount;
    return num_cu;
};

} // namespace ck_tile
