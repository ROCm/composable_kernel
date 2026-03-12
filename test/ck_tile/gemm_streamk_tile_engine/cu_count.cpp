// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <iostream>

/**
 * @brief Determines whether a `hipError` is present in the given `error_status`
 * @return true if the `error_status` has an error, otherwise false.
 */
bool has_error(const hipError_t& error_status)
{
    if(error_status != hipSuccess)
    {
        std::cerr << hipGetErrorString(error_status);
        return true;
    }

    return false;
}

/**
 * @brief Returns the number of Compute Units (CUs) on the given device.
 * @return The number of CUs on the device. If an error occurs while querying the device, zero is
 * returned.
 */
int get_cu_count()
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;

    const hipError_t device_status = hipGetDevice(&dev);

    if(has_error(device_status))
        return 0;

    const hipError_t prop_status = hipGetDeviceProperties(&dev_prop, dev);
    if(has_error(prop_status))
        return 0;

    return dev_prop.multiProcessorCount;
}

int main()
{

    std::cout << get_cu_count();

    return 0;
}
