// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_util.hpp"
#include <string>

// Helper to determine if GPU is RDNA (gfx11xx, gfx12xx) based on runtime device properties
static bool is_rdna_arch(const char* gcn_arch)
{
    std::string arch(gcn_arch);
    return (arch.find("gfx11") == 0 || arch.find("gfx12") == 0);
}

ck_tile::index_t get_cu_count()
{
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    ck_tile::hip_check_error(hipGetDevice(&dev));
    ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));

    int mp_count = dev_prop.multiProcessorCount;

    // Check if RDNA architecture based on runtime device properties
    if(is_rdna_arch(dev_prop.gcnArchName))
    {
        // RDNA in WGP mode (default): multiProcessorCount returns WGP count
        // Each WGP contains 2 CUs, so multiply by 2 to get actual CU count
        return mp_count * 2;
    }

    // Non-RDNA architectures: multiProcessorCount directly represents CU count
    return mp_count;
}
