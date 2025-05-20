// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile {

template <typename ADataType, typename BDataType>
struct RotatingMemWrapper
{
    RotatingMemWrapper() = delete;
    RotatingMemWrapper(const void* a_ptr_,
                       const void* b_ptr_,
                       std::size_t rotating_count_,
                       std::size_t size_a_,
                       std::size_t size_b_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          rotating_count(rotating_count_),
          size_a(size_a_),
          size_b(size_b_)
    {
        p_a_grids.push_back(a_ptr);
        p_b_grids.push_back(b_ptr);
        for(size_t i = 1; i < rotating_count; i++)
        {
            {
                void* pADeviceBuf;
                HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&pADeviceBuf), size_a_));
                HIP_CHECK_ERROR(hipMemcpy(static_cast<void*>(pADeviceBuf),
                                          const_cast<void*>(p_a_grids[0]),
                                          size_a_,
                                          hipMemcpyDeviceToDevice));
                p_a_grids.push_back(pADeviceBuf);
            }

            {
                void* pBDeviceBuf;
                HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&pBDeviceBuf), size_b_));
                HIP_CHECK_ERROR(hipMemcpy(static_cast<void*>(pBDeviceBuf),
                                          const_cast<void*>(p_b_grids[0]),
                                          size_b_,
                                          hipMemcpyDeviceToDevice));
                p_b_grids.push_back(pBDeviceBuf);
            }
        }
    }
    void Next()
    {
        if(rotating_count > 1)
        {
            std::size_t idx = iter++ % rotating_count;
            a_ptr           = p_a_grids[idx];
            b_ptr           = p_b_grids[idx];
        }
    }
    void Print()
    {
        std::cout << "RotatingMemWrapper: { size_a: " << size_a << ", size_b: " << size_b
                  << ", rotating_count: " << rotating_count << "}" << std::endl;
    }
    ~RotatingMemWrapper() noexcept
    {
        if(rotating_count > 1)
        {
            // restore ptr
            a_ptr = p_a_grids[0];
            b_ptr = p_b_grids[0];

            // free device mem
            for(size_t i = 1; i < rotating_count; i++)
            {
                ck_tile::hip_check_error(hipFree(const_cast<void*>(p_a_grids[i])));
                ck_tile::hip_check_error(hipFree(const_cast<void*>(p_b_grids[i])));
            }
        }
    }

    private:
    const void* a_ptr;
    const void* b_ptr;
    std::size_t iter           = 0;
    std::size_t rotating_count = 1;
    std::size_t size_a         = 0;
    std::size_t size_b         = 0;
    std::vector<const void*> p_a_grids;
    std::vector<const void*> p_b_grids;
};
inline void flush_icache()
{
    hipDeviceProp_t deviceProps;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&deviceProps, 0));
    int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

    ck_tile::flush_cache<<<dim3(gpu_block3), dim3(64), 0, nullptr>>>();
    HIP_CHECK_ERROR(hipGetLastError());
}
} // namespace ck_tile
