// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {
struct workgroup_barrier
{
    __device__ workgroup_barrier(uint32_t* ptr) : base_ptr(ptr) {}

    __device__ uint32_t ld(uint32_t offset)
    {
        return __atomic_load_n(base_ptr + offset, __ATOMIC_RELAXED);
    }

    __device__ void wait_eq(uint32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) != value) {}
        }
        __syncthreads();
    }

    __device__ void wait_lt(uint32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) < value) {}
        }
        __syncthreads();
    }

    __device__ void wait_set(uint32_t offset, uint32_t compare, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(atomicCAS(base_ptr + offset, compare, value) != compare) {}
        }
        __syncthreads();
    }

    // enter critical zoon, assume buffer is zero when launch kernel
    __device__ void aquire(uint32_t offset) { wait_set(offset, 0, 1); }

    // exit critical zoon, assume buffer is zero when launch kernel
    __device__ void release(uint32_t offset) { wait_set(offset, 1, 0); }

    __device__ void inc(uint32_t offset)
    {
        __syncthreads();
        if(threadIdx.x == 0)
        {
            atomicAdd(base_ptr + offset, 1);
        }
    }

    uint32_t* base_ptr;
};
} // namespace ck

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
