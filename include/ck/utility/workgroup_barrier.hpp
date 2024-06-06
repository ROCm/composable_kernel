#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace ck {
struct workgroup_barrier
{
    __device__ workgroup_barrier(uint32_t* ptr) : base_ptr(ptr) {}

    __device__ uint32_t ld(uint32_t offset) const
    {
        return __atomic_load_n(base_ptr + offset, __ATOMIC_RELAXED);
    }

    __device__ void st(uint32_t offset, uint32_t value)
    {
        __atomic_store_n(base_ptr + offset, value, __ATOMIC_RELEASE);
    }

    __device__ void wait_eq(uint32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) != value) {}
        }
        __builtin_amdgcn_s_barrier();
    }

    __device__ void wait_lt(uint32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) < value) {}
        }
        __builtin_amdgcn_s_barrier();
    }

    __device__ void wait_set(uint32_t offset, uint32_t compare, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(atomicCAS(base_ptr + offset, compare, value) != compare) {}
        }
        __builtin_amdgcn_s_barrier();
    }

    // enter critical zoon, assume buffer is zero when launch kernel
    __device__ void aquire(uint32_t offset) { wait_set(offset, 0, 1); }

    // exit critical zoon, assume buffer is zero when launch kernel
    __device__ void release(uint32_t offset) { wait_set(offset, 1, 0); }

    __device__ void inc(uint32_t offset)
    {
        __builtin_amdgcn_s_barrier();
        if(threadIdx.x == 0)
        {
            atomicAdd(base_ptr + offset, 1);
        }
    }

    __device__ void reset(uint32_t offset)
    {
        __builtin_amdgcn_s_barrier();
        if(threadIdx.x == 0)
        {
            st(offset, 0);
        }
    }

    uint32_t* base_ptr;
};
} // namespace ck
