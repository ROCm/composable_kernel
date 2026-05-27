// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {

//
// cluster_multicast_load: Load from global memory into VGPRs with cluster multicast.
// The data is broadcast to all WGPs whose bit is set in the participation mask.
// Only available on gfx1250+.
//

#if defined(__gfx125__)

namespace detail {
template <typename T>
__device__ __attribute__((address_space(1))) T* to_global(const T* ptr)
{
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-qual"
#endif
    return (__attribute__((address_space(1))) T*)(ptr);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}
} // namespace detail

template <typename T>
__device__ T cluster_multicast_load(const T* global_ptr, int mask)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16,
                  "cluster_multicast_load requires 4, 8, or 16 byte type");

    if constexpr(sizeof(T) == 4)
    {
        return bit_cast<T>(__builtin_amdgcn_cluster_load_b32(
            detail::to_global<int>(reinterpret_cast<const int*>(global_ptr)), 0, mask));
    }
    else if constexpr(sizeof(T) == 8)
    {
        using vec2i_t = __attribute__((vector_size(8))) int;
        return bit_cast<T>(__builtin_amdgcn_cluster_load_b64(
            detail::to_global<vec2i_t>(reinterpret_cast<const vec2i_t*>(global_ptr)), 0, mask));
    }
    else if constexpr(sizeof(T) == 16)
    {
        using vec4i_t = __attribute__((vector_size(16))) int;
        return bit_cast<T>(__builtin_amdgcn_cluster_load_b128(
            detail::to_global<vec4i_t>(reinterpret_cast<const vec4i_t*>(global_ptr)), 0, mask));
    }
}

#else

template <typename T>
__device__ T cluster_multicast_load(const T* global_ptr, int mask)
{
    // Non-gfx1250: fallback to plain load
    (void)mask;
    return *global_ptr;
}

#endif

//
// cluster_load_async: Async load from global memory directly into LDS with cluster multicast.
// Data bypasses VGPRs and lands directly in LDS. Requires explicit fence
// (cluster_load_async_wait) before consuming the LDS data.
// Only available on gfx1250+.
//

#if defined(__gfx125__)

template <int NumBytes, index_t inst_offset = 0>
__device__ void cluster_load_async(__attribute__((address_space(3))) void* lds_ptr,
                                   __attribute__((address_space(1))) const void* global_ptr,
                                   int mask)
{
    static_assert(NumBytes == 1 || NumBytes == 4 || NumBytes == 8 || NumBytes == 16,
                  "NumBytes must be 1, 4, 8, or 16");

    if constexpr(NumBytes == 1)
    {
        __builtin_amdgcn_cluster_load_async_to_lds_b8(
            const_cast<__attribute__((address_space(1))) char*>(
                static_cast<__attribute__((address_space(1))) const char*>(global_ptr)),
            static_cast<__attribute__((address_space(3))) char*>(lds_ptr),
            inst_offset,
            0,
            mask);
    }
    else if constexpr(NumBytes == 4)
    {
        __builtin_amdgcn_cluster_load_async_to_lds_b32(
            const_cast<__attribute__((address_space(1))) int*>(
                static_cast<__attribute__((address_space(1))) const int*>(global_ptr)),
            static_cast<__attribute__((address_space(3))) int*>(lds_ptr),
            inst_offset,
            0,
            mask);
    }
    else if constexpr(NumBytes == 8)
    {
        using cluster_int32x2_t = int __attribute__((ext_vector_type(2)));
        __builtin_amdgcn_cluster_load_async_to_lds_b64(
            const_cast<__attribute__((address_space(1))) cluster_int32x2_t*>(
                static_cast<__attribute__((address_space(1))) const cluster_int32x2_t*>(
                    global_ptr)),
            static_cast<__attribute__((address_space(3))) cluster_int32x2_t*>(lds_ptr),
            inst_offset,
            0,
            mask);
    }
    else if constexpr(NumBytes == 16)
    {
        using cluster_int32x4_t = int __attribute__((ext_vector_type(4)));
        __builtin_amdgcn_cluster_load_async_to_lds_b128(
            const_cast<__attribute__((address_space(1))) cluster_int32x4_t*>(
                static_cast<__attribute__((address_space(1))) const cluster_int32x4_t*>(
                    global_ptr)),
            static_cast<__attribute__((address_space(3))) cluster_int32x4_t*>(lds_ptr),
            inst_offset,
            0,
            mask);
    }
}

//
// cluster_load_async_wait: Wait for all pending async LDS loads to complete.
// Must be called before reading from LDS after cluster_load_async.
//
__device__ inline void cluster_load_async_wait() { __builtin_amdgcn_s_wait_asynccnt(0); }

#else

template <int NumBytes, index_t inst_offset = 0>
__device__ void cluster_load_async(__attribute__((address_space(3))) void* lds_ptr,
                                   __attribute__((address_space(1))) const void* global_ptr,
                                   int mask)
{
    // Non-gfx1250: not supported
    (void)lds_ptr;
    (void)global_ptr;
    (void)mask;
}

__device__ inline void cluster_load_async_wait() {}

#endif

} // namespace ck
