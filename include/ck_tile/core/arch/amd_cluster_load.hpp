// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile {

#ifdef __gfx1250__
template <typename T>
CK_TILE_DEVICE __attribute__((address_space(1))) T* to_global(const T* ptr)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-qual"
    return (__attribute__((address_space(1))) T*)(ptr);
#pragma clang diagnostic pop
}

template <typename T>
CK_TILE_DEVICE __attribute__((address_space(3))) T* to_lds(T* ptr)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (__attribute__((address_space(3))) T*)(ptr);
#pragma clang diagnostic pop
}
#endif // __gfx1250__

// Struct specializations for CLUSTER_LOAD_B32/B64/B128.
// Primary template intentionally undefined — compile error for unsupported sizes.
template <index_t bytes>
struct cluster_load;

template <>
struct cluster_load<4>
{
    template <typename T>
    CK_TILE_DEVICE T operator()(const T* addr, int mask)
    {
        static_assert(sizeof(T) == 4, "cluster_load<4> requires a 4-byte type");
#ifdef __gfx1250__
        return ck_tile::bit_cast<T>(__builtin_amdgcn_cluster_load_b32(
            to_global<int>(reinterpret_cast<const int*>(addr)), 0, mask));
#else
        (void)addr;
        (void)mask;
        static_assert(sizeof(T) == 0, "cluster_load is only supported on gfx1250");
        return T{};
#endif
    }
};

template <>
struct cluster_load<8>
{
    template <typename T>
    CK_TILE_DEVICE T operator()(const T* addr, int mask)
    {
        static_assert(sizeof(T) == 8, "cluster_load<8> requires an 8-byte type");
#ifdef __gfx1250__
        // Builtin requires LLVM native vector, not HIP int2.
        using vec2i_t = __attribute__((vector_size(8))) int;
        return ck_tile::bit_cast<T>(__builtin_amdgcn_cluster_load_b64(
            to_global<vec2i_t>(reinterpret_cast<const vec2i_t*>(addr)), 0, mask));
#else
        (void)addr;
        (void)mask;
        static_assert(sizeof(T) == 0, "cluster_load is only supported on gfx1250");
        return T{};
#endif
    }
};

template <>
struct cluster_load<16>
{
    template <typename T>
    CK_TILE_DEVICE T operator()(const T* addr, int mask)
    {
        static_assert(sizeof(T) == 16, "cluster_load<16> requires a 16-byte type");
#ifdef __gfx1250__
        // Builtin requires LLVM native vector, not HIP int4.
        using vec4i_t = __attribute__((vector_size(16))) int;
        return ck_tile::bit_cast<T>(__builtin_amdgcn_cluster_load_b128(
            to_global<vec4i_t>(reinterpret_cast<const vec4i_t*>(addr)), 0, mask));
#else
        (void)addr;
        (void)mask;
        static_assert(sizeof(T) == 0, "cluster_load is only supported on gfx1250");
        return T{};
#endif
    }
};

template <typename T>
CK_TILE_DEVICE T cluster_multicast_load(const T* addr, int mask)
{
    return cluster_load<sizeof(T)>{}(addr, mask);
}

// ---------------------------------------------------------------------------
// CLUSTER_LOAD_ASYNC_TO_LDS_B* — async global→LDS multicast (gfx1250 only)
// ---------------------------------------------------------------------------
// Unlike CLUSTER_LOAD_B*, data lands in LDS (not VGPRs) and is tracked by
// ASYNCcnt. Wait with s_wait_asynccnt(0) on the requesting wave, then use
// a barrier before other waves in the WG read from LDS.
//
// M0[15:0] = WGP participation mask (same encoding as CLUSTER_LOAD_B*).
// M0[16]   = early-timeout flag.
// The builtin sets M0 from the `mask` argument internally.
//
// LDS destination address is supplied per-lane via the `lds_dst` VGPR.
// `lds_dst` must be an address_space(3) (LDS) pointer.
// `inst_offset` is a compile-time immediate byte offset added to `lds_dst`
// by the hardware instruction (default 0).

// Struct specializations for CLUSTER_LOAD_ASYNC_TO_LDS_B32/B64/B128.
// Primary template intentionally undefined — compile error for unsupported sizes.
template <index_t bytes, index_t inst_offset = 0>
struct cluster_load_async_to_lds;

template <index_t inst_offset>
struct cluster_load_async_to_lds<4, inst_offset>
{
    CK_TILE_DEVICE void
    operator()(const int* src, __attribute__((address_space(3))) int* lds_dst, int mask)
    {
#ifdef __gfx1250__
        __attribute__((address_space(1))) int* g_src = to_global(src);
        __builtin_amdgcn_cluster_load_async_to_lds_b32(g_src, lds_dst, inst_offset, 0, mask);
#else
        (void)src;
        (void)lds_dst;
        (void)mask;
#endif
    }
};

template <index_t inst_offset>
struct cluster_load_async_to_lds<8, inst_offset>
{
    CK_TILE_DEVICE void
    operator()(const int* src, __attribute__((address_space(3))) int* lds_dst, int mask)
    {
#ifdef __gfx1250__
        using vec2i_t = __attribute__((vector_size(8))) int;
        __attribute__((address_space(1))) vec2i_t* g_src =
            to_global(reinterpret_cast<const vec2i_t*>(src));
        __attribute__((address_space(3))) vec2i_t* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) vec2i_t*>(lds_dst);
        __builtin_amdgcn_cluster_load_async_to_lds_b64(g_src, lds_ptr, inst_offset, 0, mask);
#else
        (void)src;
        (void)lds_dst;
        (void)mask;
#endif
    }
};

template <index_t inst_offset>
struct cluster_load_async_to_lds<16, inst_offset>
{
    CK_TILE_DEVICE void
    operator()(const int* src, __attribute__((address_space(3))) int* lds_dst, int mask)
    {
#ifdef __gfx1250__
        using vec4i_t = __attribute__((vector_size(16))) int;
        __attribute__((address_space(1))) vec4i_t* g_src =
            to_global(reinterpret_cast<const vec4i_t*>(src));
        __attribute__((address_space(3))) vec4i_t* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) vec4i_t*>(lds_dst);
        __builtin_amdgcn_cluster_load_async_to_lds_b128(g_src, lds_ptr, inst_offset, 0, mask);
#else
        (void)src;
        (void)lds_dst;
        (void)mask;
#endif
    }
};

// Generic wrapper: issues CLUSTER_LOAD_ASYNC_TO_LDS_B* sized to T.
// `src`         — global source pointer (generic address space; cast to global internally)
// `lds_dst`     — per-lane LDS destination pointer (must be address_space(3))
// `mask`        — M0[15:0] WGP participation mask; M0[16] sets early-timeout
// `inst_offset` — compile-time immediate byte offset added to lds_dst by the hardware
template <typename T, index_t inst_offset = 0>
CK_TILE_DEVICE void cluster_multicast_load_async_to_lds(const T* src,
                                                        __attribute__((address_space(3)))
                                                        T* lds_dst,
                                                        int mask)
{
    cluster_load_async_to_lds<sizeof(T), inst_offset>{}(
        reinterpret_cast<const int*>(src),
        reinterpret_cast<__attribute__((address_space(3))) int*>(lds_dst),
        mask);
}

} // namespace ck_tile
