// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

#if CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN

#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"
#include "ck_tile/core/arch/amd_wave_read_first_lane.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/ignore.hpp"
#include "ck_tile/core/arch/amd_buffer_coherence.hpp"

#define HAS_GLOBAL_ATOMIC_PK_ADD_BUILTIN                        \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_v2f16) && \
        __has_builtin(__builtin_amdgcn_global_atomic_fadd_v2bf16)

// This attribute gives a hint to the compiler that a branch is likely to be taken.
// Then, the compiler should remove if possible the associated s_cbranch_execz branch that would
// have been generated.
#if __cplusplus >= 202002L
#define LIKELY(x) (x) [[likely]]
#else
#define LIKELY(x) (__builtin_expect(!!(x), 1))
#endif

using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

namespace ck_tile {

union buffer_resource
{
    CK_TILE_DEVICE constexpr buffer_resource() : content{} {}

    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    array<void*, 2> address;
    array<uint32_t, 4> range;
    array<uint32_t, 4> config;
};

template <typename ForceSGPR = std::false_type>
CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void* ptr,
                                                   uint32_t size = 0xffffffff,
                                                   ForceSGPR     = {})
{
    buffer_resource res;
#if defined(__gfx125__)
    res.address[0] = const_cast<void*>(ptr);
    res.range[1] |= (size & 0x7f) << 25;
    res.range[2] = (size >> 7) & 0xffffffff;
#else
    res.address[0] = const_cast<void*>(ptr);
    res.range[2]   = size;
#endif
    res.config[3] = CK_TILE_BUFFER_RESOURCE_3RD_DWORD;

    if constexpr(std::is_same_v<ForceSGPR, std::true_type>)
    {
        return amd_wave_read_first_lane(res.content);
    }
    else
    {
        return res.content;
    }
}
CK_TILE_DEVICE __amdgpu_buffer_rsrc_t make_builtin_buffer_resource(const void* ptr,
                                                                   uint32_t size = 0xffffffff)
{
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(ptr), /*stride*/ 0, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD);
}

namespace impl {
// below type indicate the data type used for buffer load inline asm
// clang-format off
template<index_t N, typename T> struct buffer_load_trait;

template<typename T> struct buffer_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct buffer_load_trait<8 , T> { using payload_t = fp32x2_t; };
template<typename T> struct buffer_load_trait<4 , T> { using payload_t = float; };
template<typename T> struct buffer_load_trait<2 , T> { using payload_t = float; };
template<typename T> struct buffer_load_trait<1 , T> { using payload_t = float; };

#if CK_TILE_BUFFER_LOAD_RAW_BF16_WA
template<> struct buffer_load_trait<16, thread_buffer<bf16_t, 8>> { using payload_t = bf16x8_t; };
template<> struct buffer_load_trait<8 , thread_buffer<bf16_t, 4>> { using payload_t = bf16x4_t; };
template<> struct buffer_load_trait<4 , thread_buffer<bf16_t, 2>> { using payload_t = bf16x2_t; };
#endif
// clang-format on
} // namespace impl

// TODO: glc/slc/...
template <index_t bytes, bool pre_nop = false>
struct buffer_load;

template <index_t bytes, bool pre_nop = false>
struct buffer_load_if;

template <index_t bytes>
struct buffer_store;

template <index_t bytes>
struct buffer_store_if;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
#endif
// TODO: strict aliasing rule seems fail when reinterpret_cast between vector type
// (exp_vector_type(xxx))

#define HAS_RAW_BUFFER_BUILTINS                             \
    __has_builtin(__builtin_amdgcn_raw_buffer_load_b32) &&  \
        __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && \
        __has_builtin(__builtin_amdgcn_raw_buffer_store_b32)

#if HAS_RAW_BUFFER_BUILTINS
CK_TILE_DEVICE __amdgpu_buffer_rsrc_t cast_to_amdgpu_buffer_rsrc_t(int32x4_t res)
{
    __amdgpu_buffer_rsrc_t as_rsrc;
    static_assert(sizeof(res) == sizeof(as_rsrc) && "Size of buffer resource should match");
    memcpy(&as_rsrc, &res, sizeof(res));
    return as_rsrc;
}
#endif

#if defined(__gfx12__) || defined(__gfx11__)
#define READ_EXEC __builtin_amdgcn_read_exec_lo
#define BUFFER_NULL_OFFSET " null "
#define CMPX_LE_EXEC "v_cmpx_le_u32 "
#define RESTORE_EXEC "s_mov_b32 exec_lo "
#else
#define READ_EXEC __builtin_amdgcn_read_exec
#define BUFFER_NULL_OFFSET " 0 "
#define CMPX_LE_EXEC "v_cmpx_le_u32 exec,"
#define RESTORE_EXEC "s_mov_b64 exec "
#endif

template <bool pre_nop>
struct buffer_load<16, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = typename impl::buffer_load_trait<16, T>::payload_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b128(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dwordx4 %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dwordx4 %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<8, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = typename impl::buffer_load_trait<8, T>::payload_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b64(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dwordx2 %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dwordx2 %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<4, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::buffer_load_trait<4, T>::payload_t;

#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b32(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dword %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dword %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4); // subdword is buggy, use dword buf and convert manually
        using mbuf_t = typename impl::buffer_load_trait<2, T>::payload_t;

#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b16(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_ushort %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_ushort %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<1, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::buffer_load_trait<1, T>::payload_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b16(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_ubyte %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_ubyte %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

#if HAS_RAW_BUFFER_BUILTINS
template <index_t bytes, bool pre_nop>
struct buffer_load_if
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        if LIKELY(1 <= flag)
        {
            buffer_load<bytes, pre_nop>{}(
                value, res, v_offset, s_offset, i_offset, flag, bool_constant<pre_nop>{});
        }
    }
};
#else
template <bool pre_nop>
struct buffer_load_if<16, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = typename impl::buffer_load_trait<16, T>::payload_t;
        static_assert(sizeof(mbuf_t) == sizeof(T));
        auto saved_exec = READ_EXEC();
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n" CMPX_LE_EXEC "  1, %4\n"
                         "buffer_load_dwordx4 %0, %1, %2," BUFFER_NULL_OFFSET
                         "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                      "buffer_load_dwordx4 %0, %1, %2," BUFFER_NULL_OFFSET
                                      "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<8, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 8);
        auto saved_exec = READ_EXEC();
        using mbuf_t    = typename impl::buffer_load_trait<8, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n" CMPX_LE_EXEC "  1, %4\n"
                         "buffer_load_dwordx2 %0, %1, %2," BUFFER_NULL_OFFSET
                         "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                      "buffer_load_dwordx2 %0, %1, %2," BUFFER_NULL_OFFSET
                                      "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<4, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = READ_EXEC();
        using mbuf_t    = typename impl::buffer_load_trait<4, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n" CMPX_LE_EXEC "  1, %4\n"
                         "buffer_load_dword %0, %1, %2," BUFFER_NULL_OFFSET
                         "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                      "buffer_load_dword %0, %1, %2," BUFFER_NULL_OFFSET
                                      "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = READ_EXEC();
        using mbuf_t    = typename impl::buffer_load_trait<2, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n" CMPX_LE_EXEC "  1, %4\n"
                         "buffer_load_ushort %0, %1, %2," BUFFER_NULL_OFFSET
                         "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                      "buffer_load_ushort %0, %1, %2," BUFFER_NULL_OFFSET
                                      "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<1, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = READ_EXEC();
        using mbuf_t    = typename impl::buffer_load_trait<1, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n" CMPX_LE_EXEC "  1, %4\n"
                         "buffer_load_ubyte %0, %1, %2," BUFFER_NULL_OFFSET
                         "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                      "buffer_load_ubyte %0, %1, %2," BUFFER_NULL_OFFSET
                                      "offen offset:%3\n" RESTORE_EXEC " %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif // "-Wundefined-reinterpret-cast"
template <index_t bytes>
struct buffer_store;

template <>
struct buffer_store<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = uint32x4_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b128(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_dwordx4 %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = uint32x2_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b64(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_dwordx2 %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = uint32_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b32(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_dword %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 2);
        using mbuf_t = uint16_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b16(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_short %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                     :
                     : "v"(static_cast<index_t>(bit_cast<mbuf_t>(value))),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 1);
        using mbuf_t = uint8_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b8(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_byte %0, %1, %2," BUFFER_NULL_OFFSET "offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

#if HAS_RAW_BUFFER_BUILTINS
template <index_t bytes>
struct buffer_store_if
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        if LIKELY(1 <= flag)
        {
            buffer_store<bytes>{}(value, res, v_offset, s_offset, i_offset);
        }
    }
};
#else
template <>
struct buffer_store_if<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 16);
        auto save_exec = READ_EXEC();
        using mbuf_t   = fp32x4_t;
        asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                  "buffer_store_dwordx4 %0, %1, %2," BUFFER_NULL_OFFSET
                                  "offen offset:%3\n" RESTORE_EXEC " %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 8);
        auto save_exec = READ_EXEC();
        // TODO: ugly. rocm-6.0/6.1 seems neet bit_cast to same base type to avoid scratch
        using mbuf_t = ext_vector_t<typename T::value_type, T::size()>;
        asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                  "buffer_store_dwordx2 %0, %1, %2," BUFFER_NULL_OFFSET
                                  "offen offset:%3\n" RESTORE_EXEC " %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = READ_EXEC();
        using mbuf_t   = float;
        asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                  "buffer_store_dword %0, %1, %2," BUFFER_NULL_OFFSET
                                  "offen offset:%3\n" RESTORE_EXEC " %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 2);
        auto save_exec = READ_EXEC();
        using mbuf_t   = short;
        asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                  "buffer_store_short %0, %1, %2," BUFFER_NULL_OFFSET
                                  "offen offset:%3\n" RESTORE_EXEC " %5"
                     :
                     : "v"(static_cast<index_t>(bit_cast<mbuf_t>(value))),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = READ_EXEC();
        using mbuf_t   = float;
        asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                  "buffer_store_byte %0, %1, %2," BUFFER_NULL_OFFSET
                                  "offen offset:%3\n" RESTORE_EXEC " %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};
#endif

CK_TILE_DEVICE void buffer_load_fence(index_t cnt = 0)
{
#if defined(__gfx12__)
    asm volatile("s_wait_loadcnt %0" : : "n"(cnt) : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
#endif
}

CK_TILE_DEVICE void lds_load_fence(index_t cnt = 0)
{
#if defined(__gfx12__)
    asm volatile("s_wait_dscnt %0" : : "n"(cnt) : "memory");
#else
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
#endif
}

template <typename scalar_type, index_t N, bool pre_nop = false>
struct buffer_atomic_add_if;

template <bool pre_nop>
struct buffer_atomic_add_if<bf16_t, 2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = READ_EXEC();
        using mbuf_t   = float;
        asm volatile(CMPX_LE_EXEC "  1, %4\n"
                                  "global_atomic_pk_add_bf16 %0, %1, %2 offset:%3\n" RESTORE_EXEC
                                  " %5"
                     :
                     : "v"(v_offset),
                       "v"(bit_cast<mbuf_t>(value)),
                       "s"(res.xy),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <typename scalar_type, index_t N, bool pre_nop = false>
struct buffer_atomic_add;

template <bool pre_nop>
struct buffer_atomic_add<bf16_t, 2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag = 1*/)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = float;
        asm volatile("global_atomic_pk_add_bf16 %0, %1, %2 offset:%3"
                     :
                     : "v"(v_offset), "v"(bit_cast<mbuf_t>(value)), "s"(res.xy), "n"(i_offset)
                     : "memory");
    }
};

namespace impl {
// below type indicate the data type used for buffer load inline asm
// clang-format off
template<index_t N, typename T> struct smem_load_trait;

template<typename T> struct smem_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct smem_load_trait<8 , T> { using payload_t = fp32x2_t; };
template<typename T> struct smem_load_trait<4 , T> { using payload_t = float; };
template<typename T> struct smem_load_trait<2 , T> { using payload_t = float; };
template<typename T> struct smem_load_trait<1 , T> { using payload_t = float; };

// clang-format on
} // namespace impl

// NOTE: smem load/store no need pre_nop to make sure dependency by sw, happy :)
template <index_t>
struct smem_load;

template <>
struct smem_load<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = typename impl::smem_load_trait<16, T>::payload_t;
        asm volatile("ds_read_b128 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = typename impl::smem_load_trait<8, T>::payload_t;
        asm volatile("ds_read_b64 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::smem_load_trait<4, T>::payload_t;
        asm volatile("ds_read_b32 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4); // subdword is buggy, use dword buf and convert manually
        using mbuf_t = typename impl::smem_load_trait<1, T>::payload_t;
        asm volatile("ds_read_u16 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::smem_load_trait<1, T>::payload_t;
        asm volatile("ds_read_u8 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

// clang-format off
namespace impl{

// can't use "+v" since there could be potential extra move(read/write)
// use "v" can help remove such duplicated moves
// besides, fake this as "memory" operation to force later valu after this fence
// TODO: may have scratch (because this is memory?)
//       need to reduce extra move inside compiler
template<index_t N>
CK_TILE_DEVICE void insert_dummy_dep_per_dword(array<float, N>& b)
{
    constexpr auto kSize = remove_cvref_t<decltype(b)>::size(); 
    static_for<0, kSize, 1>{}([&](auto i){
        asm volatile(" " : : "v"(b.get(number<i>{})) : "memory");
    });
}
#if 1
// below specialization just merge size() of dwords into single section
template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<2>(array<float, 2>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<3>(array<float, 3>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<4>(array<float, 4>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<8>(array<float, 8>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<16>(array<float, 16>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})),
                         "v"(b.get(number<8>{})), "v"(b.get(number<9>{})), "v"(b.get(number<10>{})), "v"(b.get(number<11>{})),
                         "v"(b.get(number<12>{})), "v"(b.get(number<13>{})), "v"(b.get(number<14>{})), "v"(b.get(number<15>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<32>(array<float, 32>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})),
                         "v"(b.get(number<8>{})), "v"(b.get(number<9>{})), "v"(b.get(number<10>{})), "v"(b.get(number<11>{})),
                         "v"(b.get(number<12>{})), "v"(b.get(number<13>{})), "v"(b.get(number<14>{})), "v"(b.get(number<15>{})),
                         "v"(b.get(number<16>{})), "v"(b.get(number<17>{})), "v"(b.get(number<18>{})), "v"(b.get(number<19>{})),
                         "v"(b.get(number<20>{})), "v"(b.get(number<21>{})), "v"(b.get(number<22>{})), "v"(b.get(number<23>{})),
                         "v"(b.get(number<24>{})), "v"(b.get(number<25>{})), "v"(b.get(number<26>{})), "v"(b.get(number<27>{})),
                         "v"(b.get(number<28>{})), "v"(b.get(number<29>{})), "v"(b.get(number<30>{})), "v"(b.get(number<31>{})) : "memory");
}
#endif
CK_TILE_DEVICE void insert_dummy_dep() {}

template<typename T>
CK_TILE_DEVICE void insert_dummy_dep(T & buffer)
{
    // TODO: indeed we expect T to be multiple of dword. subdword is always buggy
    using da_type = array<float, (sizeof(T) + 3) / 4>;
    auto & dummy = reinterpret_cast<da_type&>(buffer);
    insert_dummy_dep_per_dword(dummy);
}

template<typename Tx, typename... Ty>
CK_TILE_DEVICE void insert_dummy_dep(Tx& bx, Ty&... by)
{
    insert_dummy_dep(bx);
    insert_dummy_dep(by...);
}
}
// clang-format on
template <typename... T>
CK_TILE_DEVICE void buffer_load_fence(index_t cnt = 0, T&... o)
{
#if defined(__gfx12__)
    asm volatile("s_wait_loadcnt %0" : : "n"(cnt) : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
#endif
    impl::insert_dummy_dep(o...);
}

CK_TILE_DEVICE void buffer_store_fence(index_t cnt = 0)
{
#if defined(__gfx12__)
    asm volatile("s_wait_storecnt %0" : : "n"(cnt) : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
#endif
}

CK_TILE_DEVICE auto async_load_fence_raw(index_t cnt = 0)
{
#if defined(__gfx125__)
    asm volatile("s_wait_asynccnt %0" : : "n"(cnt) : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
#endif
}

// buffer load i8
CK_TILE_DEVICE_EXTERN int8_t
llvm_amdgcn_raw_buffer_load_i8(int32x4_t srsrc,
                               index_t voffset,
                               index_t soffset,
                               index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8.v4i32");

CK_TILE_DEVICE_EXTERN int8x2_t
llvm_amdgcn_raw_buffer_load_i8x2(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i8.v4i32");

CK_TILE_DEVICE_EXTERN int8x4_t
llvm_amdgcn_raw_buffer_load_i8x4(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i8.v4i32");

// buffer load i16
CK_TILE_DEVICE_EXTERN int16_t
llvm_amdgcn_raw_buffer_load_i16(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16.v4i32");

CK_TILE_DEVICE_EXTERN int16x2_t
llvm_amdgcn_raw_buffer_load_i16x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i16.v4i32");

CK_TILE_DEVICE_EXTERN int16x4_t
llvm_amdgcn_raw_buffer_load_i16x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i16.v4i32");

// buffer load i32
CK_TILE_DEVICE_EXTERN int32_t
llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32.v4i32");

CK_TILE_DEVICE_EXTERN int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32.v4i32");

// dwordx3 - use union to convert between int32x3 and fp16/bf16 types
union dwordx3_union
{
    int32_t as_i32[3];
    fp16_t as_fp16[6];
    bf16_t as_bf16[6];
};

CK_TILE_DEVICE_EXTERN int32x3_t
llvm_amdgcn_raw_buffer_load_i32x3(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v3i32.v4i32");

CK_TILE_DEVICE_EXTERN int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32.v4i32");

// buffer load fp16
CK_TILE_DEVICE_EXTERN _Float16
llvm_amdgcn_raw_buffer_load_fp16(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16.v4i32");

CK_TILE_DEVICE_EXTERN fp16x2_t llvm_amdgcn_raw_buffer_load_fp16x2(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16.v4i32");

CK_TILE_DEVICE_EXTERN fp16x4_t llvm_amdgcn_raw_buffer_load_fp16x4(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f16.v4i32");

// buffer load fp32
CK_TILE_DEVICE_EXTERN float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32.v4i32");

CK_TILE_DEVICE_EXTERN fp32x2_t llvm_amdgcn_raw_buffer_load_fp32x2(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32.v4i32");

CK_TILE_DEVICE_EXTERN fp32x4_t llvm_amdgcn_raw_buffer_load_fp32x4(
    int32x4_t srsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32.v4i32");

// buffer store i8
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8(int8_t vdata,
                                int32x4_t rsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i8.v4i32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8x2(int8x2_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i8.v4i32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8x4(int8x4_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i8.v4i32");

// buffer store i16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i16(int16_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_i16x2(
    int16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_i16x4(
    int16x4_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16.v4i32");

// buffer store i32
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32(int32_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i32.v4i32");

// buffer store ui16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_ui16(uint16_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_ui16x2(
    uint16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_ui16x4(
    uint16x4_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_i32x2(
    int32x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i32.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_i32x3_(
    int32x3_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v3i32.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_i32x3(dwordx3_union vdata,
                                                              int32x4_t rsrc,
                                                              index_t voffset,
                                                              index_t soffset)
{
    int32x3_t v_reg;
    v_reg[0] = vdata.as_i32[0];
    v_reg[1] = vdata.as_i32[1];
    v_reg[2] = vdata.as_i32[2];
    llvm_amdgcn_raw_buffer_store_i32x3_(v_reg, rsrc, voffset, soffset, 0);
};

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_i32x4(
    int32x4_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32.v4i32");

// buffer store fp16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp16(_Float16 vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_fp16x2(
    fp16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f16.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_fp16x4(
    fp16x4_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f16.v4i32");

// buffer store fp32
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_fp32x2(
    fp32x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32.v4i32");

CK_TILE_DEVICE_EXTERN void llvm_amdgcn_raw_buffer_store_fp32x4(
    fp32x4_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32.v4i32");

// buffer atomic-add fp16
CK_TILE_DEVICE_EXTERN fp16x2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    fp16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16.v4i32");

// buffer atomic-add bf16
// TODO: Replace with bf16x2_t, but llvm builins only accept cktile_bf16x2_t now.
CK_TILE_DEVICE_EXTERN bf16x2_t llvm_amdgcn_raw_buffer_atomic_add_bf16x2(
    bf16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2bf16.v4i32");

// buffer atomic-add i32
CK_TILE_DEVICE_EXTERN int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32.v4i32");

// buffer atomic-add fp32
CK_TILE_DEVICE_EXTERN float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32.v4i32");

// buffer atomic-max fp64
CK_TILE_DEVICE_EXTERN double llvm_amdgcn_raw_buffer_atomic_max_fp64(
    double vdata,
    int32x4_t rsrc, // dst_wave_buffer_resource
    int voffset,    // dst_thread_addr_offset
    int soffset,    // dst_wave_addr_offset
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64.v4i32");

// Direct loads from global to LDS.
#if __clang_major__ >= 21 && __clang_major__ < 23
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds.v4i32");
#else
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
#endif

template <unsigned num_dwords, bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dwordxn_v(void* smem,
                                                int32x4_t rsrc,
                                                index_t voffset,
                                                index_t /*soffset*/,
                                                index_t ioffset /*max 0xFFF*/,
                                                index_t /*flag*/       = 0,
                                                bool_constant<pre_nop> = {})
{
#define CK_TILE_ASYNC_LOAD_WITH_INSTR(instr)                            \
    if constexpr(pre_nop)                                               \
        asm volatile("s_nop 4\n" instr " %1, %2, 0 offen offset:%3 lds" \
                     : "=r"(smem) /*dummy dependency for smem*/         \
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)            \
                     : "memory");                                       \
    else                                                                \
        asm volatile(instr " %1, %2, 0 offen offset:%3 lds"             \
                     : "=r"(smem) /*dummy dependency for smem*/         \
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)            \
                     : "memory");

    if constexpr(num_dwords == 1)
    {
        CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dword");
    }
#if defined(__gfx950__)
    else if constexpr(num_dwords == 3)
    {
        CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dwordx3");
    }
    else if constexpr(num_dwords == 4)
    {
        CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dwordx4");
    }
#endif
    else
    {
        static_assert(false, "wrong! not implemented data width");
    }
#undef CK_TILE_ASYNC_LOAD_WITH_INSTR
}

CK_TILE_DEVICE void async_buffer_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// Flat async load from global memory to LDS using 64-bit global addressing.
// Bypasses the SRD's 32-bit offset limit; required when the KV cache exceeds
// INT32_MAX (2GB) byte offset on the SRD voffset path.
//
// !!! M0 PRECONDITION - IMPLICIT INPUT NOT VISIBLE IN OPERAND LIST !!!
//
//   The LDS destination address is taken from M0 (per AMD CDNA3 ISA §10.3:
//   `LDS_ADDR = LDSbase + LDSoffset(M0[17:2] * 4) + INST.OFFSET + ThreadID*4`).
//   M0 does NOT appear as an operand of these instructions or of the inline
//   asm below - the compiler cannot see the dependency. Caller must:
//
//     1. Initialize M0 once before the load loop:
//          `m0_set_with_memory(amd_wave_read_first_lane(lds_byte_offset));`
//        M0 is SALU-only - `m0_set_with_memory` uses an "s" constraint to
//        enforce this. Direct VALU writes to M0 are illegal.
//
//     2. Advance M0 between successive issues:
//          `m0_inc_with_memory(size_per_issue);`
//        `size_per_issue` MUST be a multiple of 4 - GLOBAL/FLAT LDS path
//        only honors M0[17:2]*4 (dword-aligned), so low 2 bits are silently
//        dropped (NOTE: this differs from MUBUF buffer_load_lds which uses
//        M0[15:0] as a raw byte offset).
//
//     3. Never bundle `m0_inc_with_memory` and the next call to this
//        function into a single inline asm. The compiler auto-inserts a
//        hazard NOP between an SALU write to M0 and the consuming
//        `global_load_lds_*`; bundling bypasses that and may read stale M0.
//
//   The "memory" clobber on this asm is load-bearing: it prevents the
//   compiler from reordering this load across other M0-touching helpers
//   (`m0_set_with_memory` / `m0_inc_with_memory`, also "memory"-clobbered).
//
// Verified instruction emission (HIP 6.4 / clang 19, gfx942 + gfx950):
//   `global_load_lds_dwordx4` is a single instruction (encoding 0xDDF48000
//   0x007F0000), NOT software-expanded into 4x dword. Same encoding on both
//   arches. The opcode is undocumented in CDNA3 ISA spec §13.6.2 but
//   supported by the LLVM AMDGPU backend.
//
// Available on gfx940+ (CDNA3: MI300, MI355, MI350 series).
template <unsigned num_dwords, bool pre_nop = false>
CK_TILE_DEVICE void
async_global_load_lds_dwordxn(void* smem, const void* global_addr, bool_constant<pre_nop> = {})
{
#if !defined(__gfx94__) && !defined(__gfx950__)
    static_assert(always_false_v<integral_constant<unsigned, num_dwords>>,
                  "global_load_lds requires CDNA3+ (gfx940/gfx950). "
                  "Ensure kKVLoadMode is BUFFER_LOAD on this architecture.");
#endif

    static_assert(num_dwords == 1 || num_dwords == 4,
                  "global_load_lds supports num_dwords == 1 or 4 only "
                  "(2 dwords does not exist on any supported arch; "
                  "3 dwords only on CDNA4 and unused in FMHA pipeline)");

// Inline asm: only the global address is an explicit operand. The LDS
// destination is implicit via M0 (see contract above). `"=r"(smem)` is a
// SSA scheduling anchor only - `smem` is NOT written by this asm; the
// load goes to LDS at `M0[17:2]*4 + offset:0 + ThreadID*4`.
#define CK_TILE_GLOBAL_LOAD_LDS_INSTR(instr)                                 \
    if constexpr(pre_nop)                                                    \
        asm volatile("s_nop 4\n" instr " %1, off offset:0"                   \
                     : "=r"(smem) /*scheduling anchor; real LDS dest is M0*/ \
                     : "v"(global_addr)                                      \
                     : "memory" /*prevents reorder across m0_{set,inc}*/);   \
    else                                                                     \
        asm volatile(instr " %1, off offset:0"                               \
                     : "=r"(smem) /*scheduling anchor; real LDS dest is M0*/ \
                     : "v"(global_addr)                                      \
                     : "memory" /*prevents reorder across m0_{set,inc}*/);

    if constexpr(num_dwords == 1)
    {
        CK_TILE_GLOBAL_LOAD_LDS_INSTR("global_load_lds_dword");
    }
    else if constexpr(num_dwords == 4)
    {
        CK_TILE_GLOBAL_LOAD_LDS_INSTR("global_load_lds_dwordx4");
    }
#undef CK_TILE_GLOBAL_LOAD_LDS_INSTR
}

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE thread_buffer<int8_t, N>
amd_buffer_load_impl_with_bytes(int32x4_t src_wave_buffer_resource,
                                index_t src_thread_addr_offset,
                                index_t src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 12 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    using rtn_type = thread_buffer<int8_t, N>;

    if constexpr(N == 1)
    {
        return bit_cast<rtn_type>(llvm_amdgcn_raw_buffer_load_i8(src_wave_buffer_resource,
                                                                 src_thread_addr_offset,
                                                                 src_wave_addr_offset,
                                                                 static_cast<index_t>(coherence)));
    }
    else if constexpr(N == 2)
    {

        int16_t tmp = llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 4)
    {
        int32_t tmp = llvm_amdgcn_raw_buffer_load_i32(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 8)
    {
        int32x2_t tmp = llvm_amdgcn_raw_buffer_load_i32x2(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 12)
    {
        auto tmp = llvm_amdgcn_raw_buffer_load_i32x3(src_wave_buffer_resource,
                                                     src_thread_addr_offset,
                                                     src_wave_addr_offset,
                                                     static_cast<index_t>(coherence));
        dwordx3_union ret;
        ret.as_i32[0] = tmp[0];
        ret.as_i32[1] = tmp[1];
        ret.as_i32[2] = tmp[2];
        return bit_cast<rtn_type>(ret);
    }
    else if constexpr(N == 16)
    {
        int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 32)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        thread_buffer<int32_t, 8> tmp;

        tmp.template get_as<int32x4_t>()(number<0>{}) = tmp0;
        tmp.template get_as<int32x4_t>()(number<1>{}) = tmp1;

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 64)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp2 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 8 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp3 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 12 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));

        thread_buffer<int32_t, 16> tmp;

        tmp.template get_as<int32x4_t>()(number<0>{}) = tmp0;
        tmp.template get_as<int32x4_t>()(number<1>{}) = tmp1;
        tmp.template get_as<int32x4_t>()(number<2>{}) = tmp2;
        tmp.template get_as<int32x4_t>()(number<3>{}) = tmp3;

        return bit_cast<rtn_type>(tmp);
    }
}

#ifndef BUFFER_LOAD_USE_INLINEASM
#define BUFFER_LOAD_USE_INLINEASM 0
#endif

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE thread_buffer<T, N> amd_buffer_load_impl(int32x4_t src_wave_buffer_resource,
                                                        index_t src_thread_addr_offset,
                                                        index_t src_wave_addr_offset)
{
    static_assert(
        (std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 6 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, bf16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 6 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, int32_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 12 || N == 16)) ||
            (std::is_same<T, uint8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 12 || N == 16)) ||
            (std::is_same<T, e8m0_bexp_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, pk_fp4_raw_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, pk_int4_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, pk_fp4_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, pk_fp6x16_t>::value && (N == 1)),
        "wrong! not implemented");

    using rtn_type = thread_buffer<T, N>;

    if constexpr(std::is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset,
                                                 static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32x2(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 8)
        {
            thread_buffer<float, 8> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp;
        }
        else if constexpr(N == 16)
        {
            thread_buffer<float, 16> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<2>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 8 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<3>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 12 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp;
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset,
                                                 static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16x2(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 6)
        {
            // N = 6: load as dwordx3 (12 bytes = 6 fp16), using buffer_load_dwordx3 instruction
            int32x3_t tmp_i32x3 =
                llvm_amdgcn_raw_buffer_load_i32x3(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence));

            // Use union to reinterpret int32x3 as fp16x6
            dwordx3_union tmp_union;
            tmp_union.as_i32[0] = tmp_i32x3[0];
            tmp_union.as_i32[1] = tmp_i32x3[1];
            tmp_union.as_i32[2] = tmp_i32x3[2];

            thread_buffer<fp16_t, N> result;
            static_for<0, N, 1>{}([&](auto i) { result[i] = tmp_union.as_fp16[i]; });

            return result;
        }
        else
        {
            // N >= 8: build from fp32x4 chunks
            thread_buffer<float, N / 2> tmp;

            static_for<0, (N / 8), 1>{}([&](auto i) {
                constexpr index_t chunk            = i;
                tmp.template get_as<fp32x4_t>()(i) = llvm_amdgcn_raw_buffer_load_fp32x4(
                    src_wave_buffer_resource,
                    src_thread_addr_offset,
                    src_wave_addr_offset + (chunk * 4) * sizeof(float),
                    static_cast<index_t>(coherence));
            });
            return bit_cast<rtn_type>(tmp);
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                src_wave_addr_offset,
                                                static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16x2(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16x4(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 6)
        {
            // N = 6: load as dwordx3 (12 bytes = 6 bf16), using buffer_load_dwordx3 instruction
            int32x3_t tmp_i32x3 =
                llvm_amdgcn_raw_buffer_load_i32x3(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence));

            // Use union to reinterpret int32x3 as bf16x6
            dwordx3_union tmp_union;
            tmp_union.as_i32[0] = tmp_i32x3[0];
            tmp_union.as_i32[1] = tmp_i32x3[1];
            tmp_union.as_i32[2] = tmp_i32x3[2];

            thread_buffer<bf16_t, N> result;
            static_for<0, N, 1>{}([&](auto i) { result[i] = tmp_union.as_bf16[i]; });

            return result;
        }
        else
        {
            // N >= 8: build from fp32x4 chunks
            thread_buffer<float, N / 2> tmp;

            static_for<0, (N / 8), 1>{}([&](auto i) {
                constexpr index_t chunk            = i;
                tmp.template get_as<fp32x4_t>()(i) = llvm_amdgcn_raw_buffer_load_fp32x4(
                    src_wave_buffer_resource,
                    src_thread_addr_offset,
                    src_wave_addr_offset + (chunk * 4) * sizeof(float),
                    static_cast<index_t>(coherence));
            });
            return bit_cast<rtn_type>(tmp);
        }
    }
    else // other datatype
    {
        auto raw_data = amd_buffer_load_impl_with_bytes<sizeof(T) * N, coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);

        return bit_cast<rtn_type>(raw_data);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw_impl(thread_buffer<T, N>& dst,
                                             int32x4_t src_wave_buffer_resource,
                                             index_t src_thread_addr_offset,
                                             index_t src_wave_addr_offset,
                                             index_t src_linear_addr_offset,
                                             index_t flag           = 0,
                                             bool_constant<pre_nop> = {})
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_load instruction");

    using type = thread_buffer<T, N>;
    if constexpr(oob_conditional_check)
    {
        buffer_load_if<sizeof(type), pre_nop>{}(dst,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                src_wave_addr_offset,
                                                src_linear_addr_offset,
                                                flag,
                                                bool_constant<pre_nop>{});
    }
    else
    {
        buffer_load<sizeof(type), pre_nop>{}(dst,
                                             src_wave_buffer_resource,
                                             src_thread_addr_offset,
                                             src_wave_addr_offset,
                                             src_linear_addr_offset,
                                             flag,
                                             bool_constant<pre_nop>{});
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_impl(T* smem,
                                               int32x4_t src_wave_buffer_resource,
                                               index_t src_thread_addr_offset,
                                               index_t src_wave_addr_offset,
                                               index_t src_immediate_addr_offset = 0,
                                               bool_constant<pre_nop>            = {})
{
    constexpr index_t num_bytes = sizeof(T) * N;
    constexpr index_t num_words = num_bytes / 4;
    static_assert(num_bytes % 4 == 0 && (num_words == 1 || num_words == 3 || num_words == 4),
                  "wrong! only support in dword, dwordx3, dwordx4");

    async_buffer_load_dwordxn_v<num_words>(smem,
                                           src_wave_buffer_resource,
                                           src_thread_addr_offset,
                                           src_wave_addr_offset,
                                           src_immediate_addr_offset,
                                           0,
                                           bool_constant<pre_nop>{});
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          index_t IMM                         = 0>
CK_TILE_DEVICE void amd_async_buffer_load(CK_TILE_LDS_ADDR T* smem,
                                          const __amdgpu_buffer_rsrc_t rsrc,
                                          index_t src_thread_addr_offset,
                                          index_t src_wave_addr_offset              = 0,
                                          number<IMM> /*src_immediate_addr_offset*/ = {},
                                          index_t flag                              = 0,
                                          bool_constant<oob_conditional_check>      = {})
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(IMM < (1 << 12), "wrong! immediate offset too large");

#if defined(__gfx950__)
    static_assert(bytes == 4 || bytes == 12 || bytes == 16,
                  "wrong! only support in dword, dwordx3, dwordx4");
#else
    static_assert(bytes == 4, "wrong! not implemented vector size");
#endif

    // Set up v_offset:
    index_t v_offset = src_thread_addr_offset;
    if constexpr(oob_conditional_check)
        v_offset = flag ? v_offset : 0x7fffffff; // large offset to cause OOB access

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
    // Use C-style cast to change address space without dropping llvm noalias attribute
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc,
                                             smem,
                                             bytes,
                                             v_offset,
                                             src_wave_addr_offset,
                                             /*imm*/ IMM,
                                             static_cast<index_t>(coherence));
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_buffer_store_impl_with_bytes(const thread_buffer<int8_t, N> src_thread_data,
                                                     int32x4_t dst_wave_buffer_resource,
                                                     index_t dst_thread_addr_offset,
                                                     index_t dst_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 12 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        llvm_amdgcn_raw_buffer_store_i8(bit_cast<int8_t>(src_thread_data),
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 4)
    {
        llvm_amdgcn_raw_buffer_store_i32(bit_cast<int32_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 8)
    {
        llvm_amdgcn_raw_buffer_store_i32x2(bit_cast<int32x2_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 12)
    {
        llvm_amdgcn_raw_buffer_store_i32x3(bit_cast<dwordx3_union>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset);
    }
    else if constexpr(N == 16)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(bit_cast<int32x4_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 32)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<0>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<1>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 4,
            static_cast<index_t>(coherence));
    }
    else if constexpr(N == 64)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<0>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<1>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 4,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<2>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 8,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<3>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 12,
            static_cast<index_t>(coherence));
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_buffer_store_impl(const thread_buffer<T, N> src_thread_data,
                                          int32x4_t dst_wave_buffer_resource,
                                          index_t dst_thread_addr_offset,
                                          index_t dst_wave_addr_offset)
{
    static_assert(
        (std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int32_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 12 || N == 16)) ||
            (std::is_same<T, uint16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, uint8_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            std::is_same<T, pk_fp6x16_t>::value && (N == 1),
        "wrong! not implemented");

    if constexpr(std::is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<fp32x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(
                src_thread_data.template get_as<fp32x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));
            llvm_amdgcn_raw_buffer_store_fp32x4(
                src_thread_data.template get_as<fp32x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(float),
                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp16(bit_cast<_Float16>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp16x2(bit_cast<fp16x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp16x4(bit_cast<fp16x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<fp32x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_i16x2(bit_cast<int16x2_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(bit_cast<int16x4_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(
                src_thread_data.template get_as<int16x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_i16x4(
                src_thread_data.template get_as<int16x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(bf16_t),
                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(std::is_same<T, uint16_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_ui16(bit_cast<uint16_t>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_ui16x2(bit_cast<uint16x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_ui16x4(bit_cast<uint16x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_ui16x4(
                src_thread_data.template get_as<uint16x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_ui16x4(
                src_thread_data.template get_as<uint16x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(uint16_t),
                static_cast<index_t>(coherence));
        }
    }
    else
    {
        using r_t = thread_buffer<int8_t, sizeof(T) * N>;

        amd_buffer_store_impl_with_bytes<sizeof(T) * N, coherence>(bit_cast<r_t>(src_thread_data),
                                                                   dst_wave_buffer_resource,
                                                                   dst_thread_addr_offset,
                                                                   dst_wave_addr_offset);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store_raw_impl(const thread_buffer<T, N>& dst_thread_data,
                                              int32x4_t dst_wave_buffer_resource,
                                              index_t dst_thread_addr_offset,
                                              index_t dst_wave_addr_offset,
                                              index_t dst_linear_addr_offset,
                                              index_t is_valid_element = 1)
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_store instruction");

    using type = thread_buffer<T, N>;
    if constexpr(oob_conditional_check)
    {
        buffer_store_if<sizeof(type)>{}(dst_thread_data,
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        dst_linear_addr_offset,
                                        is_valid_element);
    }
    else
    {
        buffer_store<sizeof(type)>{}(dst_thread_data,
                                     dst_wave_buffer_resource,
                                     dst_thread_addr_offset,
                                     dst_wave_addr_offset,
                                     dst_linear_addr_offset);
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void
amd_global_atomic_add_impl([[maybe_unused]] const thread_buffer<T, N>& src_thread_data,
                           [[maybe_unused]] T* addr)
{
    static_assert((std::is_same<T, ck_tile::bf16_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, ck_tile::fp16_t>::value && (N == 2 || N == 4 || N == 8)),
                  "wrong! not implemented");

#if HAS_GLOBAL_ATOMIC_PK_ADD_BUILTIN
    if constexpr(std::is_same<T, ck_tile::bf16_t>::value)
    {
        static_for<0, N / 2, 1>{}([&](auto i) {
            __builtin_amdgcn_global_atomic_fadd_v2bf16(
                bit_cast<ck_tile::bf16x2_t*>(addr) + i,
                src_thread_data.template get_as<ck_tile::bf16x2_t>()[i]);
        });
    }
    else
    {
        static_assert(false, "Not supported!");
    }
#else
    static_assert(false, "Not supported!");
#endif
}

template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_add_impl(const thread_buffer<T, N>& src_thread_data,
                                               int32x4_t dst_wave_buffer_resource,
                                               index_t dst_thread_addr_offset,
                                               index_t dst_wave_addr_offset)
{
    static_assert((std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, fp16_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4))
#if defined(__gfx950__)
                      || (std::is_same<T, bf16_t>::value && (N == 2 || N == 4 || N == 8))
#endif
                      ,
                  "wrong! not implemented");

    if constexpr(std::is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(bit_cast<float>(src_thread_data),
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else
        {
            static_for<0, N, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp32(src_thread_data.template get_as<float>()[i],
                                                       dst_wave_buffer_resource,
                                                       dst_thread_addr_offset,
                                                       dst_wave_addr_offset + i * sizeof(float),
                                                       0);
            });
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp16x2(bit_cast<fp16x2_t>(src_thread_data),
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else
        {
            static_for<0, N / 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
                    src_thread_data.template get_as<fp16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(fp16x2_t),
                    0);
            });
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_bf16x2(bit_cast<bf16x2_t>(src_thread_data),
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else
        {
            static_for<0, N / 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_bf16x2(
                    src_thread_data.template get_as<bf16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(bf16x2_t),
                    0);
            });
        }
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(bit_cast<int32_t>(src_thread_data),
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);
        }
        else
        {
            static_for<0, N, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_i32(src_thread_data.template get_as<int32_t>()[i],
                                                      dst_wave_buffer_resource,
                                                      dst_thread_addr_offset,
                                                      dst_wave_addr_offset + i * sizeof(int32_t),
                                                      0);
            });
        }
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_max_impl(const thread_buffer<T, N> src_thread_data,
                                               int32x4_t dst_wave_buffer_resource,
                                               index_t dst_thread_addr_offset,
                                               index_t dst_wave_addr_offset)
{
    static_assert((std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");
    if constexpr(std::is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(bit_cast<double>(src_thread_data),
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(double),
                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(double),
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<2>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 2 * sizeof(double),
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<3>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 3 * sizeof(double),
                0);
        }
    }
}

template <typename T>
using has_type = typename T::type;

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
//   oob_conditional_check : dynamic check if out-of-bound
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return src_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    return amd_buffer_load_impl<T, N, coherence>(
        src_wave_buffer_resource, src_addr_shift + src_thread_addr_offset, 0);
#else
    if constexpr(oob_conditional_check)
    {
        if(!src_thread_element_valid)
        {
            if constexpr(is_detected<has_type, T>::value)
            {
                // Use vector_t for not valid elements to avoid permute instructions.
                // Get raw type from structure
                using vector_t = typename T::type __attribute__((ext_vector_type(N)));
                if constexpr(sizeof(vector_t) != sizeof(typename T::type) * N)
                {
                    // Not possible to use set_as
                    return thread_buffer<T, N>{numeric<T>::zero()};
                }
                else
                {
                    thread_buffer<T, N> tmp;
                    tmp.template set_as<vector_t>(number<0>{},
                                                  vector_t{numeric<typename T::type>::zero()});
                    return tmp;
                }
            }
            else
            {
                // Use vector_t for not valid elements to avoid permute instructions.
                using vector_t = T __attribute__((ext_vector_type(N)));
                if constexpr(sizeof(vector_t) != sizeof(T) * N)
                {
                    // Not possible to use set_as
                    return thread_buffer<T, N>{numeric<T>::zero()};
                }
                else
                {
                    thread_buffer<T, N> tmp;
                    tmp.template set_as<vector_t>(number<0>{}, vector_t{numeric<T>::zero()});
                    return tmp;
                }
            }
        }
    }
    return amd_buffer_load_impl<T, N, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0);
#endif
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    if constexpr(oob_conditional_check)
    {
        if(!src_thread_element_valid)
        {
            if constexpr(is_detected<has_type, T>::value)
            {
                // Use vector_t for not valid elements to avoid permute instructions.
                // Get raw type from structure
                using vector_t = typename T::type __attribute__((ext_vector_type(N)));
                if constexpr(sizeof(vector_t) != sizeof(typename T::type) * N)
                {
                    // Not possible to use set_as
                    return thread_buffer<T, N>{customized_value};
                }
                else
                {
                    thread_buffer<T, N> tmp;
                    tmp.template set_as<vector_t>(
                        number<0>{}, vector_t{static_cast<typename T::type>(customized_value)});
                    return tmp;
                }
            }
            else
            {
                // Use vector_t for not valid elements to avoid permute instructions.
                using vector_t = T __attribute__((ext_vector_type(N)));
                if constexpr(sizeof(vector_t) != sizeof(T) * N)
                {
                    // Not possible to use set_as
                    return thread_buffer<T, N>{customized_value};
                }
                else
                {
                    thread_buffer<T, N> tmp;
                    tmp.template set_as<vector_t>(number<0>{}, vector_t{customized_value});
                    return tmp;
                }
            }
        }
    }
    return amd_buffer_load_impl<T, N, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0);
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw(thread_buffer<T, N>& dst,
                                        const T* p_src_wave,
                                        index_t src_thread_element_offset,
                                        index_t src_linear_element_offset,
                                        index_t src_element_space_size,
                                        index_t is_valid_element = 0,
                                        bool_constant<pre_nop>   = {})
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_buffer_load_raw_impl<T, N, coherence, oob_conditional_check, pre_nop>(
        dst,
        src_wave_buffer_resource,
        src_thread_addr_offset,
        0,
        src_linear_addr_offset,
        is_valid_element,
        bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw(thread_buffer<T, N>& dst,
                                        const int32x4_t src_wave_buffer_resource,
                                        index_t src_thread_element_offset,
                                        index_t src_linear_element_offset,
                                        index_t is_valid_element = 0,
                                        bool_constant<pre_nop>   = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_buffer_load_raw_impl<T, N, coherence, oob_conditional_check, pre_nop>(
        dst,
        src_wave_buffer_resource,
        src_thread_addr_offset,
        0,
        src_linear_addr_offset,
        is_valid_element,
        bool_constant<pre_nop>{});
}

// unfortunately async copy can not make sure invalid data is zero inside LDS
// ... unless people manually write zero to LDS at the proper address.
// so not support invalid_element check for now.
// buffer_load OOB still working.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob_raw(T* smem,
                                                       const T* p_src_wave,
                                                       index_t src_thread_element_offset,
                                                       index_t src_linear_element_offset,
                                                       index_t src_element_space_size,
                                                       bool_constant<pre_nop> = {})
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load_impl<T, N, coherence>(smem,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                0,
                                                src_linear_addr_offset,
                                                bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob_raw(T* smem,
                                                       const int32x4_t src_wave_buffer_resource,
                                                       index_t src_thread_element_offset,
                                                       index_t src_linear_element_offset,
                                                       bool_constant<pre_nop> = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load_impl<T, N, coherence>(smem,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                0,
                                                src_linear_addr_offset,
                                                bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = false,
          typename linear_offset_t>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob(CK_TILE_LDS_ADDR T* smem,
                                                   const __amdgpu_buffer_rsrc_t rsrc,
                                                   index_t src_thread_element_offset,
                                                   index_t src_wave_element_offset,
                                                   linear_offset_t,
                                                   bool is_valid_element,
                                                   bool_constant<oob_conditional_check> = {})
{
    index_t src_thread_addr_offset           = src_thread_element_offset * sizeof(T);
    constexpr index_t src_linear_addr_offset = static_cast<index_t>(linear_offset_t{}) * sizeof(T);
    constexpr index_t PackedSize             = numeric_traits<T>::PackedSize;
    index_t src_wave_addr_offset             = src_wave_element_offset * sizeof(T) / PackedSize;

    amd_async_buffer_load<T, N, coherence>(smem,
                                           rsrc,
                                           src_thread_addr_offset,
                                           src_wave_addr_offset,
                                           number<src_linear_addr_offset>{},
                                           is_valid_element,
                                           bool_constant<oob_conditional_check>{});
}

// this async_llvm_detail namespace is used to cast data type to llvm accepted type in builtin
// function
namespace async_llvm_detail {
template <index_t N>
struct async_load_type_traits;

template <>
struct async_load_type_traits<1>
{
    using type = char;
};
template <>
struct async_load_type_traits<4>
{
    using type = int;
};
template <>
struct async_load_type_traits<8>
{
    using type = int32x2_t;
};
template <>
struct async_load_type_traits<16>
{
    using type = int32x4_t;
};

template <index_t N>
using async_load_type_t = typename async_load_type_traits<N>::type;

template <typename TargetType, typename SourceType>
CK_TILE_DEVICE auto make_async_load_ptrs(const CK_TILE_GLOBAL_ADDR SourceType* global_ptr,
                                         CK_TILE_LDS_ADDR SourceType* smem_ptr)
{
    CK_TILE_GLOBAL_ADDR TargetType* glb_ptr = const_cast<CK_TILE_GLOBAL_ADDR TargetType*>(
        reinterpret_cast<const CK_TILE_GLOBAL_ADDR TargetType*>(global_ptr));
    CK_TILE_LDS_ADDR TargetType* lds_ptr = reinterpret_cast<CK_TILE_LDS_ADDR TargetType*>(smem_ptr);
    return ck_tile::make_tuple(glb_ptr, lds_ptr);
}
} // namespace async_llvm_detail

template <typename T,
          index_t N,
          index_t static_offset               = 0,
          bool is_uniform_global_ptr          = true,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
__device__ void amd_async_global_load_to_lds(CK_TILE_LDS_ADDR T* smem_ptr,
                                             const CK_TILE_GLOBAL_ADDR T* global_ptr,
                                             index_t global_offset,
                                             bool is_valid_element)
{
    // currently only support to b8, b32, b64, b128 when one async copy
    static_assert((std::is_same_v<T, double> && (N == 1 || N == 2)) ||
                      (std::is_same_v<T, float> && (N == 1 || N == 2 || N == 4)) ||
                      (std::is_same_v<T, int32_t> && (N == 1 || N == 2 || N == 4)) ||
                      (std::is_same_v<T, half_t> && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same_v<T, bf16_t> && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same_v<T, fp8_t> && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (std::is_same_v<T, bf8_t> && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (std::is_same_v<T, int8_t> && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (std::is_same_v<T, uint8_t> && (N == 1 || N == 4 || N == 8 || N == 16)),
                  "wrong! not implemented");

#if defined(__gfx125__)
#if CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
    constexpr bool use_asm_path = is_uniform_global_ptr;
#else
    constexpr bool use_asm_path = false;
#endif
    constexpr index_t bytes_in_instr = N * sizeof(T);
    if constexpr(bytes_in_instr == 1)
    {
        auto [glb_ptr, lds_ptr] =
            async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<1>>(
                global_ptr + global_offset - static_offset, smem_ptr);
        if(is_valid_element)
        {
            if constexpr(use_asm_path)
            {
                asm volatile(
                    "global_load_async_to_lds_b8 %0, %1, %2, offset:%3\n\t" ::"v"(
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(lds_ptr))),
                    "v"(static_cast<uint32_t>((global_offset - static_offset) * sizeof(T))),
                    "s"(reinterpret_cast<uint64_t>(global_ptr)),
                    "n"(static_cast<uint32_t>(static_offset * sizeof(T)))
                    : "memory");
            }
            else
            {
                __builtin_amdgcn_global_load_async_to_lds_b8(
                    glb_ptr, lds_ptr, static_offset, static_cast<index_t>(coherence));
            }
        }
        else
        {
            auto [unused, lds_write_ptr] =
                async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<1>>(
                    global_ptr, smem_ptr + static_offset);
            *lds_write_ptr = 0;
        }
        return;
    }
    else if constexpr(bytes_in_instr == 4)
    {
        auto [glb_ptr, lds_ptr] =
            async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<4>>(
                global_ptr + global_offset - static_offset, smem_ptr);
        if(is_valid_element)
        {
            if constexpr(use_asm_path)
            {
                asm volatile(
                    "global_load_async_to_lds_b32 %0, %1, %2, offset:%3\n\t" ::"v"(
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(lds_ptr))),
                    "v"(static_cast<uint32_t>((global_offset - static_offset) * sizeof(T))),
                    "s"(reinterpret_cast<uint64_t>(global_ptr)),
                    "n"(static_cast<uint32_t>(static_offset * sizeof(T)))
                    : "memory");
            }
            else
            {
                __builtin_amdgcn_global_load_async_to_lds_b32(
                    glb_ptr, lds_ptr, static_offset * sizeof(T), static_cast<index_t>(coherence));
            }
        }
        else
        {
            auto [unused, lds_write_ptr] =
                async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<4>>(
                    global_ptr, smem_ptr + static_offset);
            *lds_write_ptr = 0;
        }
        return;
    }
    else if constexpr(bytes_in_instr == 8)
    {
        auto [glb_ptr, lds_ptr] =
            async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<8>>(
                global_ptr + global_offset - static_offset, smem_ptr);
        if(is_valid_element)
        {
            if constexpr(use_asm_path)
            {
                asm volatile(
                    "global_load_async_to_lds_b64 %0, %1, %2, offset:%3\n\t" ::"v"(
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(lds_ptr))),
                    "v"(static_cast<uint32_t>((global_offset - static_offset) * sizeof(T))),
                    "s"(reinterpret_cast<uint64_t>(global_ptr)),
                    "n"(static_cast<uint32_t>(static_offset * sizeof(T)))
                    : "memory");
            }
            else
            {
                __builtin_amdgcn_global_load_async_to_lds_b64(
                    glb_ptr, lds_ptr, static_offset * sizeof(T), static_cast<index_t>(coherence));
            }
        }
        else
        {
            auto [unused, lds_write_ptr] =
                async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<8>>(
                    global_ptr, smem_ptr + static_offset);
            *lds_write_ptr = 0;
        }
        return;
    }
    else if constexpr(bytes_in_instr == 16)
    {
        auto [glb_ptr, lds_ptr] =
            async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<16>>(
                global_ptr + global_offset - static_offset, smem_ptr);
        if(is_valid_element)
        {
            if constexpr(use_asm_path)
            {
                asm volatile(
                    "global_load_async_to_lds_b128 %0, %1, %2, offset:%3\n\t" ::"v"(
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(lds_ptr))),
                    "v"(static_cast<uint32_t>((global_offset - static_offset) * sizeof(T))),
                    "s"(reinterpret_cast<uint64_t>(global_ptr)),
                    "n"(static_cast<uint32_t>(static_offset * sizeof(T)))
                    : "memory");
            }
            else
            {
                __builtin_amdgcn_global_load_async_to_lds_b128(
                    glb_ptr, lds_ptr, static_offset * sizeof(T), static_cast<index_t>(coherence));
            }
        }
        else
        {
            auto [unused, lds_write_ptr] =
                async_llvm_detail::make_async_load_ptrs<async_llvm_detail::async_load_type_t<16>>(
                    global_ptr, smem_ptr + static_offset);
            *lds_write_ptr = 0;
        }
        return;
    }
#else
    ignore = is_valid_element;
    ignore = global_ptr;
    ignore = smem_ptr;
    ignore = global_offset;
#endif
}

// buffer_store requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store(const thread_buffer<T, N>& src_thread_data,
                                     T* p_dst_wave,
                                     const index_t dst_thread_element_offset,
                                     const bool dst_thread_element_valid,
                                     const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return dst_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    amd_buffer_store_impl<T, N, coherence>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if constexpr(oob_conditional_check)
    {
        if(dst_thread_element_valid)
        {
            amd_buffer_store_impl<T, N, coherence>(
                src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
        }
    }
    else
    {
        amd_buffer_store_impl<T, N, coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store_raw(const thread_buffer<T, N>& src_thread_data,
                                         T* p_dst_wave,
                                         const index_t dst_thread_element_offset,
                                         const index_t dst_linear_element_offset,
                                         const bool dst_thread_element_valid,
                                         const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);
    index_t dst_linear_addr_offset = dst_linear_element_offset * sizeof(T);

    amd_buffer_store_raw_impl<T, N, coherence, oob_conditional_check>(src_thread_data,
                                                                      dst_wave_buffer_resource,
                                                                      dst_thread_addr_offset,
                                                                      0,
                                                                      dst_linear_addr_offset,
                                                                      dst_thread_element_valid);
}

// buffer_atomic_add requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_add(const thread_buffer<T, N>& src_thread_data,
                                          T* p_dst_wave,
                                          const index_t dst_thread_element_offset,
                                          const bool dst_thread_element_valid,
                                          const index_t dst_element_space_size)
{
#if defined(__gfx942__)
    if constexpr(std::is_same<T, bf16_t>::value)
    {
        if(dst_thread_element_valid)
        {
            amd_global_atomic_add_impl<T, N>(src_thread_data,
                                             p_dst_wave + dst_thread_element_offset);
        }
    }
    else
    {
#endif
        const int32x4_t dst_wave_buffer_resource =
            make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

        index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
        uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

        amd_buffer_atomic_add_impl<T, N>(
            src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_add_impl<T, N>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
#if defined(__gfx942__)
    }
#endif
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_atomic_add_raw(const thread_buffer<T, N>& src_thread_data,
                                              T* p_dst_wave,
                                              const index_t dst_thread_element_offset,
                                              const index_t dst_linear_element_offset,
                                              const bool dst_thread_element_valid,
                                              const index_t dst_element_space_size,
                                              bool_constant<pre_nop> = {})
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);
    index_t dst_linear_addr_offset = dst_linear_element_offset * sizeof(T);

    if constexpr(oob_conditional_check)
    {
        buffer_atomic_add_if<T, N, pre_nop>{}(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              0,
                                              dst_linear_addr_offset,
                                              dst_thread_element_valid);
    }
    else
    {
        buffer_atomic_add<T, N, pre_nop>{}(src_thread_data,
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           0,
                                           dst_linear_addr_offset,
                                           1);
    }
}

// buffer_atomic_max requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_max(const thread_buffer<T, N>& src_thread_data,
                                          T* p_dst_wave,
                                          const index_t dst_thread_element_offset,
                                          const bool dst_thread_element_valid,
                                          const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_max_impl<T, N>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_max_impl<T, N>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

template <typename T, index_t NumElemsPerThread>
CK_TILE_DEVICE void amd_direct_load_global_to_lds(const T* global_base_ptr,
                                                  const index_t global_offset,
                                                  T* lds_base_ptr,
                                                  const index_t lds_offset,
                                                  const bool is_valid,
                                                  const index_t src_element_space_size)
{
#if defined(__gfx9__)
    const uint32_t* global_ptr =
        reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(global_base_ptr));
    const int32x4_t src_resource =
        make_wave_buffer_resource(global_ptr, src_element_space_size * sizeof(T));
    const index_t global_offset_bytes = is_valid ? global_offset * sizeof(T) : 0x80000000;

#if CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
    T* lds_ptr              = lds_base_ptr + lds_offset;
    auto const lds_ptr_sgpr = amd_wave_read_first_lane((reinterpret_cast<uintptr_t>(lds_ptr)));
    asm volatile("s_mov_b32 m0, %0; \n\t"
                 "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
                 "v"(global_offset_bytes),
                 "s"(src_resource)
                 : "memory");
#else
    // Direct loads require that each thread reads and writes exactly a single DWORD.
    constexpr auto bytes_per_thread = sizeof(T) * NumElemsPerThread;
    // Direct loads require that each thread reads and writes a multiple of DWORDs (4 bytes).
    // For gfx950: supports 1, 3, or 4 DWORDs per thread
    // For gfx942: supports exactly 1 DWORD per thread
#if defined(__gfx950__)
    constexpr auto dword_bytes = 4;
    static_assert(bytes_per_thread == dword_bytes || bytes_per_thread == dword_bytes * 3 ||
                  bytes_per_thread == dword_bytes * 4);
#elif defined(__gfx9__)
    constexpr auto dword_bytes = 4;
    static_assert(bytes_per_thread == dword_bytes);
#endif
    // LDS pointer must be attributed with the LDS address space.
    as3_uint32_ptr lds_ptr =
        reinterpret_cast<as3_uint32_ptr>(reinterpret_cast<uintptr_t>(lds_base_ptr + lds_offset));

    llvm_amdgcn_raw_buffer_load_lds(
        src_resource, lds_ptr, bytes_per_thread, global_offset_bytes, 0, 0, 0);
#endif
#else
    ignore = global_base_ptr;
    ignore = global_offset;
    ignore = lds_base_ptr;
    ignore = lds_offset;
    ignore = is_valid;
    ignore = src_element_space_size;
#endif
}

template <typename T, index_t N>
__device__ auto amd_transpose_load_to_vgpr(const T* __restrict__ in_ptr)
{
#define __LDS_ADDR __attribute__((address_space(3)))

    static_assert(__has_builtin(__builtin_amdgcn_raw_buffer_load_b32),
                  "We need to have the compatible compiler version to build this instruction");

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
    // Use C-style cast to change address space without dropping llvm noalias attribute
    const auto in_ptr_ = (__LDS_ADDR T*)(const_cast<T*>(in_ptr));
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::half_t>)
    {
#if defined(__gfx950__)
        typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 llvm_fp16x4_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_fp16x4_t*>(in_ptr_);
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_read_tr16_b64_v4f16(lds_ptr));
#elif defined(__gfx125__)
        typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 llvm_fp16x8_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_fp16x8_t*>(in_ptr_);
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_load_tr16_b128_v8f16(lds_ptr));
#else
        static_assert(false, "amd_transpose_load_to_vgpr is not supported for this architecture");
#endif
    }
    else if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::bf16_t>)
    {
#if defined(__gfx950__)
        typedef __attribute__((__vector_size__(4 * sizeof(__bf16)))) __bf16 llvm_bf16x4_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_bf16x4_t*>(in_ptr_);
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_read_tr16_b64_v4bf16(lds_ptr));
#elif defined(__gfx125__)
        typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 llvm_bf16x8_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_bf16x8_t*>(in_ptr_);
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_load_tr16_b128_v8bf16(lds_ptr));
#else
        static_assert(false, "amd_transpose_load_to_vgpr is not supported for this architecture");
#endif
    }
    else if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::fp8_t> ||
                      std::is_same_v<remove_cvref_t<T>, ck_tile::bf8_t> ||
                      std::is_same_v<remove_cvref_t<T>, ck_tile::int8_t>)
    {
        typedef __attribute__((__vector_size__(2 * sizeof(index_t)))) index_t llvm_i32x2_t;
        auto lds_ptr =
            reinterpret_cast<__LDS_ADDR llvm_i32x2_t*>(reinterpret_cast<uintptr_t>(in_ptr));
#if defined(__gfx950__)
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_read_tr8_b64_v2i32(lds_ptr));
#elif defined(__gfx125__)
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_load_tr8_b64_v2i32(lds_ptr));
#else
        ignore = lds_ptr;
        static_assert(false, "amd_transpose_load_to_vgpr is not supported for this architecture");
#endif
    }
    else if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::pk_fp4_t>)
    {
        typedef __attribute__((__vector_size__(2 * sizeof(index_t)))) index_t llvm_i32x2_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_i32x2_t*>(in_ptr_);
#if defined(__gfx950__)
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_read_tr4_b64_v2i32(lds_ptr));
#elif defined(__gfx125__)
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_load_tr4_b64_v2i32(lds_ptr));
#else
        ignore = lds_ptr;
        static_assert(false, "amd_transpose_load_to_vgpr is not supported for this architecture");
#endif
    }
    else
    {
        static_assert(false, "not implemented");
    }
#undef __LDS_ADDR
}

template <amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          typename DataType,
          index_t TensorRank,
          bool IsGatherMode = false>
CK_TILE_DEVICE void
amd_tdm_load(const TDMDescriptor<DataType, TensorRank, IsGatherMode>& descriptor)
{
#if CK_TILE_ENABLE_TDM_FEATURE
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    if constexpr(TensorRank == 2 && !IsGatherMode)
    {
        auto tdm_desc_grp = descriptor.getResourceDescriptorGroup2();
        __builtin_amdgcn_tensor_load_to_lds_d2(
            tdm_desc_grp.get(I0), tdm_desc_grp.get(I1), static_cast<index_t>(coherence));
    }
    else
    {
        auto tdm_desc_grp = descriptor.getResourceDescriptorGroup4();
        __builtin_amdgcn_tensor_load_to_lds(tdm_desc_grp.get(I0),
                                            tdm_desc_grp.get(I1),
                                            tdm_desc_grp.get(I2),
                                            tdm_desc_grp.get(I3),
                                            static_cast<index_t>(coherence));
    }
#else
    ignore = descriptor;
#endif
}

template <amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          typename DataType,
          index_t TensorRank,
          bool IsGatherMode = false>
CK_TILE_DEVICE void
amd_tdm_store(const TDMDescriptor<DataType, TensorRank, IsGatherMode>& descriptor)
{
#if CK_TILE_ENABLE_TDM_FEATURE
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    if constexpr(TensorRank == 2 && !IsGatherMode)
    {
        auto tdm_desc_grp = descriptor.getResourceDescriptorGroup2();
        __builtin_amdgcn_tensor_store_from_lds_d2(
            tdm_desc_grp.get(I0), tdm_desc_grp.get(I1), static_cast<index_t>(coherence));
    }
    else
    {
        auto tdm_desc_grp = descriptor.getResourceDescriptorGroup4();
        __builtin_amdgcn_tensor_store_from_lds(tdm_desc_grp.get(I0),
                                               tdm_desc_grp.get(I1),
                                               tdm_desc_grp.get(I2),
                                               tdm_desc_grp.get(I3),
                                               static_cast<index_t>(coherence));
    }
#else
    ignore = descriptor;
#endif
}

} // namespace ck_tile

#endif // CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
