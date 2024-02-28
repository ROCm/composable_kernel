// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile {

template <typename T>
union buffer_resource
{
    CK_TILE_DEVICE constexpr buffer_resource() : content{} {}

    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    statically_indexed_array<T*, 2> address;
    statically_indexed_array<int32_t, 4> range;
    statically_indexed_array<int32_t, 4> config;
};

template <typename T>
CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(T* p_wave, index_t element_space_size)
{
    buffer_resource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address(number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(number<2>{}) = element_space_size * sizeof(T);
    // wavewise setting (32 bit)
    wave_buffer_resource.config(number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

template <typename T>
CK_TILE_DEVICE int32x4_t make_wave_buffer_resource_with_default_range(T* p_wave)
{
    buffer_resource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address(number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(number<2>{}) = 0xffffffff; // max possible range
    // wavewise setting (32 bit)
    wave_buffer_resource.config(number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

// TODO: glc/slc/...
template <index_t bytes>
struct buffer_load;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
// TODO: strict aliasing rule seems fail when reinterpret_cast between vector type
// (exp_vector_type(xxx))
template <>
struct buffer_load<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 0)
    {
        static_assert(sizeof(T) == 16);
        using mubuf_t = float __attribute__((ext_vector_type(4)));
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
                     : "+v"(reinterpret_cast<mubuf_t&>(value))
                     : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_load<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 0)
    {
        static_assert(sizeof(T) == 8);
        using mubuf_t = float __attribute__((ext_vector_type(2)));
        asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4"
                     : "+v"(reinterpret_cast<mubuf_t&>(value))
                     : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_load<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 0)
    {
        static_assert(sizeof(T) == 4);
        using mubuf_t = float;
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:%4"
                     : "+v"(reinterpret_cast<mubuf_t&>(value))
                     : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_load<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 0)
    {
        static_assert(sizeof(T) == 4); // subdword is buggy, use dword buf and convert manually
        using mubuf_t = float;
        asm volatile("buffer_load_ushort %0, %1, %2, %3 offen offset:%4"
                     : "+v"(reinterpret_cast<mubuf_t&>(value))
                     : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_load<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 0)
    {
        static_assert(sizeof(T) == 4);
        using mubuf_t = float;
        asm volatile("buffer_load_ubyte %0, %1, %2, %3 offen offset:%4"
                     : "+v"(reinterpret_cast<mubuf_t&>(value))
                     : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <index_t bytes>
struct buffer_load_if;

template <>
struct buffer_load_if<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 0)
    {
        static_assert(sizeof(T) == 16);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mubuf_t   = float __attribute__((ext_vector_type(4)));
        static_assert(sizeof(mubuf_t) == sizeof(T));
        asm volatile(
            "v_cmpx_le_u32 exec, 1, %5\n"
            "buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4\n"
            "s_mov_b64 exec %6"
            : "+v"(reinterpret_cast<mubuf_t&>(value))
            : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(saved_exec)
            : "memory");
    }
};

template <>
struct buffer_load_if<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 0)
    {
        static_assert(sizeof(T) == 8);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mubuf_t   = float __attribute__((ext_vector_type(2)));
        asm volatile(
            "v_cmpx_le_u32 exec, 1, %5\n"
            "buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4\n"
            "s_mov_b64 exec %6"
            : "+v"(reinterpret_cast<mubuf_t&>(value))
            : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(saved_exec)
            : "memory");
    }
};

template <>
struct buffer_load_if<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 0)
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mubuf_t   = float;
        asm volatile(
            "v_cmpx_le_u32 exec, 1, %5\n"
            "buffer_load_dword %0, %1, %2, %3 offen offset:%4\n"
            "s_mov_b64 exec %6"
            : "+v"(reinterpret_cast<mubuf_t&>(value))
            : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(saved_exec)
            : "memory");
    }
};

template <>
struct buffer_load_if<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 0)
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mubuf_t   = float;
        asm volatile(
            "v_cmpx_le_u32 exec, 1, %5\n"
            "buffer_load_ushort %0, %1, %2, %3 offen offset:%4\n"
            "s_mov_b64 exec %6"
            : "+v"(reinterpret_cast<mubuf_t&>(value))
            : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(saved_exec)
            : "memory");
    }
};

template <>
struct buffer_load_if<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 0)
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mubuf_t   = float;
        asm volatile(
            "v_cmpx_le_u32 exec, 1, %5\n"
            "buffer_load_ubyte %0, %1, %2, %3 offen offset:%4\n"
            "s_mov_b64 exec %6"
            : "+v"(reinterpret_cast<mubuf_t&>(value))
            : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(saved_exec)
            : "memory");
    }
};
#pragma clang diagnostic pop // "-Wundefined-reinterpret-cast"
template <index_t bytes>
struct buffer_store;

template <>
struct buffer_store<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 16);
        asm volatile("buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4"
                     :
                     : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 8);
        asm volatile("buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4"
                     :
                     : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 4);
        asm volatile("buffer_store_dword %0, %1, %2, %3 offen offset:%4"
                     :
                     : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 2);
        asm volatile("buffer_store_short %0, %1, %2, %3 offen offset:%4"
                     :
                     : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 1);
        asm volatile("buffer_store_byte %0, %1, %2, %3 offen offset:%4"
                     :
                     : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset)
                     : "memory");
    }
};

template <index_t bytes>
struct buffer_store_if;

template <>
struct buffer_store_if<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 16);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
                     :
                     : "v"(value),
                       "v"(v_offset),
                       "s"(res),
                       "s"(s_offset),
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
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 8);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
                     :
                     : "v"(value),
                       "v"(v_offset),
                       "s"(res),
                       "s"(s_offset),
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
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_dword %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
                     :
                     : "v"(value),
                       "v"(v_offset),
                       "s"(res),
                       "s"(s_offset),
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
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 2);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_short %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
                     :
                     : "v"(value),
                       "v"(v_offset),
                       "s"(res),
                       "s"(s_offset),
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
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 1);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_byte %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
                     :
                     : "v"(value),
                       "v"(v_offset),
                       "s"(res),
                       "s"(s_offset),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

CK_TILE_DEVICE void buffer_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// clang-format off
namespace impl{

// can't use "+v" since there could be potential extra move(read/write)
// use "v" can help remove such duplicated moves
// besides, fake this as "memory" operation to force later valu after this fence
// TODO: may have scratch (because this is memory?)
//       need to reduce extra move inside compiler
template<index_t N>
CK_TILE_DEVICE void insert_dummy_dep_per_dword(static_buffer_c<float, N>& b)
{
    for (auto i = 0; i < b.size(); i++) asm volatile(" " : : "v"(b.get(i)) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<2>(static_buffer_c<float, 2>& b)
{
    asm volatile(" " : : "v"(b.get(0)), "v"(b.get(1)) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<3>(static_buffer_c<float, 3>& b)
{
    asm volatile(" " : : "v"(b.get(0)), "v"(b.get(1)), "v"(b.get(2)) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<4>(static_buffer_c<float, 4>& b)
{
    asm volatile(" " : : "v"(b.get(0)), "v"(b.get(1)), "v"(b.get(2)), "v"(b.get(3)) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<8>(static_buffer_c<float, 8>& b)
{
    asm volatile(" " : : "v"(b.get(0)), "v"(b.get(1)), "v"(b.get(2)), "v"(b.get(3)), "v"(b.get(4)), "v"(b.get(5)), "v"(b.get(6)), "v"(b.get(7)) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<16>(static_buffer_c<float, 16>& b)
{
    asm volatile(" " : : "v"(b.get(0)), "v"(b.get(1)), "v"(b.get(2)), "v"(b.get(3)), "v"(b.get(4)), "v"(b.get(5)), "v"(b.get(6)), "v"(b.get(7)),
                       "v"(b.get(8)), "v"(b.get(9)), "v"(b.get(10)), "v"(b.get(11)), "v"(b.get(12)), "v"(b.get(13)), "v"(b.get(14)), "v"(b.get(15)) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<32>(static_buffer_c<float, 32>& b)
{
    asm volatile(" " : : "v"(b.get(0)), "v"(b.get(1)), "v"(b.get(2)), "v"(b.get(3)), "v"(b.get(4)), "v"(b.get(5)), "v"(b.get(6)), "v"(b.get(7)),
                       "v"(b.get(8)), "v"(b.get(9)), "v"(b.get(10)), "v"(b.get(11)), "v"(b.get(12)), "v"(b.get(13)), "v"(b.get(14)), "v"(b.get(15)),
                       "v"(b.get(16)), "v"(b.get(17)), "v"(b.get(18)), "v"(b.get(19)), "v"(b.get(20)), "v"(b.get(21)), "v"(b.get(22)), "v"(b.get(23)),
                       "v"(b.get(24)), "v"(b.get(25)), "v"(b.get(26)), "v"(b.get(27)), "v"(b.get(28)), "v"(b.get(29)), "v"(b.get(30)), "v"(b.get(31)) : "memory");
}

CK_TILE_DEVICE void insert_dummy_dep() {}

template<typename T>
CK_TILE_DEVICE void insert_dummy_dep(T & buffer)
{
    // TODO: indeed we expect T to be multiple of dword. subdword is always buggy
    using da_type = static_buffer_c<float, (sizeof(T) + 3) / 4>;
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
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
    impl::insert_dummy_dep(o...);
}

CK_TILE_DEVICE void buffer_store_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// buffer load i8
CK_TILE_DEVICE int8_t
llvm_amdgcn_raw_buffer_load_i8(int32x4_t srsrc,
                               index_t voffset,
                               index_t soffset,
                               index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");

CK_TILE_DEVICE int8x2_t
llvm_amdgcn_raw_buffer_load_i8x2(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i8");

CK_TILE_DEVICE int8x4_t
llvm_amdgcn_raw_buffer_load_i8x4(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i8");

// buffer load i16
CK_TILE_DEVICE bhalf_t
llvm_amdgcn_raw_buffer_load_i16(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");

CK_TILE_DEVICE bhalf2_t
llvm_amdgcn_raw_buffer_load_i16x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i16");

CK_TILE_DEVICE bhalf4_t
llvm_amdgcn_raw_buffer_load_i16x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i16");

// buffer load i32
CK_TILE_DEVICE int32_t
llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

CK_TILE_DEVICE int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

CK_TILE_DEVICE int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// buffer load fp16
CK_TILE_DEVICE half_t
llvm_amdgcn_raw_buffer_load_fp16(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

CK_TILE_DEVICE half2_t
llvm_amdgcn_raw_buffer_load_fp16x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

CK_TILE_DEVICE half4_t
llvm_amdgcn_raw_buffer_load_fp16x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f16");

// buffer load fp32
CK_TILE_DEVICE float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

CK_TILE_DEVICE float2_t
llvm_amdgcn_raw_buffer_load_fp32x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

CK_TILE_DEVICE float4_t
llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

// buffer store i8
CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i8(int8_t vdata,
                                int32x4_t rsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i8");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i8x2(int8x2_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i8");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i8x4(int8x4_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i8");

// buffer store i16
CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i16(bhalf_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i16x2(bhalf2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i16x4(bhalf4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16");

// buffer store i32
CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i32(int32_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i32");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i32x2(int32x2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i32");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

// buffer store fp16
CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_fp16(half_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_fp16x2(half2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f16");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_fp16x4(half4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f16");

// buffer store fp32
CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_fp32x2(float2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_store_fp32x4(float4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

// buffer atomic-add fp16
CK_TILE_DEVICE half2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    half2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16");

// buffer atomic-add i32
CK_TILE_DEVICE int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32");

// buffer atomic-add fp32
CK_TILE_DEVICE float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32");

// buffer atomic-max fp64
CK_TILE_DEVICE double
llvm_amdgcn_raw_buffer_atomic_max_fp64(double vdata,
                                       int32x4_t rsrc, // dst_wave_buffer_resource
                                       int voffset,    // dst_thread_addr_offset
                                       int soffset,    // dst_wave_addr_offset
                                       int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64");

CK_TILE_DEVICE void async_buffer_load_dword(void* smem,
                                            int32x4_t rsrc,
                                            index_t voffset,
                                            index_t soffset,
                                            index_t ioffset /*max 0xFFF*/,
                                            index_t /*flag*/ = 0)
{
    asm volatile("buffer_load_dword %1, %2, %3 offen offset:%4 lds"
                 : "=r"(smem) /*dummy dependency for smem*/
                 : "v"(voffset), "s"(rsrc), "s"(soffset), "n"(ioffset)
                 : "memory");
}

CK_TILE_DEVICE void async_buffer_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// memory coherency bit for buffer store/load instruction
// check ISA manual for each GFX target
// e.g. for
// https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf,
// page 67~68
enum struct amd_buffer_coherence_enum
{
    coherence_default = 0, // default value
    glc               = 1,
    slc               = 2,
    glc_slc           = 3,
};

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE typename vector_type<int8_t, N>::type
amd_buffer_load_impl_raw(int32x4_t src_wave_buffer_resource,
                         index_t src_thread_addr_offset,
                         index_t src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        return llvm_amdgcn_raw_buffer_load_i8(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset,
                                              static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        int16_t tmp = llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<int8x2_t>(tmp);
    }
    else if constexpr(N == 4)
    {
        int32_t tmp = llvm_amdgcn_raw_buffer_load_i32(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<int8x4_t>(tmp);
    }
    else if constexpr(N == 8)
    {
        int32x2_t tmp = llvm_amdgcn_raw_buffer_load_i32x2(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));

        return bit_cast<int8x8_t>(tmp);
    }
    else if constexpr(N == 16)
    {
        int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));
        return bit_cast<int8x16_t>(tmp);
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
        vector_type<int32_t, 8> tmp;

        tmp.AsType<int32x4_t>()(number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(number<1>{}) = tmp1;

        return bit_cast<int8x32_t>(tmp);
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

        vector_type<int32_t, 16> tmp;

        tmp.AsType<int32x4_t>()(number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(number<1>{}) = tmp1;
        tmp.AsType<int32x4_t>()(number<2>{}) = tmp2;
        tmp.AsType<int32x4_t>()(number<3>{}) = tmp3;

        return bit_cast<int8x64_t>(tmp);
    }
}

#ifndef BUFFER_LOAD_USE_INLINEASM
#define BUFFER_LOAD_USE_INLINEASM 0
#endif

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE typename vector_type<T, N>::type
amd_buffer_load_impl(int32x4_t src_wave_buffer_resource,
                     index_t src_thread_addr_offset,
                     index_t src_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    if constexpr(is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_fp32(src_wave_buffer_resource,
                                                    src_thread_addr_offset,
                                                    src_wave_addr_offset,
                                                    static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_fp32x2(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            vector_type<float, 8> tmp;

            tmp.AsType<float4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.AsType<float4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp.AsType<float8_t>()(number<0>{});
        }
        else if constexpr(N == 16)
        {
            vector_type<float, 16> tmp;

            tmp.AsType<float4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.AsType<float4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.AsType<float4_t>()(number<2>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 8 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.AsType<float4_t>()(number<3>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 12 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp.AsType<float16_t>()(number<0>{});
        }
    }
    else if constexpr(is_same<T, half_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_fp16(src_wave_buffer_resource,
                                                    src_thread_addr_offset,
                                                    src_wave_addr_offset,
                                                    static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_fp16x2(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_fp16x4(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            // use fp32 load to mimic fp16 load
            float4_t tmp = llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                              src_thread_addr_offset,
                                                              src_wave_addr_offset,
                                                              static_cast<index_t>(coherence));

            return bit_cast<half8_t>(tmp);
        }
    }
    else if constexpr(is_same<T, bhalf_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_i16x2(src_wave_buffer_resource,
                                                     src_thread_addr_offset,
                                                     src_wave_addr_offset,
                                                     static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_i16x4(src_wave_buffer_resource,
                                                     src_thread_addr_offset,
                                                     src_wave_addr_offset,
                                                     static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                              src_thread_addr_offset,
                                                              src_wave_addr_offset,
                                                              static_cast<index_t>(coherence));

            return bit_cast<bhalf8_t>(tmp);
        }
    }
    else // other datatype
    {
        using r_t = typename vector_type<T, N>::type;

        auto raw_data = amd_buffer_load_impl_raw<sizeof(T) * N, coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);

        return bit_cast<r_t>(raw_data);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_load_raw_impl(typename vector_type<T, N>::type& dst,
                                             int32x4_t src_wave_buffer_resource,
                                             index_t src_thread_addr_offset,
                                             index_t src_wave_addr_offset,
                                             index_t flag = 0)
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_load instruction");

    using type = typename vector_type<T, N>::type;
    if constexpr(oob_conditional_check)
    {
        buffer_load_if<sizeof(type)>{}(
            dst, src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0, flag);
    }
    else
    {
        buffer_load<sizeof(type)>{}(
            dst, src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0, flag);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_async_buffer_load_impl(T* smem,
                                               int32x4_t src_wave_buffer_resource,
                                               index_t src_thread_addr_offset,
                                               index_t src_wave_addr_offset,
                                               index_t src_immediate_addr_offset = 0)
{
    static_assert(sizeof(T) * N == 4, "wrong! not implemented vector size");

    async_buffer_load_dword(smem,
                            src_wave_buffer_resource,
                            src_thread_addr_offset,
                            src_wave_addr_offset,
                            src_immediate_addr_offset);
}

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void
amd_buffer_store_impl_raw(const typename vector_type<int8_t, N>::type src_thread_data,
                          int32x4_t dst_wave_buffer_resource,
                          index_t dst_thread_addr_offset,
                          index_t dst_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        llvm_amdgcn_raw_buffer_store_i8(src_thread_data,
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
        vector_type<int32_t, 8> tmp{bit_cast<int32x8_t>(src_thread_data)};

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[number<0>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[number<1>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 4,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 64)
    {
        vector_type<int32_t, 16> tmp{bit_cast<int32x16_t>(src_thread_data)};

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[number<0>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[number<1>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 4,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[number<2>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 8,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[number<3>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 12,
                                           static_cast<index_t>(coherence));
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_buffer_store_impl(const typename vector_type<T, N>::type src_thread_data,
                                          int32x4_t dst_wave_buffer_resource,
                                          index_t dst_thread_addr_offset,
                                          index_t dst_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    if constexpr(is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp32(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp32x2(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            vector_type<float, 8> tmp{src_thread_data};
            llvm_amdgcn_raw_buffer_store_fp32x4(tmp.AsType<float4_t>()[number<0>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
            llvm_amdgcn_raw_buffer_store_fp32x4(tmp.AsType<float4_t>()[number<1>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset + 4 * sizeof(float),
                                                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(is_same<T, half_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp16(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp16x2(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp16x4(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
#if 0
            vector_type<half_t, 8> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.AsType<half4_t>()[number<0>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.AsType<half4_t>()[number<1>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset + 4 * sizeof(half_t),
                                                static_cast<index_t>(coherence));
#else
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<float4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
#endif
        }
    }
    else if constexpr(is_same<T, bhalf_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i16(src_thread_data,
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_i16x2(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            vector_type<bhalf_t, 8> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_store_i16x4(tmp.AsType<bhalf4_t>()[number<0>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_i16x4(tmp.AsType<bhalf4_t>()[number<1>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + 4 * sizeof(bhalf_t),
                                               static_cast<index_t>(coherence));
        }
    }
    else
    {
        using r_t = typename vector_type<int8_t, sizeof(T) * N>::type;

        amd_buffer_store_impl_raw<sizeof(T) * N, coherence>(bit_cast<r_t>(src_thread_data),
                                                            dst_wave_buffer_resource,
                                                            dst_thread_addr_offset,
                                                            dst_wave_addr_offset);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void
amd_buffer_store_raw_impl(const typename vector_type<T, N>::type dst_thread_data,
                          int32x4_t dst_wave_buffer_resource,
                          index_t dst_thread_addr_offset,
                          index_t dst_wave_addr_offset,
                          index_t is_valid_element = 1)
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_store instruction");

    using type = typename vector_type<T, N>::type;
    if constexpr(oob_conditional_check)
    {
        buffer_store_if<sizeof(type)>{}(dst_thread_data,
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        0,
                                        is_valid_element);
    }
    else
    {
        buffer_store<sizeof(type)>{}(dst_thread_data,
                                     dst_wave_buffer_resource,
                                     dst_thread_addr_offset,
                                     dst_wave_addr_offset,
                                     0);
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void
amd_buffer_atomic_add_impl(const typename vector_type<T, N>::type src_thread_data,
                           int32x4_t dst_wave_buffer_resource,
                           index_t dst_thread_addr_offset,
                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");

    if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<float, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<float, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[number<3>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 3 * sizeof(float),
                                                   0);
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp16x2(src_thread_data,
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else if constexpr(N == 4)
        {
            vector_type<half_t, 4> tmp{src_thread_data};

            static_for<0, 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
        else if constexpr(N == 8)
        {
            vector_type<half_t, 8> tmp{src_thread_data};

            static_for<0, 4, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
    }
    else if constexpr(is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(src_thread_data,
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);
        }
        else if constexpr(N == 2)
        {
            vector_type<int32_t, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);
        }
        else if constexpr(N == 4)
        {
            vector_type<int32_t, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[number<2>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 2 * sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[number<3>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 3 * sizeof(int32_t),
                                                  0);
        }
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void
amd_buffer_atomic_max_impl(const typename vector_type<T, N>::type src_thread_data,
                           int32x4_t dst_wave_buffer_resource,
                           index_t dst_thread_addr_offset,
                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, double>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");
    if constexpr(is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<double, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<double, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[number<3>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 3 * sizeof(double),
                                                   0);
        }
    }
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
//   oob_conditional_check : dynamic check if out-of-bound
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return src_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    return amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_addr_shift + src_thread_addr_offset, 0);
#else
    vector_t tmp = amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0);
    if constexpr(oob_conditional_check)
        return src_thread_element_valid ? tmp : vector_t(0);
    else
        return tmp;
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
CK_TILE_DEVICE typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    vector_t tmp = amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0);

    if constexpr(oob_conditional_check)
        return src_thread_element_valid ? tmp : vector_t(customized_value);
    else
        return tmp;
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_load_raw(typename vector_type_maker<T, N>::type::type& dst,
                                        const T* p_src_wave,
                                        index_t src_thread_element_offset,
                                        index_t src_element_space_size,
                                        index_t is_valid_element = 0)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    amd_buffer_load_raw_impl<scalar_t, vector_size, coherence, oob_conditional_check>(
        dst, src_wave_buffer_resource, src_thread_addr_offset, 0, is_valid_element);
}

// unfortunately async copy can not make sure invalid data is zero inside LDS
// ... unless people manually write zero to LDS at the proper address.
// so not support invalid_element check for now.
// buffer_load OOB still working.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob(T* smem,
                                                   const T* p_src_wave,
                                                   index_t src_thread_element_offset,
                                                   index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    amd_async_buffer_load_impl<T, N, coherence>(
        smem, src_wave_buffer_resource, src_thread_addr_offset, 0, 0);
}

// buffer_store requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void
amd_buffer_store(const typename vector_type_maker<T, N>::type::type src_thread_data,
                 T* p_dst_wave,
                 const index_t dst_thread_element_offset,
                 const bool dst_thread_element_valid,
                 const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return dst_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    amd_buffer_store_impl<scalar_t, vector_size, coherence>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if constexpr(oob_conditional_check)
    {
        if(dst_thread_element_valid)
        {
            amd_buffer_store_impl<scalar_t, vector_size, coherence>(
                src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
        }
    }
    else
    {
        amd_buffer_store_impl<scalar_t, vector_size, coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void
amd_buffer_store_raw(const typename vector_type_maker<T, N>::type::type src_thread_data,
                     T* p_dst_wave,
                     const index_t dst_thread_element_offset,
                     const bool dst_thread_element_valid,
                     const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    amd_buffer_store_raw_impl<scalar_t, vector_size, coherence, oob_conditional_check>(
        src_thread_data,
        dst_wave_buffer_resource,
        dst_thread_addr_offset,
        0,
        dst_thread_element_valid);
}

// buffer_atomic_add requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void
amd_buffer_atomic_add(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_add_impl<scalar_t, vector_size>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_add_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// buffer_atomic_max requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void
amd_buffer_atomic_max(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_max_impl<scalar_t, vector_size>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_max_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// Direct loads from global to LDS.
CK_TILE_DEVICE void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

template <typename T, index_t NumElemsPerThread>
CK_TILE_DEVICE void amd_direct_load_global_to_lds(const T* global_base_ptr,
                                                  const index_t global_offset,
                                                  T* lds_base_ptr,
                                                  const index_t lds_offset,
                                                  const bool is_valid,
                                                  const index_t src_element_space_size)
{
    // Direct loads require that each thread reads and writes exactly a single DWORD.
    constexpr auto dword_bytes      = 4;
    constexpr auto bytes_per_thread = sizeof(T) * NumElemsPerThread;
    static_assert(bytes_per_thread == dword_bytes);

    const uint32_t* global_ptr =
        reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(global_base_ptr));
    const int32x4_t src_resource = make_wave_buffer_resource(global_ptr, src_element_space_size);
    const index_t global_offset_bytes = is_valid ? global_offset * sizeof(T) : 0x80000000;

#if CK_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
    T* lds_ptr = lds_base_ptr + lds_offset;
    auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(lds_ptr)));
    asm volatile("s_mov_b32 m0, %0; \n\t"
                 "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
                 "v"(global_offset_bytes),
                 "s"(src_resource));
#else
    // LDS pointer must be attributed with the LDS address space.
    __attribute__((address_space(3))) uint32_t* lds_ptr =
        reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(
            reinterpret_cast<uintptr_t>(lds_base_ptr + lds_offset));

    llvm_amdgcn_raw_buffer_load_lds(
        src_resource, lds_ptr, sizeof(uint32_t), global_offset_bytes, 0, 0, 0);
#endif
}

} // namespace ck_tile
