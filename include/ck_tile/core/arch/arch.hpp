// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"

#define CK_TILE_S_CNT_MAX 0b1100'1111'0111'1111
#define CK_TILE_VMCNT(cnt)                                              \
    ([]() { static_assert(!((cnt) >> 6), "VMCNT only has 6 bits"); }(), \
     ((cnt) & 0b1111) | (((cnt) & 0b110000) << 10))
#define CK_TILE_EXPCNT(cnt) \
    ([]() { static_assert(!((cnt) >> 3), "EXP only has 3 bits"); }(), ((cnt) << 4))
#define CK_TILE_LGKMCNT(cnt) \
    ([]() { static_assert(!((cnt) >> 4), "LGKM only has 4 bits"); }(), ((cnt) << 8))

namespace ck_tile {

template <typename, bool>
struct safe_underlying_type;

template <typename T>
struct safe_underlying_type<T, true>
{
    using type = std::underlying_type_t<T>;
};

template <typename T>
struct safe_underlying_type<T, false>
{
    using type = void;
};

template <typename T>
using safe_underlying_type_t = typename safe_underlying_type<T, std::is_enum<T>::value>::type;

enum struct address_space_enum : std::uint16_t
{
    generic = 0,
    global,
    lds,
    sgpr,
    constant,
    vgpr
};

enum struct memory_operation_enum : std::uint16_t
{
    set = 0,
    atomic_add,
    atomic_max,
    add
};

CK_TILE_HOST_DEVICE constexpr index_t get_warp_size()
{
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
    return 64;
#else
    return 32;
#endif
}

CK_TILE_DEVICE index_t get_grid_size() { return gridDim.x; }

CK_TILE_DEVICE index_t get_block_size() { return blockDim.x; }

// TODO: deprecate these
CK_TILE_DEVICE index_t get_thread_local_1d_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

CK_TILE_DEVICE index_t get_block_1d_id() { return blockIdx.x; }

// Use these instead
CK_TILE_DEVICE index_t get_lane_id() { return __lane_id(); }

CK_TILE_DEVICE index_t get_warp_id()
{
    return __builtin_amdgcn_readfirstlane(threadIdx.x / get_warp_size());
}

CK_TILE_DEVICE index_t get_thread_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_block_id() { return blockIdx.x; }

CK_TILE_DEVICE void block_sync_lds()
{
#if CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
    // asm volatile("\
    // s_waitcnt lgkmcnt(0) \n \
    // s_barrier \
    // " ::);

    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
#else
    __syncthreads();
#endif
}

CK_TILE_DEVICE void block_sync_load_raw(index_t cnt = 0)
{
#ifdef __gfx12__
    asm volatile("s_wait_loadcnt %0 \n"
                 "s_barrier_signal -1 \n"
                 "s_barrier_wait -1"
                 :
                 : "n"(cnt)
                 : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0) \n"
                 "s_barrier"
                 :
                 : "n"(cnt)
                 : "memory");
#endif
}

// https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
struct waitcnt_arg
{
    // bit numbers (hex) -------------------------> FE'DC'BA98'7'654'3210
    // [V]M [E]XP [L]GKM counters and [U]NUSED ---> VV'UU'LLLL'U'EEE'VVVV
    CK_TILE_DEVICE static constexpr index_t MAX = 0b11'00'1111'0'111'1111;

    CK_TILE_DEVICE static constexpr index_t kMaxVmCnt   = 0b111111;
    CK_TILE_DEVICE static constexpr index_t kMaxExpCnt  = 0b111;
    CK_TILE_DEVICE static constexpr index_t kMaxLgkmCnt = 0b1111;

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_vmcnt()
    {
        static_assert(cnt >= 0 && !(cnt >> 6), "valid range is [0..63]");
        return MAX & ((cnt & 0b1111) | ((cnt & 0b110000) << 10));
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_expcnt()
    {
        static_assert(cnt >= 0 && !(cnt >> 3), "valid range is [0..7]");
        return MAX & (cnt << 4);
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_lgkmcnt()
    {
        static_assert(cnt >= 0 && !(cnt >> 4), "valid range is [0..15]");
        return MAX & (cnt << 8);
    }
};

template <index_t vmcnt   = waitcnt_arg::kMaxVmCnt,
          index_t expcnt  = waitcnt_arg::kMaxExpCnt,
          index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void s_waitcnt()
{
    __builtin_amdgcn_s_waitcnt(waitcnt_arg::from_vmcnt<vmcnt>() |
                               waitcnt_arg::from_expcnt<expcnt>() |
                               waitcnt_arg::from_lgkmcnt<lgkmcnt>());
}

template <index_t vmcnt   = waitcnt_arg::kMaxVmCnt,
          index_t expcnt  = waitcnt_arg::kMaxExpCnt,
          index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void s_waitcnt_barrier()
{
    s_waitcnt<vmcnt, expcnt, lgkmcnt>();
    __builtin_amdgcn_s_barrier();
}

CK_TILE_DEVICE void block_sync_lds_direct_load()
{
#if 1
    // invoke clang builtins which *should* produce the same result as the inline asm below
    // difference: inline asm is being compiled to wait vmcnt(0) after the barrier
    s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>();
#else
    // same content as in old CK (#999)
    asm volatile("\
    s_waitcnt vmcnt(0) \n \
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#endif
}

CK_TILE_DEVICE void s_nop(index_t cnt = 0)
{
#if 1
    asm volatile("s_nop %0" : : "n"(cnt) :);
#else
    __builtin_amdgcn_sched_barrier(cnt);
#endif
}

#define CK_CONSTANT_ADDRESS_SPACE \
    __attribute__((address_space( \
        static_cast<safe_underlying_type_t<address_space_enum>>(address_space_enum::constant))))

template <typename T>
__device__ T* cast_pointer_to_generic_address_space(T CK_CONSTANT_ADDRESS_SPACE* p)
{
    // cast a pointer in "Constant" address space (4) to "Generic" address space (0)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T*)(p); // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

template <typename T>
__host__ __device__ T CK_CONSTANT_ADDRESS_SPACE* cast_pointer_to_constant_address_space(T* p)
{
    // cast a pointer in "Generic" address space (0) to "Constant" address space (4)
    // only c-style pointer cast seems be able to be compiled;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T CK_CONSTANT_ADDRESS_SPACE*)p; // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

CK_TILE_HOST_DEVICE constexpr index_t get_smem_capacity()
{
#if defined(__gfx950__)
    return 163840;
#else
    return 65536;
#endif
}

// ============================================================================
// Architecture-specific parameter definitions
// We'll define all parameters for all supported architectures here.
// ============================================================================

// Parameters for gfx90x (example values, adjust as needed)
namespace Gfx90x {
constexpr ck_tile::index_t WarpTile = 64;

constexpr ck_tile::index_t VecLenFP16   = 4;
constexpr ck_tile::index_t VecLenFP32   = 4;

constexpr ck_tile::index_t kCMLane      = 4;
constexpr ck_tile::index_t kCM0PerLane  = 1;
constexpr ck_tile::index_t kCM1PerLane  = 4;
}

// Parameters for gfx120x (using a namespace for organization or just global constexpr)
namespace Gfx120x {  
constexpr ck_tile::index_t WarpTile = 32;

constexpr ck_tile::index_t VecLenFP16   = 16;
constexpr ck_tile::index_t VecLenFP32   = 8;

constexpr ck_tile::index_t kCMLane      = 2;
constexpr ck_tile::index_t kCM0PerLane  = 8;
constexpr ck_tile::index_t kCM1PerLane  = 1;
}

// Generic Parameters - should never be used in this example
// templated run function should only be instantiated for Gfx120x and Gfx90x
namespace Generic {
constexpr ck_tile::index_t WarpTile = -1;
}

// Helper to get VecLen based on GfxId
template <int GfxId>
struct GfxConfig
{
    static constexpr index_t get_vec_len_fp16()
    {
        if constexpr (GfxId == 1200)
        {
            return Gfx120x::VecLenFP16;
        }
        else if constexpr (GfxId == 900)
        {
            return Gfx90x::VecLenFP16;
        }
    }

    static constexpr index_t get_vec_len_fp32()
    {
        if constexpr (GfxId == 1200)
        {
            return Gfx120x::VecLenFP32;
        }
        else if constexpr (GfxId == 900)
        {
            return Gfx90x::VecLenFP32;
        }
    }    

    static constexpr index_t get_k_cm_lane()
    {
        if constexpr (GfxId == 1200)
        {
            return Gfx120x::kCMLane;
        }
        else if constexpr (GfxId == 900)
        {
            return Gfx90x::kCMLane;
        }        
    }
    
    static constexpr index_t get_k_cm0_per_lane()
    {
        if constexpr (GfxId == 1200)
        {
            return Gfx120x::kCM0PerLane;
        }
        else if constexpr (GfxId == 900)
        {
            return Gfx90x::kCM0PerLane;
        }        
    }    
    static constexpr index_t get_k_cm1_per_lane()
    {
        if constexpr (GfxId == 1200)
        {
            return Gfx120x::kCM1PerLane;
        }
        else if constexpr (GfxId == 900)
        {
            return Gfx90x::kCM1PerLane;
        }        
    }        
};

} // namespace ck_tile
