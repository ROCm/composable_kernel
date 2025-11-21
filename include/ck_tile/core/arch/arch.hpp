// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing_builtins.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/utility/ignore.hpp"

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

CK_TILE_HOST bool is_wave32()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return false;
    }
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return false;
    }
    return props.major > 9;
}

CK_TILE_DEVICE index_t get_grid_size() { return gridDim.x; }

CK_TILE_DEVICE index_t get_block_size() { return blockDim.x; }

// TODO: deprecate these
CK_TILE_DEVICE index_t get_thread_local_1d_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

CK_TILE_DEVICE index_t get_block_1d_id() { return blockIdx.x; }

// Use these instead
CK_TILE_DEVICE index_t get_lane_id() { return __lane_id(); }

template <bool ReturnSgpr = true>
CK_TILE_DEVICE index_t get_warp_id(bool_constant<ReturnSgpr> = {})
{
    const index_t warp_id = threadIdx.x / get_warp_size();
    if constexpr(ReturnSgpr)
    {
        return amd_wave_read_first_lane(warp_id);
    }
    else
    {
        return warp_id;
    }
}

CK_TILE_DEVICE index_t get_thread_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_block_id() { return blockIdx.x; }

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

struct WaitcntLayoutGfx12
{ // s_wait_loadcnt_dscnt: mem[13:8], ds[5:0]
    CK_TILE_DEVICE static constexpr index_t VM_MASK   = 0x3F; // mem
    CK_TILE_DEVICE static constexpr index_t LGKM_MASK = 0x3F; // ds
    CK_TILE_DEVICE static constexpr bool HAS_EXP      = false;

    CK_TILE_DEVICE static constexpr index_t pack_vm(index_t c) { return ((c & VM_MASK) << 8); }
    CK_TILE_DEVICE static constexpr index_t pack_lgkm(index_t c) { return ((c & LGKM_MASK) << 0); }
    CK_TILE_DEVICE static constexpr index_t pack_exp(index_t) { return 0; }
};

struct WaitcntLayoutGfx11
{ // vm[15:10] (6), lgkm[9:4] (6), exp unused
    CK_TILE_DEVICE static constexpr index_t VM_MASK   = 0x3F;
    CK_TILE_DEVICE static constexpr index_t LGKM_MASK = 0x3F;
    CK_TILE_DEVICE static constexpr bool HAS_EXP      = false;

    CK_TILE_DEVICE static constexpr index_t pack_vm(index_t c) { return ((c & VM_MASK) << 10); }
    CK_TILE_DEVICE static constexpr index_t pack_lgkm(index_t c) { return ((c & LGKM_MASK) << 4); }
    CK_TILE_DEVICE static constexpr index_t pack_exp(index_t) { return 0; }
};

struct WaitcntLayoutLegacy
{ // FE'DC'BA98'7'654'3210 => VV'UU'LLLL'U'EEE'VVVV
    CK_TILE_DEVICE static constexpr index_t VM_MASK   = 0x3F; // split: low4 + hi2
    CK_TILE_DEVICE static constexpr index_t LGKM_MASK = 0x0F; // [11:8]
    CK_TILE_DEVICE static constexpr index_t EXP_MASK  = 0x07; // [6:4]
    CK_TILE_DEVICE static constexpr bool HAS_EXP      = true;

    CK_TILE_DEVICE static constexpr index_t pack_vm(index_t c)
    {
        c &= VM_MASK;
        return ((c & 0xF) << 0) | ((c & 0x30) << 10);
    }
    CK_TILE_DEVICE static constexpr index_t pack_lgkm(index_t c) { return ((c & LGKM_MASK) << 8); }
    CK_TILE_DEVICE static constexpr index_t pack_exp(index_t c) { return ((c & EXP_MASK) << 4); }
};

// Select active layout
#if defined(__gfx12__)
using Waitcnt = WaitcntLayoutGfx12;
#elif defined(__gfx11__)
using Waitcnt = WaitcntLayoutGfx11;
#else
using Waitcnt = WaitcntLayoutLegacy;
#endif

//----------------------------------------------
// Public API: only from_* (constexpr templates)
//----------------------------------------------
struct waitcnt_arg
{
    // kMax* exposed for callers; match field widths per-arch
#if defined(__gfx12__) || defined(__gfx11__)
    CK_TILE_DEVICE static constexpr index_t kMaxVmCnt   = 0x3F; // 6 bits
    CK_TILE_DEVICE static constexpr index_t kMaxLgkmCnt = 0x3F; // 6 bits
    CK_TILE_DEVICE static constexpr index_t kMaxExpCnt  = 0x0;  // none
#else
    CK_TILE_DEVICE static constexpr index_t kMaxVmCnt   = 0x3F; // 6 bits (split)
    CK_TILE_DEVICE static constexpr index_t kMaxLgkmCnt = 0x0F; // 4 bits
    CK_TILE_DEVICE static constexpr index_t kMaxExpCnt  = 0x07; // 3 bits
#endif

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_vmcnt()
    {
        static_assert((cnt & ~Waitcnt::VM_MASK) == 0, "vmcnt out of range");
        return Waitcnt::pack_vm(cnt);
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_lgkmcnt()
    {
        static_assert((cnt & ~Waitcnt::LGKM_MASK) == 0, "lgkmcnt out of range");
        return Waitcnt::pack_lgkm(cnt);
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_expcnt()
    {
        if constexpr(Waitcnt::HAS_EXP)
        {
            // EXP_MASK only exists on legacy
#if !defined(__gfx12__) && !defined(__gfx11__)
            static_assert((cnt & ~Waitcnt::EXP_MASK) == 0, "expcnt out of range");
            return Waitcnt::pack_exp(cnt);
#else
            (void)cnt;
            return 0;
#endif
        }
        else
        {
            static_assert(cnt == 0, "expcnt unsupported on this arch");
            return 0;
        }
    }
};

template <index_t vmcnt   = waitcnt_arg::kMaxVmCnt,
          index_t expcnt  = waitcnt_arg::kMaxExpCnt,
          index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void s_waitcnt()
{
#if defined(__gfx12__)
    // GFX12 do't use __builtin_amdgcn_s_waitcnt
    constexpr index_t wait_mask = waitcnt_arg::from_vmcnt<vmcnt>() |
                                  waitcnt_arg::from_expcnt<expcnt>() |
                                  waitcnt_arg::from_lgkmcnt<lgkmcnt>();

    asm volatile("s_wait_loadcnt_dscnt %0" : : "n"(wait_mask) : "memory");
#else
    __builtin_amdgcn_s_waitcnt(waitcnt_arg::from_vmcnt<vmcnt>() |
                               waitcnt_arg::from_expcnt<expcnt>() |
                               waitcnt_arg::from_lgkmcnt<lgkmcnt>());
#endif
}

template <index_t vmcnt   = waitcnt_arg::kMaxVmCnt,
          index_t expcnt  = waitcnt_arg::kMaxExpCnt,
          index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void s_waitcnt_barrier()
{
#if defined(__gfx12__)
    // GFX12 optimization: Manual barrier implementation avoids performance penalty
    // from __builtin_amdgcn_s_barrier which inserts extra s_wait_loadcnt_dscnt 0x0
    constexpr index_t wait_mask = waitcnt_arg::from_vmcnt<vmcnt>() |
                                  waitcnt_arg::from_expcnt<expcnt>() |
                                  waitcnt_arg::from_lgkmcnt<lgkmcnt>();

    asm volatile("s_wait_loadcnt_dscnt %0\n"
                 "s_barrier_signal -1\n"
                 "s_barrier_wait -1"
                 :
                 : "n"(wait_mask)
                 : "memory");
#else
    s_waitcnt<vmcnt, expcnt, lgkmcnt>();
    __builtin_amdgcn_s_barrier();
#endif
}

template <index_t lgkmcnt = 0>
CK_TILE_DEVICE void block_sync_lds()
{
    s_waitcnt_barrier<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, lgkmcnt>();
}

template <index_t vmcnt = 0>
CK_TILE_DEVICE void block_sync_lds_direct_load()
{
    s_waitcnt_barrier<vmcnt, waitcnt_arg::kMaxExpCnt, waitcnt_arg::kMaxLgkmCnt>();
}

CK_TILE_DEVICE void s_nop(index_t cnt = 0)
{
#if 1
    asm volatile("s_nop %0" : : "n"(cnt) :);
#else
    __builtin_amdgcn_sched_barrier(cnt);
#endif
}

#define CK_TILE_CONSTANT_ADDRESS_SPACE \
    __attribute__((address_space(      \
        static_cast<safe_underlying_type_t<address_space_enum>>(address_space_enum::constant))))

template <typename T>
__device__ T* cast_pointer_to_generic_address_space(T CK_TILE_CONSTANT_ADDRESS_SPACE* p)
{
    // cast a pointer in "Constant" address space (4) to "Generic" address space (0)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T*)(p); // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

template <typename T>
__host__ __device__ T CK_TILE_CONSTANT_ADDRESS_SPACE* cast_pointer_to_constant_address_space(T* p)
{
    // cast a pointer in "Generic" address space (0) to "Constant" address space (4)
    // only c-style pointer cast seems be able to be compiled;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T CK_TILE_CONSTANT_ADDRESS_SPACE*)p; // NOLINT(old-style-cast)
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

/// Helper function to convert address space enum to string
CK_TILE_HOST_DEVICE constexpr const char* address_space_to_string(address_space_enum addr_space)
{
    switch(addr_space)
    {
    case address_space_enum::generic: return "generic";
    case address_space_enum::global: return "global";
    case address_space_enum::lds: return "lds";
    case address_space_enum::sgpr: return "sgpr";
    case address_space_enum::constant: return "constant";
    case address_space_enum::vgpr: return "vgpr";
    default: return "unknown";
    }
}

// Architecture tags
struct gfx9_t
{
};
struct gfx950_t
{
};
struct gfx103_t
{
};
struct gfx11_t
{
};
struct gfx12_t
{
};
struct gfx_invalid_t
{
};

CK_TILE_DEVICE static constexpr auto get_device_arch()
{
// FIXME(0): on all devices except gfx11 it returns gfx12_t
// FIXME(1): during the host compilation pass it returns gfx12_t
#if defined(__gfx11__)
    return gfx11_t{};
#else
    return gfx12_t{};
#endif
}

CK_TILE_DEVICE static constexpr auto get_n_words_per_128b() { return 4; }

namespace detail {
CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx9_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx103_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx11_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx12_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx950_t) { return 64; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx_invalid_t) { return 0; }

CK_TILE_DEVICE static constexpr auto arch_tag_dispatch()
{
#if defined(__gfx103__)
    return gfx103_t{};
#elif defined(__gfx11__)
    return gfx11_t{};
#elif defined(__gfx12__)
    return gfx12_t{};
#elif defined(__gfx950__)
    return gfx950_t{};
#elif defined(__gfx9__)
    return gfx9_t{};
#else
    return gfx_invalid_t{};
#endif
}
} // namespace detail
CK_TILE_DEVICE static constexpr auto get_n_lds_banks()
{
    return detail::get_n_lds_banks(detail::arch_tag_dispatch());
}

enum LLVMSchedGroupMask : int32_t
{
    NONE       = 0,
    ALU        = 1 << 0,
    VALU       = 1 << 1,
    SALU       = 1 << 2,
    MFMA       = 1 << 3,
    VMEM       = 1 << 4,
    VMEM_READ  = 1 << 5,
    VMEM_WRITE = 1 << 6,
    DS         = 1 << 7,
    DS_READ    = 1 << 8,
    DS_WRITE   = 1 << 9,
    ALL        = (DS_WRITE << 1) - 1,
};
} // namespace ck_tile
