// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
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

/*! @enum amdgcn_target_arch_id
 * @brief Defines constants for AMDGCN architecture target IDs
 */
enum struct amdgcn_target_arch_id
{
    GFX908  = 0x0908,
    GFX90A  = 0x090A,
    GFX942  = 0x0942,
    GFX950  = 0x0950,
    GFX1100 = 0x1100,
    GFX1101 = 0x1101,
    GFX1102 = 0x1102,
    GFX1151 = 0x1151,
    GFX1200 = 0x1200,
    GFX1201 = 0x1201,
    HOST    = 0x0000,
};

/*! @enum amdgcn_wave_size
 * @brief Defines constants for AMDGCN architecture wave sizes
 */
enum struct amdgcn_wave_size
{
    WAVE32 = 32u,
    WAVE64 = 64u,
    HOST   = 1u,
};

/**
 * @brief Converts a lower-case string to the corresponding amdgcn_target_arch_id value.
 *        Returns amdgcn_target_arch_id::HOST if no match is found.
 *        Matches if the input contains the architecture substring.
 *        Example: "gfx908", "gfx90a", "gfx1100", etc. can be parsed from hip runtime info.
 */
constexpr inline auto gfx_target_string_to_arch_id(char const* testStr)
{
    auto str = std::string(testStr);
    if(str.find("gfx908") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX908;
    }
    else if(str.find("gfx90a") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX90A;
    }
    else if(str.find("gfx942") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX942;
    }
    else if(str.find("gfx950") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX950;
    }
    else if(str.find("gfx1100") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX1100;
    }
    else if(str.find("gfx1101") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX1101;
    }
    else if(str.find("gfx1102") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX1102;
    }
    else if(str.find("gfx1151") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX1151;
    }
    else if(str.find("gfx1200") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX1200;
    }
    else if(str.find("gfx1201") != std::string::npos)
    {
        return amdgcn_target_arch_id::GFX1201;
    }
    else
    {
        return amdgcn_target_arch_id::HOST;
    }
}

/*! @brief Returns true if the given arch_id is a gfx9 architecture */
CK_TILE_HOST_DEVICE constexpr bool is_gfx9_arch_id(amdgcn_target_arch_id arch_id)
{
    return is_any_value_of(arch_id,
                           amdgcn_target_arch_id::GFX908,
                           amdgcn_target_arch_id::GFX90A,
                           amdgcn_target_arch_id::GFX942,
                           amdgcn_target_arch_id::GFX950);
}
/*! @brief Returns true if the given arch_id is a gfx11 architecture */
CK_TILE_HOST_DEVICE constexpr bool is_gfx11_arch_id(amdgcn_target_arch_id arch_id)
{
    return is_any_value_of(arch_id,
                           amdgcn_target_arch_id::GFX1100,
                           amdgcn_target_arch_id::GFX1101,
                           amdgcn_target_arch_id::GFX1102,
                           amdgcn_target_arch_id::GFX1151);
}

/*! @brief Returns true if the given arch_id is a gfx12 architecture */
CK_TILE_HOST_DEVICE constexpr bool is_gfx12_arch_id(amdgcn_target_arch_id arch_id)
{
    return is_any_value_of(arch_id, amdgcn_target_arch_id::GFX1200, amdgcn_target_arch_id::GFX1201);
}

/*! @brief Returns true if the given arch_id is a CDNA architecture */
CK_TILE_HOST_DEVICE constexpr bool is_cdna_arch_id(amdgcn_target_arch_id arch_id)
{
    return is_gfx9_arch_id(arch_id);
}

/*! @brief Returns true if the given arch_id is a RDNA architecture */
CK_TILE_HOST_DEVICE constexpr bool is_rdna_arch_id(amdgcn_target_arch_id arch_id)
{
    return is_gfx11_arch_id(arch_id) || is_gfx12_arch_id(arch_id);
}

/*! @brief Returns true if the given arch_id maps to wave32 (RDNA) */
CK_TILE_HOST_DEVICE constexpr bool is_wave32_arch_id(amdgcn_target_arch_id arch_id)
{
    return is_rdna_arch_id(arch_id);
}

/*! @brief Returns true if the given arch_id maps to wave64 (CDNA) */
CK_TILE_HOST_DEVICE constexpr bool is_wave64_arch_id(amdgcn_target_arch_id arch_id)
{
    return is_cdna_arch_id(arch_id);
}

/*! @brief SFINAE enabler for target architecture if it is in the list of supported architectures
 * @tparam TargetId The target architecture ID to check
 * @tparam SupportedArchs The list of supported architecture IDs
 */
template <amdgcn_target_arch_id TargetId, amdgcn_target_arch_id... SupportedArchs>
using enable_if_target_arch_id_t = std::enable_if_t<is_any_value_of(TargetId, SupportedArchs...)>;

/*! @brief SFINAE enabler for target architecture if it is CDNA arch
 * @tparam TargetId The target architecture ID to check
 */
template <amdgcn_target_arch_id TargetId>
using enable_if_cdna_target_id_t = std::enable_if_t<is_cdna_arch_id(TargetId)>;

/*! @brief SFINAE enabler for target architecture if it is CDNA arch
 * @tparam TargetId The target architecture ID to check
 */
template <amdgcn_target_arch_id TargetId>
using enable_if_rdna_target_id_t = std::enable_if_t<is_rdna_arch_id(TargetId)>;

/*! @brief SFINAE enabler for target architecture if it is gfx9
 * @tparam TargetId The target architecture ID to check
 */
template <amdgcn_target_arch_id TargetId>
using enable_if_gfx9_target_id_t = std::enable_if_t<is_gfx9_arch_id(TargetId)>;

/*! @brief SFINAE enabler for target architecture if it is gfx11
 * @tparam TargetId The target architecture ID to check
 */
template <amdgcn_target_arch_id TargetId>
using enable_if_gfx11_target_id_t = std::enable_if_t<is_gfx11_arch_id(TargetId)>;

/*! @brief SFINAE enabler for target architecture if it is gfx12
 * @tparam TargetId The target architecture ID to check
 */
template <amdgcn_target_arch_id TargetId>
using enable_if_gfx12_target_id_t = std::enable_if_t<is_gfx12_arch_id(TargetId)>;

/*! @brief SFINAE enabler for target architecture if it is wave32
 * @tparam TargetId The target architecture ID to check
 */
template <amdgcn_target_arch_id TargetId>
using enable_if_wave32_target_id_t = std::enable_if_t<is_wave32_arch_id(TargetId)>;

/*! @brief SFINAE enabler for target architecture if it is wave64
 * @tparam TargetId The target architecture ID to check
 */
template <amdgcn_target_arch_id TargetId>
using enable_if_wave64_target_id_t = std::enable_if_t<is_wave64_arch_id(TargetId)>;

/*! @brief Returns the amdgcn_target_arch_id of the current compiler pass
 */
CK_TILE_HOST_DEVICE constexpr auto get_target_arch_id()
{
    if constexpr(CK_TILE_ARCH_GFX908)
    {
        return amdgcn_target_arch_id::GFX908;
    }
    else if constexpr(CK_TILE_ARCH_GFX90A)
    {
        return amdgcn_target_arch_id::GFX90A;
    }
    else if constexpr(CK_TILE_ARCH_GFX942)
    {
        return amdgcn_target_arch_id::GFX942;
    }
    else if constexpr(CK_TILE_ARCH_GFX950)
    {
        return amdgcn_target_arch_id::GFX950;
    }
    else if constexpr(CK_TILE_ARCH_GFX1100)
    {
        return amdgcn_target_arch_id::GFX1100;
    }
    else if constexpr(CK_TILE_ARCH_GFX1101)
    {
        return amdgcn_target_arch_id::GFX1101;
    }
    else if constexpr(CK_TILE_ARCH_GFX1102)
    {
        return amdgcn_target_arch_id::GFX1102;
    }
    else if constexpr(CK_TILE_ARCH_GFX1151)
    {
        return amdgcn_target_arch_id::GFX1151;
    }
    else if constexpr(CK_TILE_ARCH_GFX1200)
    {
        return amdgcn_target_arch_id::GFX1200;
    }
    else if constexpr(CK_TILE_ARCH_GFX1201)
    {
        return amdgcn_target_arch_id::GFX1201;
    }
    else // Host default
    {
        return amdgcn_target_arch_id::HOST;
    }
}

/*! @brief Returns the amdgcn_wave_size of the current compiler pass
 */
CK_TILE_HOST_DEVICE constexpr auto get_warp_size()
{
    if constexpr(CK_TILE_WAVE64_MODE)
    {
        return static_cast<uint32_t>(amdgcn_wave_size::WAVE64);
    }
    else if constexpr(CK_TILE_WAVE32_MODE)
    {
        return static_cast<uint32_t>(amdgcn_wave_size::WAVE32);
    }
    else // Host default
    {
        return static_cast<uint32_t>(amdgcn_wave_size::HOST);
    }
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

// https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
struct waitcnt_arg
{
#if defined(__gfx12__)
    // use s_wait_loadcnt_dscnt in this instruction; in this instruction, ds [5:0]; mem [13:8]
    CK_TILE_DEVICE static constexpr index_t MAX = 0b00'111111'00'111111;

    CK_TILE_DEVICE static constexpr index_t kMaxVmCnt   = 0b111111;
    CK_TILE_DEVICE static constexpr index_t kMaxExpCnt  = 0b111;
    CK_TILE_DEVICE static constexpr index_t kMaxLgkmCnt = 0b111111;

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_vmcnt()
    {
        static_assert(cnt >= 0 && !(cnt >> 6), "valid range is [0..63]");
        return MAX & (cnt << 8);
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_expcnt()
    {
        return 0; // no export in MI series
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_lgkmcnt()
    {
        static_assert(cnt >= 0 && !(cnt >> 6), "valid range is [0..63]");
        return MAX & cnt;
    }
#else
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
#endif
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
struct gfx11_t
{
};
struct gfx12_t
{
};

CK_TILE_DEVICE static constexpr auto get_device_arch()
{
#if defined(__gfx11__)
    return gfx11_t{};
#else // if defined(__gfx12__)
    return gfx12_t{};
#endif
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
