// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "data_type.hpp"

namespace ck {

#if defined(__gfx12__)
template <typename T>
__device__ auto amd_global_load_transpose_to_vgpr(const T* in_ptr)
{
    using vector_t = typename vector_type<T, 8>::type;
    if constexpr(sizeof(T) == 2)
    {
        using llvm_half8_t = NativeVectorT<llvm_half_t, 8>;
        __attribute__((address_space(1))) llvm_half8_t* glb_ptr =
            reinterpret_cast<__attribute__((address_space(1))) llvm_half8_t*>(
                reinterpret_cast<uintptr_t>(in_ptr));
        return bit_cast<vector_t>(__builtin_amdgcn_global_load_tr_b128_v8f16(glb_ptr));
    }
    else if constexpr(sizeof(T) == 1)
    {
        typedef int llvm_intx2_t __attribute__((ext_vector_type(2)));
        __attribute__((address_space(1))) llvm_intx2_t* glb_ptr =
            reinterpret_cast<__attribute__((address_space(1))) llvm_intx2_t*>(
                reinterpret_cast<uintptr_t>(in_ptr));
        return bit_cast<vector_t>(__builtin_amdgcn_global_load_tr_b64_v2i32(glb_ptr));
    }
    else
    {
        static_assert(false, "not implemented");
    }
}
#endif

#if defined(__gfx125__) || defined(__gfx950__)
template <typename T>
__device__ auto amd_lds_load_transpose_to_vgpr(const T* __restrict__ in_ptr)
{
#define CK_LDS_ADDR __attribute__((address_space(3)))
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
    // Use C-style cast to change address space without dropping llvm noalias attribute
    const auto in_ptr_ = (CK_LDS_ADDR T*)(const_cast<T*>(in_ptr));
#ifdef __clang__
#pragma clang diagnostic pop
#endif

    if constexpr(is_same<T, half_t>::value)
    {
#if defined(__gfx125__)
        using vector_t      = typename vector_type<T, 8>::type;
        using llvm_fp16x8_t = NativeVectorT<llvm_half_t, 8>;
        auto lds_ptr        = reinterpret_cast<CK_LDS_ADDR llvm_fp16x8_t*>(in_ptr_);
        return bit_cast<vector_t>(__builtin_amdgcn_ds_load_tr16_b128_v8f16(lds_ptr));
#elif defined(__gfx950__)
        using vector_t      = typename vector_type<T, 4>::type;
        using llvm_fp16x4_t = NativeVectorT<llvm_half_t, 4>;
        auto lds_ptr        = reinterpret_cast<CK_LDS_ADDR llvm_fp16x4_t*>(in_ptr_);
        return bit_cast<vector_t>(__builtin_amdgcn_ds_read_tr16_b64_v4f16(lds_ptr));
#endif
    }
    else if constexpr(is_same<T, bhalf_t>::value)
    {
#if defined(__gfx125__)
        using vector_t = typename vector_type<T, 8>::type;
        auto lds_ptr   = reinterpret_cast<CK_LDS_ADDR llvm_bf16x8_t*>(in_ptr_);
        return bit_cast<vector_t>(__builtin_amdgcn_ds_load_tr16_b128_v8bf16(lds_ptr));
#elif defined(__gfx950__)
        using vector_t = typename vector_type<T, 4>::type;
        auto lds_ptr   = reinterpret_cast<CK_LDS_ADDR llvm_bf16x4_t*>(in_ptr_);
        return bit_cast<vector_t>(__builtin_amdgcn_ds_read_tr16_b64_v4bf16(lds_ptr));
#endif
    }
    else
    {
        static_assert(false, "not implemented");
    }
#undef CK_LDS_ADDR
}
#endif

} // namespace ck
