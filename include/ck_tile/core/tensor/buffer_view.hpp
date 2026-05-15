// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing_builtins.hpp"
#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"
#include "ck_tile/core/arch/generic_memory_space_atomic.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/ignore.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#pragma clang diagnostic ignored "-Wlifetime-safety-lifetimebound-violation"

namespace ck_tile {

// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
// FIXME: amd_buffer_coherence_enum is only meaningful for buffer addressing. Need to split
// buffer_view definition for different memory address space (Global/GenericLds/Vgpr)
template <address_space_enum BufferAddressSpace,
          typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default>
struct buffer_view;

struct null_buffer_view
{
};
// Address Space: generic
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::generic,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() {}

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::generic;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              index_t static_offset      = 0,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t linear_offset,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i + linear_offset]), sizeof(X));

            return tmp;
#else
            return *c_style_pointer_cast<const X*>(&p_data_[i + linear_offset]);
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    /*
    In the generic address space, we do not support the transpose instruction in the buffer view.
    Will report compilation error when developer wants to use it.
    */
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto transpose_get(index_t i,
                                                index_t linear_offset,
                                                bool is_valid_element,
                                                bool_constant<oob_conditional_check> = {}) const
    {
        static_assert(false, "Error: transpose load not supported in global memory space.");
        ignore = i;
        ignore = linear_offset;
        ignore = is_valid_element;
        return;
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, linear_offset, is_valid_element);
            this->template set<X>(i, linear_offset, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i + linear_offset]), &tmp, sizeof(X));
#else
            *c_style_pointer_cast<X*>(&p_data_[i + linear_offset]) = x;
#endif
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }
};

// Address Space: Global
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence>
struct buffer_view<address_space_enum::global,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   Coherence>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    int32x4_t cached_buf_res_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    static constexpr index_t PackedSize = ck_tile::numeric_traits<remove_cvref_t<T>>::PackedSize;

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, cached_buf_res_{0}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view([[clang::lifetimebound]] T* __restrict__ p_data,
                                              BufferSizeType buffer_size)
        : p_data_{p_data},
          buffer_size_{buffer_size / PackedSize},
          cached_buf_res_{0},
          invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data},
          buffer_size_{buffer_size / PackedSize},
          cached_buf_res_{0},
          invalid_element_value_{invalid_element_value}
    {
    }

    // this is non constexpr intentially (will call some intrinsic internally)
    // Must call for buffers that need *_raw load/store
    CK_TILE_HOST_DEVICE void init_raw()
    {
        cached_buf_res_ = make_wave_buffer_resource(p_data_, (buffer_size_) * sizeof(type));
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::global;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              index_t static_offset      = 0,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t linear_offset,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_USE_AMD_BUFFER_LOAD
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif
        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return amd_buffer_load_invalid_element_return_zero<remove_cvref_t<T>,
                                                                   t_per_x,
                                                                   Coherence,
                                                                   oob_conditional_check>(
                    p_data_, i + linear_offset, is_valid_element, buffer_size_);
            }
            else
            {
                return amd_buffer_load_invalid_element_return_customized_value<
                    remove_cvref_t<T>,
                    t_per_x,
                    Coherence,
                    oob_conditional_check>(p_data_,
                                           i + linear_offset,
                                           is_valid_element,
                                           buffer_size_,
                                           invalid_element_value_);
            }
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp;

                __builtin_memcpy(&tmp, &(p_data_[i + linear_offset]), sizeof(X));

                return tmp;
#else
                return *c_style_pointer_cast<const X*>(&p_data_[i + linear_offset]);
#endif
            }
            else
            {
                if constexpr(InvalidElementUseNumericalZeroValue)
                {
                    return X{numeric<remove_cvref_t<T>>::zero()};
                }
                else
                {
                    return X{invalid_element_value_};
                }
            }
        }
    }

    /*
    In the global memory address space, we do not support the transpose instruction in the buffer
    view. Will report compilation error when developer wants to use it.
    */
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto transpose_get(index_t i,
                                                index_t linear_offset,
                                                bool is_valid_element,
                                                bool_constant<oob_conditional_check> = {}) const
    {
        static_assert(false, "Error: transpose load not supported in global memory space.");
        ignore = i;
        ignore = linear_offset;
        ignore = is_valid_element;
        return;
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get_raw(remove_cvref_t<X>& dst,
                                          index_t v_offset,
                                          index_t i_offset,
                                          bool is_valid_element,
                                          bool_constant<pre_nop> = {}) const
    {
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_buffer_load_raw<remove_cvref_t<T>, t_per_x, Coherence, oob_conditional_check, pre_nop>(
            dst, cached_buf_res_, v_offset, i_offset, is_valid_element, bool_constant<pre_nop>{});
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false,
              typename linear_offset_t>
    CK_TILE_DEVICE constexpr auto async_get(CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                                            index_t i,
                                            index_t wave_i,
                                            linear_offset_t&& linear_offset,
                                            bool is_valid_element,
                                            bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T

        // If T is already a vector, how many elements are in T?
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        // If X is a vector, how many elements are in X?
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;
        // X should be a multiple of T for X to exactly contain every T.
        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        // how many chunks of T are in one X?
        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

#if defined(__gfx125__) // for gfx125; there uses another instruction to do async load
        auto p_uniform_ptr              = amd_wave_read_first_lane(p_data_);
        constexpr index_t static_offset = linear_offset_t{}.value;
        amd_async_global_load_to_lds<remove_cvref_t<T>, t_per_x, static_offset, true, Coherence>(
            smem, p_uniform_ptr, i + wave_i, is_valid_element);
        ignore = linear_offset;
#else
        const auto rsrc = make_builtin_buffer_resource(p_data_, buffer_size_ * sizeof(type));

        amd_async_buffer_load_with_oob<remove_cvref_t<T>, t_per_x, Coherence>(
            smem,
            rsrc,
            i,
            wave_i,
            std::forward<linear_offset_t>(linear_offset),
            is_valid_element,
            bool_constant<oob_conditional_check>{});
#endif
    }

    // i is offset of T, not X. i should be aligned to X.
    // mask — M0[15:0] WGP participation mask; M0[16] sets early-timeout.
    template <typename X,
              index_t inst_offset = 0,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    cluster_async_get(remove_cvref_t<T>* smem, index_t i, index_t linear_offset, int mask) const
    {
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#ifdef __gfx1250__
        auto p_uniform_ptr = amd_wave_read_first_lane(p_data_);

        const remove_cvref_t<X>* g_src =
            reinterpret_cast<const remove_cvref_t<X>*>(p_uniform_ptr + i + linear_offset);

        // reinterpret_cast changes only the element type (generic→generic, no address-space
        // change). to_lds then converts generic→address_space(3) using a pragma-guarded
        // C-style cast, matching the pattern used by the rest of the codebase.
        auto* lds_ptr = to_lds(reinterpret_cast<remove_cvref_t<X>*>(smem));

        cluster_multicast_load_async_to_lds<remove_cvref_t<X>, inst_offset>(g_src, lds_ptr, mask);
#else
        (void)smem;
        (void)i;
        (void)linear_offset;
        (void)mask;
        static_assert(sizeof(X) == 0, "cluster_async_get is only supported on gfx1250");
#endif
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool pre_nop = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto async_get_raw(remove_cvref_t<T>* smem,
                                                index_t i,
                                                index_t linear_offset,
                                                bool /*is_valid_element*/,
                                                bool_constant<pre_nop> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_async_buffer_load_with_oob_raw<remove_cvref_t<T>, t_per_x, Coherence>(
            smem, cached_buf_res_, i, linear_offset, bool_constant<pre_nop>{});
    }

    template <amd_buffer_coherence_enum Coherence_ = Coherence>
    struct GlobalPrefetchDataOp
    {
        // addr needs to point to global memory!
        CK_TILE_DEVICE void operator()([[maybe_unused]] const void* addr) const
        {
#if defined(__gfx125__)
            // Compiler fence to not move ds_loads freely before/after this prefetch builtin
            asm volatile("" ::: "memory");
            __builtin_amdgcn_global_prefetch(addr, static_cast<index_t>(Coherence_));
            asm volatile("" ::: "memory");
#endif
        }

#if defined(__gfx12__)
        static constexpr bool is_cu_scope = []() {
            constexpr int coherence    = static_cast<int>(Coherence_);
            constexpr int se_scope     = static_cast<int>(amd_buffer_coherence_enum::SE);
            constexpr int device_scope = static_cast<int>(amd_buffer_coherence_enum::DEVICE);
            constexpr int system_scope = static_cast<int>(amd_buffer_coherence_enum::SYSTEM);
            // CU scope: check if scope bits are zero
            return !(coherence & se_scope || coherence & device_scope || coherence & system_scope);
        }();
#endif

        CK_TILE_DEVICE constexpr bool need_oob_check() const
        {
#if defined(__gfx125__)
            // we need oob check for non-speculative prefetch to not get Page Fault
            constexpr int coherence = static_cast<int>(Coherence_);
            constexpr int rt_non_spec =
                static_cast<int>(amd_buffer_coherence_enum::RT_NON_SPECULATIVE);
            constexpr int ht_non_spec =
                static_cast<int>(amd_buffer_coherence_enum::HT_NON_SPECULATIVE);

            if constexpr(is_cu_scope) // for all CU scope we have non-speculative prefetch
            {
                return true;
            }
            else if constexpr(((coherence & rt_non_spec) == rt_non_spec) ||
                              ((coherence & ht_non_spec) ==
                               ht_non_spec)) // for all other scopes we have speculative prefetch
                                             // unless set otherwise by Temporal Hint
            {
                return true;
            }
#endif
            return false;
        }
    };

    // i is offset of T, not X. i should be aligned to X
    // static_offset is compile-time offset for LDS access optimization
    template <typename X,
              amd_buffer_coherence_enum Coherence_ = Coherence,
              index_t static_offset                = 0,
              bool oob_conditional_check           = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void prefetch(index_t i,
                                           index_t linear_offset,
                                           bool is_valid_element,
                                           bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if constexpr(!GlobalPrefetchDataOp<Coherence_>{}.need_oob_check())
        {
            is_valid_element = true;
        }

        if(is_valid_element)
        {
            // call prefetch here
            GlobalPrefetchDataOp<Coherence_>{}(
                c_style_pointer_cast<const void*>(&(p_data_[i + linear_offset + static_offset])));
        }

        // Note: prefetch is a hint instruction that doesn't return a value
        // No action needed for invalid elements
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i,
                               index_t linear_offset,
                               bool is_valid_element,
                               const X& x,
                               bool_constant<oob_conditional_check> = {})
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X, oob_conditional_check>(i, linear_offset, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_add)
        {
            this->template atomic_add<X, oob_conditional_check>(
                i, linear_offset, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_max)
        {
            this->template atomic_max<X, oob_conditional_check>(
                i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp =
                this->template get<X, oob_conditional_check>(i, linear_offset, is_valid_element);
            this->template set<X, oob_conditional_check>(
                i, linear_offset, is_valid_element, x + tmp);
            // tmp += x;
            // this->template set<X>(i, is_valid_element, tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update_raw(index_t i,
                                   index_t linear_offset,
                                   bool is_valid_element,
                                   const X& x,
                                   bool_constant<oob_conditional_check> = {},
                                   bool_constant<pre_nop>               = {})
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set_raw<X, oob_conditional_check>(i, linear_offset, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_add)
        {
            this->template atomic_add_raw<X, oob_conditional_check, pre_nop>(
                i, linear_offset, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_max)
        {
            // this->template atomic_max_raw<X>(i, linear_offset, is_valid_element, x);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_USE_AMD_BUFFER_STORE
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_store<remove_cvref_t<T>, t_per_x, Coherence>(
                x, p_data_, i + linear_offset, is_valid_element, buffer_size_);
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp = x;

                __builtin_memcpy(&(p_data_[i + linear_offset]), &tmp, sizeof(X));
#else
                *c_style_pointer_cast<X*>(&p_data_[i + linear_offset]) = x;
#endif
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set_raw(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;
        amd_buffer_store_raw<remove_cvref_t<T>, t_per_x, Coherence, oob_conditional_check>(
            x, p_data_, i, linear_offset, is_valid_element, buffer_size_);
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void
    atomic_add(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        using scalar_t = typename vector_traits<remove_cvref_t<T>>::scalar_type;

        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(get_address_space() == address_space_enum::global, "only support global mem");

#if CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, int32_t> ||
            std::is_same_v<remove_cvref_t<scalar_t>, float> ||
            (std::is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0)
#if defined(__gfx942__) || defined(__gfx950__) // only gfx942 and gfx950 support atomic_pk_add_bf16
            ||
            (std::is_same_v<remove_cvref_t<scalar_t>, bfloat16_t> && scalar_per_x_vector % 2 == 0)
#endif
            ;
#elif CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && (!CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT)
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, int32_t>;
#elif(!CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER) && CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, float> ||
            (std::is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0)
#if defined(__gfx942__) || defined(__gfx950__) // only gfx942 and gfx950 support atomic_pk_add_bf16
            ||
            (std::is_same_v<remove_cvref_t<scalar_t>, bfloat16_t> && scalar_per_x_vector % 2 == 0)
#endif
            ;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        if constexpr(use_amd_buffer_addressing)
        {
            amd_buffer_atomic_add<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i + linear_offset, is_valid_element, buffer_size_);
        }
        else
        {
            if(is_valid_element)
            {
                atomic_add_g<remove_cvref_t<T>, t_per_x>(&p_data_[i + linear_offset], x);
            }
        }
    }

    template <typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void
    atomic_add_raw(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // using scalar_t = typename vector_traits<remove_cvref_t<T>>::scalar_type;

        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(get_address_space() == address_space_enum::global, "only support global mem");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_buffer_atomic_add_raw<remove_cvref_t<T>,
                                  t_per_x,
                                  Coherence,
                                  oob_conditional_check,
                                  pre_nop>(
            x, p_data_, i, linear_offset, is_valid_element, buffer_size_);
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void
    atomic_max(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(get_address_space() == address_space_enum::global, "only support global mem");

#if CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64
        using scalar_t = typename vector_traits<remove_cvref_t<T>>::scalar_type;
        bool constexpr use_amd_buffer_addressing = std::is_same_v<remove_cvref_t<scalar_t>, double>;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        if constexpr(use_amd_buffer_addressing)
        {
            amd_buffer_atomic_max<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i + linear_offset, is_valid_element, buffer_size_);
        }
        else if(is_valid_element)
        {
            atomic_max_g<remove_cvref_t<T>, t_per_x>(&p_data_[i + linear_offset], x);
        }
    }

    template <typename TDMConfig_,
              typename DimTuple_,
              typename BoxDim_,
              index_t num_tensor_dims,
              typename GatherIndexView_ = null_buffer_view,
              index_t gather_index_offset>
    CK_TILE_DEVICE void tdm_get(const TDMConfig_& tdm_config,
                                CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                                index_t linear_offset,
                                const DimTuple_& tensor_dims,
                                const DimTuple_& global_strides,
                                number<num_tensor_dims>                   = {},
                                const GatherIndexView_& gather_index_view = null_buffer_view{},
                                number<gather_index_offset>               = {})
    {
        // Convert tensor dimensions to uint32_t array
        array<uint32_t, num_tensor_dims> tensor_dims_uint32;
        static_for<0, num_tensor_dims, 1>{}(
            [&](auto i) { tensor_dims_uint32(i) = static_cast<uint32_t>(tensor_dims[i]); });

        // Convert global strides to uint64_t array
        // Note: gfx1250 SPG mentiones tensor_dim1_stride is in units of tensor_dim0_stride
        array<uint64_t, num_tensor_dims> global_strides_uint64;
        static_for<0, num_tensor_dims, 1>{}(
            [&](auto i) { global_strides_uint64(i) = static_cast<uint64_t>(global_strides[i]); });

        // Convert box dimensions to uint16_t array
        constexpr auto box_dim = BoxDim_{};
        constexpr auto box_dim_uint16 =
            generate_array([&](auto i) { return static_cast<uint16_t>(box_dim.at(i)); },
                           number<num_tensor_dims>{});

        auto TDMDescriptor = [&]() {
            if constexpr(std::is_same_v<GatherIndexView_, null_buffer_view>)
            {
                return createTDMDescriptor<remove_cvref_t<T>, num_tensor_dims>(
                    p_data_ + linear_offset,
                    smem,
                    tensor_dims_uint32.data,
                    global_strides_uint64.data,
                    box_dim_uint16.data,
                    tdm_config);
            }
            else
            {
                using GatherIndexType = typename GatherIndexView_::type;

                constexpr TDMGatherIndexSize tdm_index_size =
                    std::is_same_v<GatherIndexType, int32_t> ? TDMGatherIndexSize::Row32bit_Index
                                                             : TDMGatherIndexSize::Row16bit_Index;

                return createTDMDescriptor<remove_cvref_t<T>, num_tensor_dims, true>(
                    p_data_ + linear_offset,
                    smem,
                    tensor_dims_uint32.data,
                    global_strides_uint64.data,
                    box_dim_uint16.data,
                    tdm_config,
                    gather_index_view.p_data_ + gather_index_offset,
                    tdm_index_size);
            }
        }();

        amd_tdm_load<Coherence>(TDMDescriptor);
    }

    template <typename TDMConfig_, typename DimTuple_, typename BoxDim_, index_t num_tensor_dims>
    CK_TILE_DEVICE void tdm_store(const TDMConfig_& tdm_config,
                                  CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                                  index_t linear_offset,
                                  const DimTuple_& tensor_dims,
                                  const DimTuple_& global_strides,
                                  number<num_tensor_dims> = {})
    {
        // Convert tensor dimensions to uint32_t array
        array<uint32_t, num_tensor_dims> tensor_dims_uint32;
        static_for<0, num_tensor_dims, 1>{}(
            [&](auto i) { tensor_dims_uint32(i) = static_cast<uint32_t>(tensor_dims[i]); });

        // Convert global strides to uint64_t array
        array<uint64_t, num_tensor_dims> global_strides_uint64;
        static_for<0, num_tensor_dims, 1>{}(
            [&](auto i) { global_strides_uint64(i) = static_cast<uint64_t>(global_strides[i]); });

        // Convert box dimensions to uint16_t array
        constexpr auto box_dim = BoxDim_{};
        constexpr auto box_dim_uint16 =
            generate_array([&](auto i) { return static_cast<uint16_t>(box_dim.at(i)); },
                           number<num_tensor_dims>{});

        auto TDMDescriptor =
            createTDMDescriptor<remove_cvref_t<T>, num_tensor_dims>(p_data_ + linear_offset,
                                                                    smem,
                                                                    tensor_dims_uint32.data,
                                                                    global_strides_uint64.data,
                                                                    box_dim_uint16.data,
                                                                    tdm_config);

        amd_tdm_store<Coherence>(TDMDescriptor);
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }
};

// Address Space: LDS
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::lds,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view([[clang::lifetimebound]] T* __restrict__ p_data,
                                              BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view([[clang::lifetimebound]] T* __restrict__ p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() {}

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::lds;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    // static_offset is compile-time offset for LDS access optimization
    template <typename X,
              index_t static_offset      = 0,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t linear_offset,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i + linear_offset + static_offset]), sizeof(X));

            return tmp;
#else
            constexpr index_t load_elts = scalar_per_t_vector * scalar_per_x_vector;
            if constexpr(load_elts == 12 && sizeof(typename X::value_type) == 1)
            {
                auto rtn = reinterpret_cast<const int32_t*>(p_data_) +
                           (i + linear_offset + static_offset) / 4;
                struct
                {
                    int32_t x, y, z;
                } tmp = {rtn[0], rtn[1], rtn[2]};
                return bit_cast<X>(tmp);
            }
            else
            {
                using buf_t = ext_vector_t<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                           scalar_per_t_vector * scalar_per_x_vector>;
                auto rtn    = *c_style_pointer_cast<const buf_t*>(
                    &p_data_[i + linear_offset + static_offset]);
                return bit_cast<X>(rtn);
            }
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get_raw(remove_cvref_t<X>& dst,
                                          index_t v_offset,
                                          index_t i_offset,
                                          bool /*is_valid_element*/,
                                          bool_constant<pre_nop> = {}) const
    {
        smem_load<sizeof(X)>{}(dst, v_offset * sizeof(T), i_offset * sizeof(T));
    }

    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto transpose_get([[maybe_unused]] index_t i,
                                                [[maybe_unused]] index_t linear_offset,
                                                bool is_valid_element) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if defined(__gfx950__) || defined(__gfx125__)
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;
            return amd_transpose_load_to_vgpr<remove_cvref_t<T>, t_per_x>(p_data_ + i +
                                                                          linear_offset);
#else
            return X{numeric<remove_cvref_t<T>>::zero()};
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, linear_offset, is_valid_element);
            this->template set<X>(i, linear_offset, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
        bool constexpr workaround_int8_ds_write_issue = true;
#else
        bool constexpr workaround_int8_ds_write_issue = false;
#endif

        i += linear_offset; // simplicity
        if constexpr(std::is_same_v<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                    int8_t> &&
                     workaround_int8_ds_write_issue)
        {
            if(is_valid_element)
            {
                // HACK: compiler would lower IR "store<i8, 16> address_space(3)" into inefficient
                // ISA, so I try to let compiler emit IR "store<i32, 4>" which would be lower to
                // ds_write_b128
                // TODO: remove this after compiler fix
                // clang-format off
                static_assert(
                    (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8_t>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8x2_t>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8x4_t>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8x8_t>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8x16_t>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8x4_t> && std::is_same_v<remove_cvref_t<X>, int8x4_t>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8x8_t> && std::is_same_v<remove_cvref_t<X>, int8x8_t>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8x16_t> && std::is_same_v<remove_cvref_t<X>, int8x16_t>) ||
                        // int8 on thread buffer
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 16>>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 12>>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 8>>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 4>>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 2>>) ||
                        (std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 1>>) ||
                        // ext_vector_type for pk_int4 must use int8_t as type
                        (std::is_same_v<remove_cvref_t<T>, pk_int4_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 1>>) ||
                        (std::is_same_v<remove_cvref_t<T>, pk_int4_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 2>>) ||
                        (std::is_same_v<remove_cvref_t<T>, pk_int4_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 4>>) ||
                        (std::is_same_v<remove_cvref_t<T>, pk_int4_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 8>>) ||
                        (std::is_same_v<remove_cvref_t<T>, pk_int4_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 16>>) ||
                        (std::is_same_v<remove_cvref_t<T>, pk_int4x4_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 4>>) ||
                        (std::is_same_v<remove_cvref_t<T>, pk_int4x8_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 8>>) ||
                        (std::is_same_v<remove_cvref_t<T>, pk_int4x16_t> && std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 16>>),
                    "wrong! not implemented for this combination, please add "
                    "implementation");
                // clang-format on

                if constexpr((std::is_same_v<remove_cvref_t<T>, int8_t> &&
                              std::is_same_v<remove_cvref_t<X>, int8_t>) ||
                             (std::is_same_v<remove_cvref_t<T>, int8_t> &&
                              std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 1>>) ||
                             (std::is_same_v<remove_cvref_t<T>, pk_int4_t> &&
                              std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 1>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int8_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int8_t*>(&x);
                }
                else if constexpr((std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, int8x2_t>) ||
                                  (std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 2>>) ||
                                  (std::is_same_v<remove_cvref_t<T>, pk_int4_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 2>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int16_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int16_t*>(&x);
                }
                else if constexpr((std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, int8x4_t>) ||
                                  (std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 4>>) ||
                                  (std::is_same_v<remove_cvref_t<T>, pk_int4_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 4>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr((std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, int8x8_t>) ||
                                  (std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 8>>) ||
                                  (std::is_same_v<remove_cvref_t<T>, pk_int4_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 8>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr(std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 12>>)
                {
                    *c_style_pointer_cast<dwordx3_union*>(&p_data_[i]) =
                        *c_style_pointer_cast<const dwordx3_union*>(&x);
                }
                else if constexpr((std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, int8x16_t>) ||
                                  (std::is_same_v<remove_cvref_t<T>, int8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<int8_t, 16>>) ||
                                  (std::is_same_v<remove_cvref_t<T>, pk_int4_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 16>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
                else if constexpr((std::is_same_v<remove_cvref_t<T>, int8x4_t> &&
                                   std::is_same_v<remove_cvref_t<X>, int8x4_t>) ||
                                  (std::is_same_v<remove_cvref_t<T>, pk_int4x4_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 4>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr((std::is_same_v<remove_cvref_t<T>, int8x8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, int8x8_t>) ||
                                  (std::is_same_v<remove_cvref_t<T>, pk_int4x8_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 8>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr((std::is_same_v<remove_cvref_t<T>, int8x16_t> &&
                                   std::is_same_v<remove_cvref_t<X>, int8x16_t>) ||
                                  (std::is_same_v<remove_cvref_t<T>, pk_int4x16_t> &&
                                   std::is_same_v<remove_cvref_t<X>, thread_buffer<pk_int4_t, 16>>))
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
                else
                {
                    static_assert(false,
                                  "wrong! not implemented for this combination, please add "
                                  "implementation");
                }
            }
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp = x;

                __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
                using buf_t = ext_vector_t<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                           scalar_per_t_vector * scalar_per_x_vector>;

                *c_style_pointer_cast<buf_t*>(&p_data_[i]) = reinterpret_cast<const buf_t&>(x);
#endif
            }
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }
};

// Address Space: Vgpr
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::vgpr,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() {}

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::vgpr;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              index_t static_offset      = 0,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t /*linear_offset*/,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

            return tmp;
#else
            return *c_style_pointer_cast<const X*>(&p_data_[i]);
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, linear_offset, is_valid_element);
            this->template set<X>(i, linear_offset, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i + linear_offset]), &tmp, sizeof(X));
#else
            *c_style_pointer_cast<X*>(&p_data_[i + linear_offset]) = x;
#endif
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }
};

template <address_space_enum BufferAddressSpace,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename T,
          typename BufferSizeType>
CK_TILE_HOST_DEVICE constexpr auto make_buffer_view([[clang::lifetimebound]] T* __restrict__ p,
                                                    BufferSizeType buffer_size)
{
    return buffer_view<BufferAddressSpace, T, BufferSizeType, true, Coherence>{p, buffer_size};
}

template <address_space_enum BufferAddressSpace,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename T,
          typename BufferSizeType,
          typename X,
          typename std::enable_if<std::is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value,
                                  bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_buffer_view(T* __restrict__ p, BufferSizeType buffer_size, X invalid_element_value)
{
    return buffer_view<BufferAddressSpace, T, BufferSizeType, false, Coherence>{
        p, buffer_size, invalid_element_value};
}

// Generalized print function for all buffer_view variants
template <address_space_enum BufferAddressSpace,
          typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence>
CK_TILE_HOST_DEVICE void print(const buffer_view<BufferAddressSpace,
                                                 T,
                                                 BufferSizeType,
                                                 InvalidElementUseNumericalZeroValue,
                                                 Coherence>& bv)
{
    printf("buffer_view{AddressSpace: %s, p_data_: %p, buffer_size_: ",
           address_space_to_string(BufferAddressSpace),
           static_cast<void*>(const_cast<remove_cvref_t<T>*>(bv.p_data_)));
    print(bv.buffer_size_);
    printf(", invalid_element_value_: ");
    print(bv.invalid_element_value_);
    printf("}");
}

} // namespace ck_tile

#pragma clang diagnostic pop
