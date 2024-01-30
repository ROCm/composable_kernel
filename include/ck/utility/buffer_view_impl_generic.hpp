// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/buffer_view_declare.hpp"
#include "ck/utility/generic_memory_space_atomic.hpp"

namespace ck {

// Address Space: Generic
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of TensorView/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct BufferView<AddressSpaceEnum::Generic,
                  T,
                  BufferSizeType,
                  InvalidElementUseNumericalZeroValue,
                  AmdBufferCoherenceEnum::DefaultCoherence>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    __host__ __device__ constexpr BufferView() : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    __host__ __device__ constexpr BufferView(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    __host__ __device__ constexpr BufferView(T* p_data,
                                             BufferSizeType buffer_size,
                                             T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    __device__ static constexpr AddressSpaceEnum GetAddressSpace()
    {
        return AddressSpaceEnum::Generic;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    __device__ constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    __device__ constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check     = true,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __device__ constexpr auto
    Get(index_t i, bool is_valid_element, bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
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
                return X{0};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <InMemoryDataOperationEnum Op,
              typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __device__ void Update(index_t i, bool is_valid_element, const X& x)
    {
        if constexpr(Op == InMemoryDataOperationEnum::Set)
        {
            this->template Set<X>(i, is_valid_element, x);
        }
        // FIXME: remove InMemoryDataOperationEnum::Add
        else if constexpr(Op == InMemoryDataOperationEnum::Add)
        {
            auto tmp = this->template Get<X>(i, is_valid_element);
            this->template Set<X>(i, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __device__ void Set(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
            *c_style_pointer_cast<X*>(&p_data_[i]) = x;
#endif
        }
    }

    // FIXME: remove
    __device__ static constexpr bool IsStaticBuffer() { return false; }

    // FIXME: remove
    __device__ static constexpr bool IsDynamicBuffer() { return true; }

    __host__ __device__ void Print() const
    {
        printf("BufferView{");

        // AddressSpace
        printf("AddressSpace: Generic, ");

        // p_data_
        printf("p_data_: %p, ", static_cast<void*>(const_cast<remove_cvref_t<T>*>(p_data_)));

        // buffer_size_
        printf("buffer_size_: ");
        print(buffer_size_);
        printf(", ");

        // invalid_element_value_
        printf("invalid_element_value_: ");
        print(invalid_element_value_);

        printf("}");
    }
};

} // namespace ck
