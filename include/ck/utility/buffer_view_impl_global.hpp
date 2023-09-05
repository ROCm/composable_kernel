// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/buffer_view.hpp"
#include "ck/utility/amd_buffer_addressing.hpp"

namespace ck {

// Address Space: Global
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of TensorView/Tensor
template <typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          AmdBufferCoherenceEnum Coherence>
struct BufferView<AddressSpaceEnum::Global,
                  T,
                  BufferSizeType,
                  InvalidElementUseNumericalZeroValue,
                  Coherence>
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
        return AddressSpaceEnum::Global;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    __device__ constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    __device__ constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __device__ constexpr auto Get(index_t i, bool is_valid_element) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_USE_AMD_BUFFER_LOAD
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
                                                                   Coherence>(
                    p_data_, i, is_valid_element, buffer_size_);
            }
            else
            {
                return amd_buffer_load_invalid_element_return_customized_value<remove_cvref_t<T>,
                                                                               t_per_x,
                                                                               Coherence>(
                    p_data_, i, is_valid_element, buffer_size_, invalid_element_value_);
            }
        }
        else
        {
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
        else if constexpr(Op == InMemoryDataOperationEnum::AtomicAdd)
        {
            this->template AtomicAdd<X>(i, is_valid_element, x);
        }
        else if constexpr(Op == InMemoryDataOperationEnum::AtomicMax)
        {
            this->template AtomicMax<X>(i, is_valid_element, x);
        }
        // FIXME: remove InMemoryDataOperationEnum::Add
        else if constexpr(Op == InMemoryDataOperationEnum::Add)
        {
            auto tmp = this->template Get<X>(i, is_valid_element);
            this->template Set<X>(i, is_valid_element, x + tmp);
            // tmp += x;
            // this->template Set<X>(i, is_valid_element, tmp);
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

#if CK_USE_AMD_BUFFER_STORE
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_store<remove_cvref_t<T>, t_per_x, Coherence>(
                x, p_data_, i, is_valid_element, buffer_size_);
        }
        else
        {
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
    }

    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __device__ void AtomicAdd(index_t i, bool is_valid_element, const X& x)
    {
        using scalar_t = typename scalar_type<remove_cvref_t<T>>::type;

        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(GetAddressSpace() == AddressSpaceEnum::Global, "only support global mem");

#if CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            is_same_v<remove_cvref_t<scalar_t>, int32_t> ||
            is_same_v<remove_cvref_t<scalar_t>, float> ||
            (is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0);
#elif CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && (!CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT)
        bool constexpr use_amd_buffer_addressing = is_same_v<remove_cvref_t<scalar_t>, int32_t>;
#elif(!CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER) && CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            is_same_v<remove_cvref_t<scalar_t>, float> ||
            (is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0);
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_atomic_add<remove_cvref_t<T>, t_per_x, Coherence>(
                x, p_data_, i, is_valid_element, buffer_size_);
        }
        else
        {
            if(is_valid_element)
            {
                atomic_add<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
            }
        }
    }

    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __device__ void AtomicMax(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(GetAddressSpace() == AddressSpaceEnum::Global, "only support global mem");

#if CK_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64
        using scalar_t                           = typename scalar_type<remove_cvref_t<T>>::type;
        bool constexpr use_amd_buffer_addressing = is_same_v<remove_cvref_t<scalar_t>, double>;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_atomic_max<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i, is_valid_element, buffer_size_);
        }
        else if(is_valid_element)
        {
            atomic_max<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
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
        printf("AddressSpace: Global, ");

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
