// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/buffer_view.hpp"

namespace ck {

// Address Space: LDS
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of TensorView/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct BufferView<AddressSpaceEnum::Lds,
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

    __device__ static constexpr AddressSpaceEnum GetAddressSpace() { return AddressSpaceEnum::Lds; }

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

#if CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
        bool constexpr workaround_int8_ds_write_issue = true;
#else
        bool constexpr workaround_int8_ds_write_issue = false;
#endif

        if constexpr(is_same<typename scalar_type<remove_cvref_t<T>>::type, int8_t>::value &&
                     workaround_int8_ds_write_issue)
        {
            if(is_valid_element)
            {
                // HACK: compiler would lower IR "store<i8, 16> address_space(3)" into inefficient
                // ISA, so I try to let compiler emit IR "store<i32, 4>" which would be lower to
                // ds_write_b128
                // TODO: remove this after compiler fix
                static_assert((is_same<remove_cvref_t<T>, int8_t>::value &&
                               is_same<remove_cvref_t<X>, int8_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x2_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x4_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x8_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x16_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8x4_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x4_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8x8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x8_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8x16_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x16_t>::value),
                              "wrong! not implemented for this combination, please add "
                              "implementation");

                if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                             is_same<remove_cvref_t<X>, int8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int8_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int8_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x2_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int16_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int16_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x4_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x16_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8x4_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x4_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8x8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8x16_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x16_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
            }
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

    // FIXME: remove
    __device__ static constexpr bool IsStaticBuffer() { return false; }

    // FIXME: remove
    __device__ static constexpr bool IsDynamicBuffer() { return true; }

    __host__ __device__ void Print() const
    {
        printf("BufferView{");

        // AddressSpace
        printf("AddressSpace: Lds, ");

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
