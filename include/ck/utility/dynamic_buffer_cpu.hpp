#ifndef CK_BUFFER_CPU_HPP
#define CK_BUFFER_CPU_HPP

#include "ck/ck.hpp"
#include "enable_if.hpp"
#include "data_type_cpu.hpp"

namespace ck {
namespace cpu {

template <AddressSpaceEnum BufferAddressSpace,
          typename T,
          typename ElementSpaceSize,
          bool InvalidElementUseNumericalZeroValue>
struct DynamicBuffer
{
    using type = T;

    static_assert(BufferAddressSpace ==
                  AddressSpaceEnum::Global); // only valid for global address space on cpu

    T* p_data_;
    ElementSpaceSize element_space_size_;
    T invalid_element_value_ = T{0};

    constexpr DynamicBuffer(T* p_data, ElementSpaceSize element_space_size)
        : p_data_{p_data}, element_space_size_{element_space_size}
    {
    }

    constexpr DynamicBuffer(T* p_data, ElementSpaceSize element_space_size, T invalid_element_value)
        : p_data_{p_data},
          element_space_size_{element_space_size},
          invalid_element_value_{invalid_element_value}
    {
    }

    static constexpr AddressSpaceEnum GetAddressSpace() { return BufferAddressSpace; }

    constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    constexpr T& operator()(index_t i) { return p_data_[i]; }

    // X should be data_type::type, not directly data_type
    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    constexpr auto Get(index_t i, bool is_valid_element) const
    {
        if constexpr(InvalidElementUseNumericalZeroValue)
        {
            X v;
            if(is_valid_element)
                load_vector(v, &p_data_[i]);
            else
                clear_vector(v);
            return v;
        }
        else
        {
            X v;
            if(is_valid_element)
                load_vector(v, &p_data_[i]);
            else
                set_vector(v, invalid_element_value_);
            return v;
        }
    }

    template <InMemoryDataOperationEnum Op,
              typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    void Update(index_t i, bool is_valid_element, const X& x)
    {
        if constexpr(Op == InMemoryDataOperationEnum::Set)
        {
            this->template Set<X>(i, is_valid_element, x);
        }
        else if constexpr(Op == InMemoryDataOperationEnum::Add)
        {
            auto tmp = this->template Get<X>(i, is_valid_element);
            this->template Set<X>(i, is_valid_element, x + tmp);
        }
    }

    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    void Set(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X need to be multiple T");

        if(is_valid_element)
        {
            store_vector(x, &p_data_[i]);
        }
    }

    static constexpr bool IsStaticBuffer() { return false; }

    static constexpr bool IsDynamicBuffer() { return true; }
};

template <AddressSpaceEnum BufferAddressSpace, typename T, typename ElementSpaceSize>
constexpr auto make_dynamic_buffer(T* p, ElementSpaceSize element_space_size)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, true>{p, element_space_size};
}

template <
    AddressSpaceEnum BufferAddressSpace,
    typename T,
    typename ElementSpaceSize,
    typename X,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
constexpr auto
make_dynamic_buffer(T* p, ElementSpaceSize element_space_size, X invalid_element_value)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, false>{
        p, element_space_size, invalid_element_value};
}

} // namespace cpu
} // namespace ck
#endif
