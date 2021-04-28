#ifndef CK_BUFFER_HPP
#define CK_BUFFER_HPP

#include "statically_indexed_array.hpp"

namespace ck {

template <typename T, index_t N>
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }
};

template <typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<T, N>{};
}

template <typename T>
struct DynamicBuffer
{
    using type = T;

    T* p_data_;

    __host__ __device__ constexpr DynamicBuffer(T* p_data) : p_data_{p_data} {}

    __host__ __device__ constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    __host__ __device__ constexpr T& operator()(index_t i) { return p_data_[i]; }

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ constexpr const auto Get(index_t i) const
    {
        return *reinterpret_cast<const X*>(&p_data_[i]);
    }

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ void Set(index_t i, const X& x)
    {
        *reinterpret_cast<X*>(&p_data_[i]) = x;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return false; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return true; }
};

template <typename T>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p)
{
    return DynamicBuffer<T>{p};
}

} // namespace ck
#endif
