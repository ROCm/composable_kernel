#ifndef CK_STATIC_BUFFER_HPP
#define CK_STATIC_BUFFER_HPP

#include "statically_indexed_array.hpp"

namespace ck {

// static buffer for scalar
template <AddressSpaceEnum_t AddressSpace,
          typename T,
          index_t N,
          bool InvalidElementUseNumericalZeroValue> // TODO remove this bool, no longer needed
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    __host__ __device__ static constexpr AddressSpaceEnum_t GetAddressSpace()
    {
        return AddressSpace;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const T& operator[](Number<I> i) const
    {
        return base::operator[](i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr T& operator()(Number<I> i)
    {
        return base::operator()(i);
    }
};

// static buffer for vector
template <AddressSpaceEnum_t AddressSpace,
          typename T,
          index_t NumOfVector,
          index_t ScalarPerVector,
          bool InvalidElementUseNumericalZeroValue> // TODO remove this bool, no longer needed
struct StaticBufferTupleOfVector
    : public StaticallyIndexedArray<vector_type<T, ScalarPerVector>, NumOfVector>
{
    using type = T;
    using base = StaticallyIndexedArray<vector_type<T, ScalarPerVector>, NumOfVector>;

    static constexpr auto scalar_per_vector = Number<ScalarPerVector>{};
    static constexpr auto num_of_vector_    = Number<NumOfVector>{};

    __host__ __device__ constexpr StaticBufferTupleOfVector() : base{} {}

    __host__ __device__ static constexpr AddressSpaceEnum_t GetAddressSpace()
    {
        return AddressSpace;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const T& operator[](Number<I> i) const
    {
        constexpr auto vector_i = i / scalar_per_vector;
        constexpr auto scalar_i = i % scalar_per_vector;

        return base::operator[](vector_i).template AsType<T>()[scalar_i];
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr T& operator()(Number<I> i)
    {
        constexpr auto vector_i = i / scalar_per_vector;
        constexpr auto scalar_i = i % scalar_per_vector;

        return base::operator()(vector_i).template AsType<T>()(scalar_i);
    }
};

template <AddressSpaceEnum_t AddressSpace, typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<AddressSpace, T, N, true>{};
}

} // namespace ck
#endif
