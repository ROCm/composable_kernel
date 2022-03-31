#ifndef CK_STATIC_BUFFER_HPP
#define CK_STATIC_BUFFER_HPP

#include "statically_indexed_array.hpp"

namespace ck {

// static buffer for scalar
template <AddressSpaceEnum AddressSpace,
          typename T,
          index_t N,
          bool InvalidElementUseNumericalZeroValue> // TODO remove this bool, no longer needed
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    __host__ __device__ static constexpr AddressSpaceEnum GetAddressSpace() { return AddressSpace; }

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
template <AddressSpaceEnum AddressSpace,
          typename S,
          index_t NumOfVector,
          index_t ScalarPerVector,
          bool InvalidElementUseNumericalZeroValue, // TODO remove this bool, no longer needed,
          typename enable_if<is_scalar_type<S>::value, bool>::type = false>
struct StaticBufferTupleOfVector
    : public StaticallyIndexedArray<vector_type<S, ScalarPerVector>, NumOfVector>
{
    using V    = typename vector_type<S, ScalarPerVector>::type;
    using base = StaticallyIndexedArray<vector_type<S, ScalarPerVector>, NumOfVector>;

    static constexpr auto s_per_v   = Number<ScalarPerVector>{};
    static constexpr auto num_of_v_ = Number<NumOfVector>{};

    __host__ __device__ constexpr StaticBufferTupleOfVector() : base{} {}

    __host__ __device__ static constexpr AddressSpaceEnum GetAddressSpace() { return AddressSpace; }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }

    // Get S
    // i is offset of S
    template <index_t I>
    __host__ __device__ constexpr const S& operator[](Number<I> i) const
    {
        constexpr auto i_v = i / s_per_v;
        constexpr auto i_s = i % s_per_v;

        return base::operator[](i_v).template AsType<S>()[i_s];
    }

    // Set S
    // i is offset of S
    template <index_t I>
    __host__ __device__ constexpr S& operator()(Number<I> i)
    {
        constexpr auto i_v = i / s_per_v;
        constexpr auto i_s = i % s_per_v;

        return base::operator()(i_v).template AsType<S>()(i_s);
    }

    // Get X
    // i is offset of S, not X. i should be aligned to X
    template <typename X,
              index_t I,
              typename enable_if<has_same_scalar_type<S, X>::value, bool>::type = false>
    __host__ __device__ constexpr auto GetAsType(Number<I> i) const
    {
        constexpr auto s_per_x = Number<scalar_type<remove_cvref_t<X>>::vector_size>{};

        static_assert(s_per_v % s_per_x == 0, "wrong! V must  one or multiple X");
        static_assert(i % s_per_x == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;
        constexpr auto i_x = (i % s_per_v) / s_per_x;

        return base::operator[](i_v).template AsType<X>()[i_x];
    }

    // Set X
    // i is offset of S, not X. i should be aligned to X
    template <typename X,
              index_t I,
              typename enable_if<has_same_scalar_type<S, X>::value, bool>::type = false>
    __host__ __device__ constexpr void SetAsType(Number<I> i, X x)
    {
        constexpr auto s_per_x = Number<scalar_type<remove_cvref_t<X>>::vector_size>{};

        static_assert(s_per_v % s_per_x == 0, "wrong! V must contain one or multiple X");
        static_assert(i % s_per_x == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;
        constexpr auto i_x = (i % s_per_v) / s_per_x;

        base::operator()(i_v).template AsType<X>()(i_x) = x;
    }

    // Get read access to vector_type V
    // i is offset of S, not V. i should be aligned to V
    template <index_t I>
    __host__ __device__ constexpr const auto& GetVectorTypeReference(Number<I> i) const
    {
        static_assert(i % s_per_v == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;

        return base::operator[](i_v);
    }

    // Get write access to vector_type V
    // i is offset of S, not V. i should be aligned to V
    template <index_t I>
    __host__ __device__ constexpr auto& GetVectorTypeReference(Number<I> i)
    {
        static_assert(i % s_per_v == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;

        return base::operator()(i_v);
    }

    __host__ __device__ void Clear()
    {
        const index_t numScalars = NumOfVector * ScalarPerVector;

        static_for<0, Number<numScalars>{}, 1>{}([&](auto i) { SetAsType(i, S{0}); });
    }
};

template <AddressSpaceEnum AddressSpace, typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<AddressSpace, T, N, true>{};
}

} // namespace ck
#endif
