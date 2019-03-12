#pragma once

__device__ unsigned get_thread_local_1d_id() { return threadIdx.x; }

__device__ unsigned get_block_1d_id() { return blockIdx.x; }

template <class T1, class T2>
struct is_same
{
    static const bool value = false;
};

template <class T>
struct is_same<T, T>
{
    static const bool value = true;
};

template <class T, unsigned N>
struct vector_type
{
};

template <>
struct vector_type<float, 1>
{
    using type = float;
};

template <>
struct vector_type<float, 2>
{
    using type = float2;
};

template <>
struct vector_type<float, 4>
{
    using type = float4;
};

#if 0
template <>
struct vector_type<half_float::half, 1>
{
    using type = half_float::half;
};

template <>
struct vector_type<half_float::half, 2>
{
    using type = float;
};

template <>
struct vector_type<half_float::half, 4>
{
    using type = float2;
};

template <>
struct vector_type<half_float::half, 8>
{
    using type = float4;
};
#endif

#if 0
template <>
struct vector_type<half, 1>
{
    using type = half;
};

template <>
struct vector_type<half, 2>
{
    using type = half2;
};

template <>
struct vector_type<half, 4>
{
    using type = float2;
};

template <>
struct vector_type<half, 8>
{
    using type = float4;
};
#endif

template <class T, T N>
struct integral_constant
{
    static const T value = N;

    __host__ __device__ constexpr T Get() const { return value; }
};

template <unsigned N>
using Number = integral_constant<unsigned, N>;

template <unsigned... Is>
struct Sequence
{
    using Type = Sequence<Is...>;

    static constexpr unsigned nDim = sizeof...(Is);

    const unsigned mData[nDim] = {Is...};

    template <unsigned I>
    __host__ __device__ constexpr unsigned Get(Number<I>) const
    {
        return mData[I];
    }

    template <unsigned I0, unsigned I1, unsigned I2, unsigned I3>
    __host__ __device__ constexpr auto ReorderByGetNewFromOld(Sequence<I0, I1, I2, I3>) const
    {
        constexpr auto old_sequence = Type{};

        constexpr unsigned NR0 = old_sequence.mData[I0];
        constexpr unsigned NR1 = old_sequence.mData[I1];
        constexpr unsigned NR2 = old_sequence.mData[I2];
        constexpr unsigned NR3 = old_sequence.mData[I3];

        return Sequence<NR0, NR1, NR2, NR3>{};
    }

    template <unsigned I0, unsigned I1, unsigned I2, unsigned I3>
    __host__ __device__ constexpr auto ReorderByPutOldToNew(Sequence<I0, I1, I2, I3>) const
    {
        // don't know how to implement this
        printf("Sequence::ReorderByPutOldToNew not implemented");
        assert(false);
    }
};

#if DEVICE_BACKEND_CUDA
template <typename T>
__host__ __device__ constexpr T max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__host__ __device__ constexpr T min(T a, T b)
{
    return a < b ? a : b;
}
#endif

__host__ __device__ constexpr unsigned integer_divide_ceil(unsigned a, unsigned b)
{
    return (a + b - 1) / b;
}
