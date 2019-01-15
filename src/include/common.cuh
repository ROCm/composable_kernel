#pragma once

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

__device__ unsigned get_thread_local_1d_id() { return threadIdx.x; }

__device__ unsigned get_block_1d_id() { return blockIdx.x; }

template <class T, T N>
struct Constant
{
    static const T mValue = N;

    __host__ __device__ constexpr T Get() const { return mValue; }
};

template <unsigned N>
using Number = Constant<unsigned, N>;

template <unsigned... Is>
struct Sequence
{
    static constexpr unsigned nDim = sizeof...(Is);

    const unsigned mData[nDim] = {Is...};

    template <unsigned I>
    __host__ __device__ constexpr unsigned Get(Number<I>) const
    {
        return mData[I];
    }

    template <unsigned I0, unsigned I1>
    __host__ __device__ constexpr auto Reorder(Number<I0>, Number<I1>) const
    {
        constexpr unsigned IR0 = Get(Number<I0>{});
        constexpr unsigned IR1 = Get(Number<I1>{});

        return Sequence<IR0, IR1>{};
    }

    template <unsigned I0, unsigned I1, unsigned I2>
    __host__ __device__ constexpr auto Reorder(Number<I0>, Number<I1>, Number<I2>) const
    {
        constexpr unsigned IR0 = Get(Number<I0>{});
        constexpr unsigned IR1 = Get(Number<I1>{});
        constexpr unsigned IR2 = Get(Number<I2>{});

        return Sequence<IR0, IR1, IR2>{};
    }

    template <unsigned I0, unsigned I1, unsigned I2, unsigned I3>
    __host__ __device__ constexpr auto Reorder(Number<I0>, Number<I1>, Number<I2>, Number<I3>) const
    {
        constexpr unsigned IR0 = Get(Number<I0>{});
        constexpr unsigned IR1 = Get(Number<I1>{});
        constexpr unsigned IR2 = Get(Number<I2>{});
        constexpr unsigned IR3 = Get(Number<I3>{});

        return Sequence<IR0, IR1, IR2, IR3>{};
    }

    template <unsigned I0, unsigned I1, unsigned I2, unsigned I3>
    __host__ __device__ constexpr auto Reorder(Sequence<I0, I1, I2, I3>) const
    {
        constexpr unsigned IR0 = Get(Number<I0>{});
        constexpr unsigned IR1 = Get(Number<I1>{});
        constexpr unsigned IR2 = Get(Number<I2>{});
        constexpr unsigned IR3 = Get(Number<I3>{});

        return Sequence<IR0, IR1, IR2, IR3>{};
    }
};
