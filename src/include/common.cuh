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
