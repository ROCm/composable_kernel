#pragma once
#include "common.cuh"

// this is ugly, only for 4d
template <unsigned L0, unsigned L1, unsigned L2, unsigned L3>
__host__ __device__ constexpr auto calculate_default_strides(Sequence<L0, L1, L2, L3>)
{
    return Sequence<L1 * L2 * L3, L2 * L3, L3, 1>{};
}

// this is ugly, only for 4d
template <unsigned S0, unsigned S1, unsigned S2, unsigned S3>
__host__ __device__ constexpr auto calculate_full_lengths(Sequence<S0, S1, S2, S3>)
{
    static_assert((S0 % S1 == 0) && (S1 % S2 == 0) && (S2 % S3 == 0), "cannot be evenly divided!");

    return Sequence<1, S0 / S1, S1 / S2, S2 / S3>{};
}

template <class Lengths, class Strides>
struct ConstantTensorDescriptor
{
    static constexpr unsigned nDim = Lengths::nDim;
    using NDimConstant             = Number<nDim>;

    __host__ __device__ constexpr ConstantTensorDescriptor()
    {
        static_assert(Lengths::nDim == Strides::nDim, "nDim not consistent");
    }

    __host__ __device__ constexpr unsigned GetDimension() const { return nDim; }

    __host__ __device__ constexpr Lengths GetLengths() const { return Lengths{}; }

    __host__ __device__ constexpr Strides GetStrides() const { return Strides{}; }

    template <unsigned I>
    __host__ __device__ constexpr unsigned GetLength(Number<I>) const
    {
        return Lengths{}.Get(Number<I>{});
    }

    template <unsigned I>
    __host__ __device__ constexpr unsigned GetStride(Number<I>) const
    {
        return Strides{}.Get(Number<I>{});
    }

    // this is ugly, only for 4d
    __host__ __device__ constexpr unsigned GetElementSize() const
    {
        static_assert(nDim == 4, "nDim is not 4");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        return GetLength(I0) * GetLength(I1) * GetLength(I2) * GetLength(I3);
    }

    // this is ugly, only for 4d
    __host__ __device__ constexpr unsigned GetElementSpace() const
    {
        static_assert(nDim == 4, "nDim is not 4");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
               (GetLength(I2) - 1) * GetStride(I2) + (GetLength(I3) - 1) * GetStride(I3) + 1;
    }

    // this is ugly, only for 4d
    __host__ __device__ unsigned
    Get1dIndex(unsigned i0, unsigned i1, unsigned i2, unsigned i3) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(nDim == 4, "nDim is not 4");
        return i0 * GetStride(I0) + i1 * GetStride(I1) + i2 * GetStride(I2) + i3 * GetStride(I3);
    }

    __host__ __device__ constexpr auto Condense() const
    {
        constexpr auto default_strides = calculate_default_strides(Lengths{});
        return ConstantTensorDescriptor<Lengths, decltype(default_strides)>{};
    }
};

template <class Lengths>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor(Lengths)
{
    using Strides = decltype(calculate_default_strides(Lengths{}));
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

template <class Lengths, class Strides>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor(Lengths, Strides)
{
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

// this is ugly, only for 4d
template <class TDesc>
__host__ __device__ void print_ConstantTensorDescriptor(TDesc, const char* s)
{
    constexpr auto desc = TDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    static_assert(desc.GetDimension() == 4, "dim is not 4");

    printf("%s dim %u, lengths {%u %u %u %u}, strides {%u %u %u %u}\n",
           s,
           desc.GetDimension(),
           desc.GetLength(I0),
           desc.GetLength(I1),
           desc.GetLength(I2),
           desc.GetLength(I3),
           desc.GetStride(I0),
           desc.GetStride(I1),
           desc.GetStride(I2),
           desc.GetStride(I3));
}