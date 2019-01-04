#pragma once
#include "common.cuh"

template <class T, T N>
struct Constant
{
    static const T mValue = N;
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

    template <unsigned I>
    __host__ __device__ constexpr auto GetNumber(Number<I>) const
    {
        constexpr unsigned N = Get(I);

        return Number<N>{};
    }
};

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
    __host__ __device__ unsigned Get1dIndex(unsigned n, unsigned c, unsigned h, unsigned w) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(nDim == 4, "nDim is not 4");
        return n * GetStride(I0) + c * GetStride(I1) + h * GetStride(I2) + w * GetStride(I3);
    }
};

// this is ugly, only for 4d
template <unsigned N, unsigned C, unsigned H, unsigned W>
__host__ __device__ constexpr auto calculate_default_strides(Sequence<N, C, H, W>)
{
    return Sequence<C * H * W, H * W, W, 1>{};
}

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
template <class Desc, class Reorder>
__host__ __device__ constexpr auto get_reordered_4d_tensor_descriptor(Desc, Reorder)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto IT0 = Reorder{}.GetNumber(I0);
    constexpr auto IT1 = Reorder{}.GetNumber(I1);
    constexpr auto IT2 = Reorder{}.GetNumber(I2);
    constexpr auto IT3 = Reorder{}.GetNumber(I3);

    constexpr unsigned L0 = Desc{}.GetLength(IT0);
    constexpr unsigned L1 = Desc{}.GetLength(IT1);
    constexpr unsigned L2 = Desc{}.GetLength(IT2);
    constexpr unsigned L3 = Desc{}.GetLength(IT3);

    return make_ConstantTensorDescriptor(Sequence<L0, L1, L2, L3>{});
}

// this is ugly, only for 4d
template <class InDesc, class WeiDesc>
__host__ __device__ constexpr auto get_convolution_output_4d_tensor_descriptor(InDesc, WeiDesc)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    static_assert(in_desc.GetDimension() == 4, "input nDim is not 4");
    static_assert(wei_desc.GetDimension() == 4, "weight nDim is not 4");
    static_assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1),
                  "input & weight dimension not consistent");

    constexpr auto N  = in_desc.GetLength(I0);
    constexpr auto HI = in_desc.GetLength(I2);
    constexpr auto WI = in_desc.GetLength(I3);

    constexpr auto K = wei_desc.GetLength(I0);
    constexpr auto S = wei_desc.GetLength(I2);
    constexpr auto R = wei_desc.GetLength(I3);

    constexpr auto HO = HI - S + 1;
    constexpr auto WO = WI - R + 1;

    return make_ConstantTensorDescriptor(Sequence<N, K, HO, WO>{});
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
