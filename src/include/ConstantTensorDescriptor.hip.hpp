#pragma once
#include "common.hip.hpp"

// this is ugly, only for 2d
template <unsigned L0, unsigned L1>
__host__ __device__ constexpr auto calculate_default_strides(Sequence<L0, L1>)
{
    return Sequence<L1, 1>{};
}

// this is ugly, only for 4d
template <unsigned L0, unsigned L1, unsigned L2, unsigned L3>
__host__ __device__ constexpr auto calculate_default_strides(Sequence<L0, L1, L2, L3>)
{
    return Sequence<L1 * L2 * L3, L2 * L3, L3, 1>{};
}

// this is ugly, only for 6d
template <unsigned L0, unsigned L1, unsigned L2, unsigned L3, unsigned L4, unsigned L5>
__host__ __device__ constexpr auto calculate_default_strides(Sequence<L0, L1, L2, L3, L4, L5>)
{
    return Sequence<L1 * L2 * L3 * L4 * L5, L2 * L3 * L4 * L5, L3 * L4 * L5, L4 * L5, L5, 1>{};
}

// this is ugly, only for 8d
template <unsigned L0,
          unsigned L1,
          unsigned L2,
          unsigned L3,
          unsigned L4,
          unsigned L5,
          unsigned L6,
          unsigned L7>
__host__ __device__ constexpr auto
    calculate_default_strides(Sequence<L0, L1, L2, L3, L4, L5, L6, L7>)
{
    return Sequence<L1 * L2 * L3 * L4 * L5 * L6 * L7,
                    L2 * L3 * L4 * L5 * L6 * L7,
                    L3 * L4 * L5 * L6 * L7,
                    L4 * L5 * L6 * L7,
                    L5 * L6 * L7,
                    L6 * L7,
                    L7,
                    1>{};
}

// this is ugly, only for 2d
template <unsigned L0, unsigned L1, unsigned Align>
__host__ __device__ constexpr auto calculate_default_strides_aligned(Sequence<L0, L1>,
                                                                     Number<Align>)
{
    constexpr unsigned L1_align = Align * ((L1 + Align - 1) / Align);
    return Sequence<L1_align, 1>{};
}

// this is ugly, only for 4d
template <unsigned L0, unsigned L1, unsigned L2, unsigned L3, unsigned Align>
__host__ __device__ constexpr auto calculate_default_strides_aligned(Sequence<L0, L1, L2, L3>,
                                                                     Number<Align>)
{
    constexpr unsigned L3_align = Align * ((L3 + Align - 1) / Align);
    return Sequence<L1 * L2 * L3_align, L2 * L3_align, L3_align, 1>{};
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

    __host__ __device__ constexpr unsigned GetElementSize() const
    {
        static_assert(nDim >= 2 && nDim <= 8, "nDim");

        if(nDim == 2)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};

            return GetLength(I0) * GetLength(I1);
        }
        else if(nDim == 3)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};

            return GetLength(I0) * GetLength(I1) * GetLength(I2);
        }
        else if(nDim == 4)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};

            return GetLength(I0) * GetLength(I1) * GetLength(I2) * GetLength(I3);
        }
        else if(nDim == 5)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};

            return GetLength(I0) * GetLength(I1) * GetLength(I2) * GetLength(I3) * GetLength(I4);
        }
        else if(nDim == 6)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};
            constexpr auto I5 = Number<5>{};

            return GetLength(I0) * GetLength(I1) * GetLength(I2) * GetLength(I3) * GetLength(I4) *
                   GetLength(I5);
        }
        else if(nDim == 7)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};
            constexpr auto I5 = Number<5>{};
            constexpr auto I6 = Number<6>{};

            return GetLength(I0) * GetLength(I1) * GetLength(I2) * GetLength(I3) * GetLength(I4) *
                   GetLength(I5) * GetLength(I6);
        }
        else if(nDim == 8)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};
            constexpr auto I5 = Number<5>{};
            constexpr auto I6 = Number<6>{};
            constexpr auto I7 = Number<7>{};

            return GetLength(I0) * GetLength(I1) * GetLength(I2) * GetLength(I3) * GetLength(I4) *
                   GetLength(I5) * GetLength(I6) * GetLength(I7);
        }
        else
        {
            assert(false);
        }
    }

    template <class Align = Number<1>>
    __host__ __device__ constexpr unsigned GetElementSpace(Align align = Align{}) const
    {
        static_assert(nDim >= 2 && nDim <= 8, "nDim");

        constexpr unsigned align_size = align.Get();

        if(nDim == 2)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};

            return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
                   align_size;
        }
        else if(nDim == 3)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};

            return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
                   (GetLength(I2) - 1) * GetStride(I2) + align_size;
        }
        else if(nDim == 4)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};

            return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
                   (GetLength(I2) - 1) * GetStride(I2) + (GetLength(I3) - 1) * GetStride(I3) +
                   align_size;
        }
        else if(nDim == 5)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};

            return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
                   (GetLength(I2) - 1) * GetStride(I2) + (GetLength(I3) - 1) * GetStride(I3) +
                   (GetLength(I4) - 1) * GetStride(I4) + align_size;
        }
        else if(nDim == 6)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};
            constexpr auto I5 = Number<5>{};

            return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
                   (GetLength(I2) - 1) * GetStride(I2) + (GetLength(I3) - 1) * GetStride(I3) +
                   (GetLength(I4) - 1) * GetStride(I4) + (GetLength(I5) - 1) * GetStride(I5) +
                   align_size;
        }
        else if(nDim == 7)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};
            constexpr auto I5 = Number<5>{};
            constexpr auto I6 = Number<6>{};

            return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
                   (GetLength(I2) - 1) * GetStride(I2) + (GetLength(I3) - 1) * GetStride(I3) +
                   (GetLength(I4) - 1) * GetStride(I4) + (GetLength(I5) - 1) * GetStride(I5) +
                   (GetLength(I6) - 1) * GetStride(I6) + align_size;
        }
        else if(nDim == 8)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};
            constexpr auto I3 = Number<3>{};
            constexpr auto I4 = Number<4>{};
            constexpr auto I5 = Number<5>{};
            constexpr auto I6 = Number<6>{};
            constexpr auto I7 = Number<7>{};

            return (GetLength(I0) - 1) * GetStride(I0) + (GetLength(I1) - 1) * GetStride(I1) +
                   (GetLength(I2) - 1) * GetStride(I2) + (GetLength(I3) - 1) * GetStride(I3) +
                   (GetLength(I4) - 1) * GetStride(I4) + (GetLength(I5) - 1) * GetStride(I5) +
                   (GetLength(I6) - 1) * GetStride(I6) + (GetLength(I7) - 1) * GetStride(I7) +
                   align_size;
        }
    }

    // this is ugly, only for 2d
    __host__ __device__ unsigned Get1dIndex(unsigned i0, unsigned i1) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_assert(nDim == 2, "nDim is not 2");
        return i0 * GetStride(I0) + i1 * GetStride(I1);
    }

    // this is ugly, only for 3d
    __host__ __device__ unsigned Get1dIndex(unsigned i0, unsigned i1, unsigned i2) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        static_assert(nDim == 3, "nDim is not 3");
        return i0 * GetStride(I0) + i1 * GetStride(I1) + i2 * GetStride(I2);
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

    // this is ugly, only for 5d
    __host__ __device__ unsigned
    Get1dIndex(unsigned i0, unsigned i1, unsigned i2, unsigned i3, unsigned i4) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        static_assert(nDim == 5, "nDim is not 5");
        return i0 * GetStride(I0) + i1 * GetStride(I1) + i2 * GetStride(I2) + i3 * GetStride(I3) +
               i4 * GetStride(I4);
    }

    // this is ugly, only for 6d
    __host__ __device__ unsigned
    Get1dIndex(unsigned i0, unsigned i1, unsigned i2, unsigned i3, unsigned i4, unsigned i5) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};

        static_assert(nDim == 6, "nDim is not 6");
        return i0 * GetStride(I0) + i1 * GetStride(I1) + i2 * GetStride(I2) + i3 * GetStride(I3) +
               i4 * GetStride(I4) + i5 * GetStride(I5);
    }

    // this is ugly, only for 7d
    __host__ __device__ unsigned Get1dIndex(unsigned i0,
                                            unsigned i1,
                                            unsigned i2,
                                            unsigned i3,
                                            unsigned i4,
                                            unsigned i5,
                                            unsigned i6) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};

        static_assert(nDim == 7, "nDim is not 7");
        return i0 * GetStride(I0) + i1 * GetStride(I1) + i2 * GetStride(I2) + i3 * GetStride(I3) +
               i4 * GetStride(I4) + i5 * GetStride(I5) + i6 * GetStride(I6);
    }

    // this is ugly, only for 8d
    __host__ __device__ unsigned Get1dIndex(unsigned i0,
                                            unsigned i1,
                                            unsigned i2,
                                            unsigned i3,
                                            unsigned i4,
                                            unsigned i5,
                                            unsigned i6,
                                            unsigned i7) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};

        static_assert(nDim == 8, "nDim is not 8");
        return i0 * GetStride(I0) + i1 * GetStride(I1) + i2 * GetStride(I2) + i3 * GetStride(I3) +
               i4 * GetStride(I4) + i5 * GetStride(I5) + i6 * GetStride(I6) + i7 * GetStride(I7);
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

template <class Lengths, unsigned Align>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor_aligned(Lengths, Number<Align>)
{
    using Strides = decltype(calculate_default_strides_aligned(Lengths{}, Number<Align>{}));
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

template <class TDesc>
__host__ __device__ void print_ConstantTensorDescriptor(TDesc, const char* s)
{
    constexpr auto desc     = TDesc{};
    constexpr unsigned ndim = desc.GetDimension();

    static_assert(ndim >= 2 && ndim <= 8, "wrong!");

    if(ndim == 2)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        printf("%s dim %u, lengths {%u %u}, strides {%u %u}\n",
               s,
               desc.GetDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetStride(I0),
               desc.GetStride(I1));
    }
    else if(ndim == 4)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

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
    else if(ndim == 5)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        printf("%s dim %u, lengths {%u %u %u %u %u}, strides {%u %u %u %u %u}\n",
               s,
               desc.GetDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetLength(I2),
               desc.GetLength(I3),
               desc.GetLength(I4),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetStride(I3),
               desc.GetStride(I4));
    }
    else if(ndim == 6)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};

        printf("%s dim %u, lengths {%u %u %u %u %u %u}, strides {%u %u %u %u %u %u}\n",
               s,
               desc.GetDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetLength(I2),
               desc.GetLength(I3),
               desc.GetLength(I4),
               desc.GetLength(I5),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetStride(I3),
               desc.GetStride(I4),
               desc.GetStride(I5));
    }
    else if(ndim == 7)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u}\n",
               s,
               desc.GetDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetLength(I2),
               desc.GetLength(I3),
               desc.GetLength(I4),
               desc.GetLength(I5),
               desc.GetLength(I6),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetStride(I3),
               desc.GetStride(I4),
               desc.GetStride(I5),
               desc.GetStride(I6));
    }
    else if(ndim == 8)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u}\n",
               s,
               desc.GetDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetLength(I2),
               desc.GetLength(I3),
               desc.GetLength(I4),
               desc.GetLength(I5),
               desc.GetLength(I6),
               desc.GetLength(I7),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetStride(I3),
               desc.GetStride(I4),
               desc.GetStride(I5),
               desc.GetStride(I6),
               desc.GetStride(I7));
    }
}
