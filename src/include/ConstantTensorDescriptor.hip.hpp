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
    using Type                     = ConstantTensorDescriptor<Lengths, Strides>;
    static constexpr unsigned nDim = Lengths::nDim;

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

    // c++14 doesn't support constexpr lambdas, has to use this trick instead
    struct GetElementSize_f
    {
        template <class IDim>
        __host__ __device__ constexpr unsigned operator()(IDim idim) const
        {
            return Type{}.GetLength(idim);
        }
    };

    __host__ __device__ constexpr unsigned GetElementSize() const
    {
        // c++14 doesn't support constexpr lambdas, has to use this trick instead
        struct multiply
        {
            __host__ __device__ constexpr unsigned operator()(unsigned a, unsigned b) const
            {
                return a * b;
            }
        };

        return static_const_reduce_n<nDim>{}(GetElementSize_f{}, multiply{});
    }

    // c++14 doesn't support constexpr lambdas, has to use this trick instead
    struct GetElementSpace_f
    {
        template <class IDim>
        __host__ __device__ constexpr unsigned operator()(IDim idim) const
        {
            return (Type{}.GetLength(idim) - 1) * Type{}.GetStride(idim);
        }
    };

    template <class Align = Number<1>>
    __host__ __device__ constexpr unsigned GetElementSpace(Align align = Align{}) const
    {
        // c++14 doesn't support constexpr lambdas, has to use this trick instead
        struct add
        {
            __host__ __device__ constexpr unsigned operator()(unsigned a, unsigned b) const
            {
                return a + b;
            }
        };

        return static_const_reduce_n<nDim>{}(GetElementSpace_f{}, add{}) + align.Get();
    }

    template <class... Is>
    __host__ __device__ unsigned Get1dIndex(Is... is) const
    {
        static_assert(sizeof...(Is) == nDim, "number of multi-index is wrong");

        const auto multi_id = Array<unsigned, nDim>(is...);

        unsigned id = 0;

        static_loop_n<nDim>{}([&](auto IDim) {
            constexpr unsigned idim = IDim.Get();
            id += multi_id[idim] * GetStride(IDim);
        });

        return id;
    }

    __host__ __device__ constexpr auto Condense() const
    {
        constexpr auto default_strides = calculate_default_strides(Lengths{});
        return ConstantTensorDescriptor<Lengths, decltype(default_strides)>{};
    }

    template <unsigned IDim, unsigned NVector>
    __host__ __device__ constexpr auto Vectorize(Number<IDim>, Number<NVector>) const
    {
        assert(false); // not implemented
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
