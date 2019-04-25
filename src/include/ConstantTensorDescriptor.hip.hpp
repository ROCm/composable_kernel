#pragma once
#include "common.hip.hpp"

template <class PreviousStrides, class RemainLengths>
__host__ __device__ constexpr auto calculate_default_strides_impl(PreviousStrides, RemainLengths)
{
    constexpr index_t previous_stride = PreviousStrides{}.Front();
    constexpr index_t current_length  = RemainLengths{}.Back();
    constexpr index_t current_stride  = current_length * previous_stride;

    return calculate_default_strides_impl(PreviousStrides{}.PushFront(Number<current_stride>{}),
                                          RemainLengths{}.PopBack());
}

template <class PreviousStrides, index_t L0, index_t L1>
__host__ __device__ constexpr auto calculate_default_strides_impl(PreviousStrides, Sequence<L0, L1>)
{
    constexpr index_t previous_stride = PreviousStrides{}.Front();
    constexpr index_t current_stride  = L1 * previous_stride;

    return PreviousStrides{}.PushFront(Number<current_stride>{});
}

template <class Lengths>
__host__ __device__ constexpr auto calculate_default_strides(Lengths)
{
    return calculate_default_strides_impl(Sequence<1>{}, Lengths{});
}

// this is ugly, only for 2d
template <index_t L0, index_t L1, index_t Align>
__host__ __device__ constexpr auto calculate_default_strides_aligned(Sequence<L0, L1>,
                                                                     Number<Align>)
{
    constexpr index_t L1_align = Align * ((L1 + Align - 1) / Align);
    return Sequence<L1_align, 1>{};
}

// this is ugly, only for 3d
template <index_t L0, index_t L1, index_t L2, index_t Align>
__host__ __device__ constexpr auto calculate_default_strides_aligned(Sequence<L0, L1, L2>,
                                                                     Number<Align>)
{
    constexpr index_t L2_align = Align * ((L2 + Align - 1) / Align);
    return Sequence<L1 * L2_align, L2_align, 1>{};
}

// this is ugly, only for 4d
template <index_t L0, index_t L1, index_t L2, index_t L3, index_t Align>
__host__ __device__ constexpr auto calculate_default_strides_aligned(Sequence<L0, L1, L2, L3>,
                                                                     Number<Align>)
{
    constexpr index_t L3_align = Align * ((L3 + Align - 1) / Align);
    return Sequence<L1 * L2 * L3_align, L2 * L3_align, L3_align, 1>{};
}

template <class Lengths, class Strides>
struct ConstantTensorDescriptor
{
    using Type                    = ConstantTensorDescriptor<Lengths, Strides>;
    static constexpr index_t nDim = Lengths::GetSize();

    __host__ __device__ constexpr ConstantTensorDescriptor()
    {
        static_assert(Lengths::GetSize() == Strides::GetSize(), "nDim not consistent");
    }

    __host__ __device__ static constexpr index_t GetDimension() { return nDim; }

    __host__ __device__ static constexpr Lengths GetLengths() { return Lengths{}; }

    __host__ __device__ static constexpr Strides GetStrides() { return Strides{}; }

    template <index_t I>
    __host__ __device__ static constexpr index_t GetLength(Number<I>)
    {
        return Lengths{}.Get(Number<I>{});
    }

    template <index_t I>
    __host__ __device__ static constexpr index_t GetStride(Number<I>)
    {
        return Strides{}.Get(Number<I>{});
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return accumulate_on_sequence(Lengths{}, mod_conv::multiplies<index_t>{}, Number<1>{});
    }

    // c++14 doesn't support constexpr lambdas, has to use this trick instead
    struct GetElementSpace_f
    {
        template <class IDim>
        __host__ __device__ constexpr index_t operator()(IDim idim) const
        {
            return (Type{}.GetLength(idim) - 1) * Type{}.GetStride(idim);
        }
    };

    template <class Align = Number<1>>
    __host__ __device__ static constexpr index_t GetElementSpace(Align align = Align{})
    {
        index_t element_space_unaligned =
            static_const_reduce_n<nDim>{}(GetElementSpace_f{}, mod_conv::plus<index_t>{}) + 1;

        return align.Get() * ((element_space_unaligned + align.Get() - 1) / align.Get());
    }

    template <index_t NSize>
    __host__ __device__ static index_t Get1dIndex(Array<index_t, NSize> multi_id)
    {
        static_assert(NSize == nDim, "wrong! Dimension not consistent");

        index_t id = 0;

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr index_t idim = IDim.Get();
            id += multi_id[idim] * GetStride(IDim);
        });

        return id;
    }

    template <class... Is>
    __host__ __device__ static index_t Get1dIndex(Is... is)
    {
        static_assert(sizeof...(Is) == nDim, "number of multi-index is wrong");

        const auto multi_id = Array<index_t, nDim>(is...);

        return Get1dIndex(multi_id);
    }

    template <index_t... Is>
    __host__ __device__ static constexpr index_t Get1dIndex(Sequence<Is...> /*multi_id*/)
    {
        static_assert(sizeof...(Is) == nDim, "wrong! Dimension not consistent");

        constexpr auto multi_id = Sequence<Is...>{};

        constexpr auto seq_tmp =
            transform_sequences(mod_conv::multiplies<index_t>{}, multi_id, GetStrides());

        return accumulate_on_sequence(seq_tmp, mod_conv::plus<index_t>{}, Number<0>{});
    }

    __host__ __device__ static Array<index_t, nDim> GetMultiIndex(index_t id)
    {
        Array<index_t, nDim> multi_id;

        static_for<0, nDim - 1, 1>{}([&](auto IDim) {
            constexpr index_t idim = IDim.Get();
            multi_id[idim]         = id / GetStride(IDim);
            id -= multi_id[idim] * GetStride(IDim);
        });

        multi_id[nDim - 1] = id / GetStride(Number<nDim - 1>{});

        return multi_id;
    }

    __host__ __device__ static constexpr auto Condense()
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

template <class Lengths, index_t Align>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor_aligned(Lengths, Number<Align>)
{
    using Strides = decltype(calculate_default_strides_aligned(Lengths{}, Number<Align>{}));
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

template <class TDesc>
__host__ __device__ void print_ConstantTensorDescriptor(TDesc, const char* s)
{
    constexpr auto desc    = TDesc{};
    constexpr index_t ndim = desc.GetDimension();

    static_assert(ndim >= 2 && ndim <= 10, "wrong!");

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
    else if(ndim == 3)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        printf("%s dim %u, lengths {%u %u %u}, strides {%u %u %u}\n",
               s,
               desc.GetDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetLength(I2),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2));
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
    else if(ndim == 9)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};
        constexpr auto I8 = Number<8>{};

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u "
               "%u}\n",
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
               desc.GetLength(I8),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetStride(I3),
               desc.GetStride(I4),
               desc.GetStride(I5),
               desc.GetStride(I6),
               desc.GetStride(I7),
               desc.GetStride(I8));
    }
    else if(ndim == 10)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};
        constexpr auto I8 = Number<8>{};
        constexpr auto I9 = Number<9>{};

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u "
               "%u %u %u}\n",
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
               desc.GetLength(I8),
               desc.GetLength(I9),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetStride(I3),
               desc.GetStride(I4),
               desc.GetStride(I5),
               desc.GetStride(I6),
               desc.GetStride(I7),
               desc.GetStride(I8),
               desc.GetStride(I9));
    }
}
