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
    using Type                    = ConstantTensorDescriptor;
    static constexpr index_t nDim = Lengths::GetSize();

    __host__ __device__ constexpr ConstantTensorDescriptor()
    {
        static_assert(Lengths::GetSize() == Strides::GetSize(), "nDim not consistent");
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return nDim; }

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
        return accumulate_on_sequence(Lengths{}, std::multiplies<index_t>{}, Number<1>{});
    }

    template <class Align = Number<1>>
    __host__ __device__ static constexpr index_t GetElementSpace(Align align = Align{})
    {
        constexpr index_t element_space_unaligned = accumulate_on_sequence(
            (GetLengths() - Number<1>{}) * GetStrides(), std::plus<index_t>{}, Number<1>{});

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

        return accumulate_on_sequence(multi_id * GetStrides(), std::plus<index_t>{}, Number<0>{});
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

    __host__ __device__ static constexpr auto Pack()
    {
        constexpr auto default_strides = calculate_default_strides(Lengths{});
        return ConstantTensorDescriptor<Lengths, decltype(default_strides)>{};
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto Extract(Number<IDims>... extract_dims)
    {
        static_assert(sizeof...(IDims) <= GetNumOfDimension(),
                      "wrong! too many number of dimensions to be extracted");

        return make_ConstantTensorDescriptor(Lengths{}.Extract(extract_dims...),
                                             Strides{}.Extract(extract_dims...));
    }

    template <index_t IDim, index_t SliceLen>
    __host__ __device__ static constexpr auto Slice(Number<IDim>, Number<SliceLen>)
    {
        return make_ConstantTensorDescriptor(Lengths{}.Modify(Number<IDim>{}, Number<SliceLen>{}),
                                             Strides{});
    }

    template <index_t IDim, index_t... FoldIntervals>
    __host__ __device__ static constexpr auto Fold(Number<IDim>, Number<FoldIntervals>...)
    {
        constexpr auto fold_intervals = Sequence<FoldIntervals...>{};

        constexpr index_t fold_intervals_product =
            accumulate_on_sequence(fold_intervals, std::multiplies<index_t>{}, Number<1>{});

        constexpr auto unfold_length = GetLength(Number<IDim>{});
        constexpr auto unfold_stride = GetStride(Number<IDim>{});

        // length of the dimension to be folded needs to be dividable by fold_interval_product,
        // otherwise, folding is invalid
        static_assert(unfold_length % fold_intervals_product == 0,
                      "wrong! length on the dimension to be folded cannot be evenly divided!");

        // folded lengths
        constexpr auto fold_lengths =
            Sequence<unfold_length / fold_intervals_product>{}.Append(fold_intervals);

        // folded strides
        constexpr auto fold_strides =
            Number<unfold_stride>{} *
            reverse_scan_sequence(fold_intervals.PushBack(Number<1>{}), std::multiplies<index_t>{});

        // left and right
        constexpr auto left  = make_increasing_sequence(Number<0>{}, Number<IDim>{}, Number<1>{});
        constexpr auto right = make_increasing_sequence(
            Number<IDim + 1>{}, Number<GetNumOfDimension()>{}, Number<1>{});

        return make_ConstantTensorDescriptor(
            GetLengths().Extract(left).Append(fold_lengths).Append(GetLengths().Extract(right)),
            GetStrides().Extract(left).Append(fold_strides).Append(GetStrides().Extract(right)));
    }

    template <index_t FirstUnfoldDim, index_t LastUnfoldDim>
    __host__ __device__ static constexpr auto Unfold(Number<FirstUnfoldDim>, Number<LastUnfoldDim>)
    {
        static_assert(FirstUnfoldDim >= 0 && LastUnfoldDim < nDim &&
                          FirstUnfoldDim <= LastUnfoldDim,
                      "wrong! should have FirstUnfoldDim <= LastUnfoldDim!");

        // dimensions to be unfold need to be in descending order (w.r.t. strides), and need to be
        // packed in memory, otherwise, unfolding is invalid
        static_for<FirstUnfoldDim, LastUnfoldDim, 1>{}([&](auto IDim) {
            static_assert(
                GetStride(IDim) >= GetStride(Number<IDim.Get() + 1>{}),
                "wrong! dimensions to be unfolded need to be in descending order w.r.t strides");

            static_assert(GetStride(IDim + 1) * GetLength(IDim + 1) == GetStride(IDim),
                          "wrong! dimensions to be unfolded need to be packed");
        });

        // left and right
        constexpr auto left =
            make_increasing_sequence(Number<0>{}, Number<FirstUnfoldDim>{}, Number<1>{});
        constexpr auto middle = make_increasing_sequence(
            Number<FirstUnfoldDim>{}, Number<LastUnfoldDim + 1>{}, Number<1>{});
        constexpr auto right = make_increasing_sequence(
            Number<LastUnfoldDim + 1>{}, Number<GetNumOfDimension()>{}, Number<1>{});

        // length and stride
        constexpr index_t unfold_length = accumulate_on_sequence(
            GetLengths().Extract(middle), std::multiplies<index_t>{}, Number<1>{});

        constexpr index_t unfold_stride = GetStride(Number<LastUnfoldDim>{});

        return make_ConstantTensorDescriptor(GetLengths()
                                                 .Extract(left)
                                                 .PushBack(Number<unfold_length>{})
                                                 .Append(GetLengths().Extract(right)),
                                             GetStrides()
                                                 .Extract(left)
                                                 .PushBack(Number<unfold_stride>{})
                                                 .Append(GetStrides().Extract(right)));
    }

    template <index_t... IRs>
    __host__ __device__ static constexpr auto ReorderGivenNew2Old(Sequence<IRs...> /*new2old*/)
    {
        static_assert(sizeof...(IRs) == GetNumOfDimension(), "wrong! dimension is wrong");
        constexpr auto map_new2old = Sequence<IRs...>{};
        return make_ConstantTensorDescriptor(Lengths{}.ReorderGivenNew2Old(map_new2old),
                                             Strides{}.ReorderGivenNew2Old(map_new2old));
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
    constexpr index_t ndim = desc.GetNumOfDimension();

    static_assert(ndim >= 2 && ndim <= 10, "wrong!");

    if(ndim == 2)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        printf("%s dim %u, lengths {%u %u}, strides {%u %u}\n",
               s,
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
               desc.GetNumOfDimension(),
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
