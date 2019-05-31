#pragma once
#include "common.hip.hpp"

template <class Lengths>
__host__ __device__ constexpr auto calculate_tensor_strides_default_rank_packed(Lengths)
{
    return reverse_inclusive_scan_sequence(Lengths{}.PopFront(), mod_conv::multiplies<index_t>{})
        .PushBack(Number<1>{});
}

template <class Lengths, index_t Align>
__host__ __device__ constexpr auto calculate_tensor_strides_default_rank_aligned(Lengths,
                                                                                 Number<Align>)
{
    constexpr index_t L_back_align =
        Align * mod_conv::integer_divide_ceiler<index_t>{}(Lengths{}.Back(), Align);

    return calculate_tensor_strides_default_rank_packed(
        Lengths{}.Modify(Number<Lengths{}.GetSize() - 1>{}, Number<L_back_align>{}));
}

// MemoryRanks of dimensions is for conversion from offset to multi-index
template <class Lengths, class Strides, class MemoryRanks>
struct ConstantTensorDescriptor
{
    using Type = ConstantTensorDescriptor;

    static constexpr index_t nDim = Lengths::GetSize();

    __host__ __device__ constexpr ConstantTensorDescriptor()
    {
        static_assert(Lengths::GetSize() == Strides::GetSize() &&
                          Lengths::GetSize() == MemoryRanks::GetSize(),
                      "nDim not consistent");

#if 0 // require sequence_sort, but it's not implemented yet
        static_assert(is_same<typename sequence_sort<MemoryRanks>::SortedSeqType,
                              typename arithmetic_sequence_gen<0, nDim, 1>::SeqType>::value,
                      "wrong! invalid MemoryRanks");
#endif
    }

    __host__ __device__ static constexpr auto GetOriginalTensorDescriptor() { return Type{}; }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetContainedOriginalDimensions(Number<IDim>)
    {
        return Sequence<IDim>{};
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return nDim; }

    __host__ __device__ static constexpr auto GetLengths() { return Lengths{}; }

    __host__ __device__ static constexpr auto GetStrides() { return Strides{}; }

    __host__ __device__ static constexpr auto GetMemoryRanks() { return MemoryRanks{}; }

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

    template <index_t I>
    __host__ __device__ static constexpr index_t GetMemoryRank(Number<I>)
    {
        return MemoryRanks{}.Get(Number<I>{});
    }

    __host__ __device__ static constexpr bool AreStridesNonAscending()
    {
        bool flag = true;

        static_for<0, nDim - 1, 1>{}([&](auto IDim) {
            constexpr auto IDim_p1 = Number<IDim.Get() + 1>{};

            flag = flag && (GetLength(IDim) >= GetLength(IDim_p1));
        });

        return flag;
    }

    template <class T>
    __host__ __device__ static constexpr bool ContainMultipleOriginalDimensions(T)
    {
        return false;
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return accumulate_on_sequence(Lengths{}, mod_conv::multiplies<index_t>{}, Number<1>{});
    }

    // WRONG! ReorderGivenOld2New is broken
    template <class Align = Number<1>>
    __host__ __device__ static constexpr index_t GetElementSpace(Align align = Align{})
    {
#if 0
        constexpr auto lengths_in_rank = GetLengths().ReorderGivenOld2New(MemoryRank{});
        constexpr auto strides_in_rank = GetStrides().ReorderGivenOld2new(MemoryRank{});

        constexpr index_t element_space_unaligned = accumulate_on_sequence(
            (lengths_in_rank - Number<1>{}) * strides_in_rank, mod_conv::plus<index_t>{}, Number<1>{});
#else // WRONG! align shouldbe applied to the last memory rank, not the last tensor dimension
        constexpr index_t element_space_unaligned = accumulate_on_sequence(
            (GetLengths() - Number<1>{}) * GetStrides(), mod_conv::plus<index_t>{}, Number<1>{});
#endif

        return align.Get() * ((element_space_unaligned + align.Get() - 1) / align.Get());
    }

    template <index_t NSize>
    __host__ __device__ static index_t GetOffsetFromMultiIndex(Array<index_t, NSize> multi_id)
    {
        static_assert(NSize == nDim, "wrong! Dimension not consistent");

        index_t offset = 0;

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr index_t idim = IDim.Get();
            offset += multi_id[idim] * GetStride(IDim);
        });

        return offset;
    }

    template <class... Is>
    __host__ __device__ static index_t GetOffsetFromMultiIndex(Is... is)
    {
        return GetOffsetFromMultiIndex(Array<index_t, sizeof...(Is)>{is...});
    }

    template <index_t... Is>
    __host__ __device__ static constexpr index_t GetOffsetFromMultiIndex(Sequence<Is...>)
    {
        static_assert(sizeof...(Is) == nDim, "wrong! Dimension not consistent");

        constexpr auto multi_id = Sequence<Is...>{};

        return accumulate_on_sequence(
            multi_id * GetStrides(), mod_conv::plus<index_t>{}, Number<0>{});
    }

#if 0 // ReorderGivenOld2new is broken
    __host__ __device__ static Array<index_t, nDim> GetMultiIndexFromOffset(index_t offset)
    {
        Array<index_t, nDim> ranked_multi_id;

        constexpr auto ranked_strides =
            GetStrides().ReorderGivenOld2new(MemoryRanks{}); // check this

        // calculate index in each of the dimensions in the order of their rank (not dimension)
        static_for<0, nDim - 1, 1>{}([&](auto IDim) {
            constexpr index_t idim   = IDim.Get();
            constexpr index_t stride = ranked_strides.Get(Number<idim>{});
            ranked_multi_id[idim]    = offset / stride;
            offset -= ranked_multi_id[idim] * stride;
        });

        ranked_multi_id[nDim - 1] = offset / ranked_strides.Get(Number<nDim - 1>{});

        return reorder_array_given_new2old(ranked_multi_id, MemoryRanks{}); // check this
    }
#endif

    __host__ __device__ static Array<index_t, nDim> GetMultiIndexFrom1dIndex(index_t id)
    {
        Array<index_t, nDim> multi_id;

        constexpr auto dummy_strides = calculate_tensor_strides_default_rank_packed(GetLengths());

        // calculate index in each of the dimensions in the order of their dimension (not rank)
        static_for<0, nDim - 1, 1>{}([&](auto IDim) {
            constexpr index_t idim   = IDim.Get();
            constexpr index_t stride = dummy_strides.Get(Number<idim>{});
            multi_id[idim]           = id / stride;
            id -= multi_id[idim] * stride;
        });

        multi_id[nDim - 1] = id / dummy_strides.Get(Number<nDim - 1>{});

        return multi_id;
    }

    __host__ __device__ static auto
    GetOriginalMultiIndexFromMultiIndex(Array<index_t, nDim> multi_id)
    {
        return multi_id;
    }

    // This function doesn't do carry check on the highest dimension, for performance reason.
    // It is the user's responsibility to make sure the result "new_mutli_id" is not out-of-bound
    // on the highest dimension
    template <bool PositiveDirection>
    __host__ __device__ static Array<index_t, nDim>
    UpdateMultiIndexGivenStepSizeOf1dIndex(Array<index_t, nDim> old_multi_id,
                                           index_t step_size_of_1d_index,
                                           integral_constant<bool, PositiveDirection>)
    {
        Array<index_t, nDim> new_multi_id;

        const auto step_sizes = GetMultiIndexFrom1dIndex(step_size_of_1d_index);

        static_if<PositiveDirection>{}([&](auto) {
            new_multi_id = old_multi_id + step_sizes;

            bool carry = false;

            // do carry check in reversed order, starting from lowest dimension
            // don't check the highest dimension
            static_for<0, nDim, 1>{}([&](auto IDimReverse) {
                constexpr index_t idim = nDim - 1 - IDimReverse.Get();
                constexpr auto IDim    = Number<idim>{};

                if(carry)
                {
                    ++new_multi_id[idim];
                }

                carry = false;

                if(new_multi_id[idim] >= GetLength(IDim))
                {
                    new_multi_id[idim] -= GetLength(IDim);
                    carry = true;
                }
            });
        }).Else([&](auto) {
            // shift up multi-id to avoid unsigned integer underflow during intermediate
            // calculations. After the shift, should have new_multi_id[...] >= 1
            new_multi_id = old_multi_id + (GetLengths() - step_sizes);

            bool borrow = false;

            // do borrow check in reversed order, starting from lowest dimension
            // don't check the highest dimension
            static_for<0, nDim, 1>{}([&](auto IDimReverse) {
                constexpr index_t idim = nDim - 1 - IDimReverse.Get();
                constexpr auto IDim    = Number<idim>{};

                if(borrow)
                {
                    --new_multi_id[idim];
                }

                borrow = false;

                if(new_multi_id[idim] < GetLength(IDim))
                {
                    new_multi_id[idim] += GetLength(IDim);
                    borrow = true;
                }
            });

            // shift back down multi-id
            // here, should have new_multi_id[...] >= GetLengths()
            new_multi_id = new_multi_id - GetLengths();
        });

        return new_multi_id;
    }

    // WRONG! Ranks is broken
    template <index_t... IDims>
    __host__ __device__ static constexpr auto Extract(Number<IDims>... extract_dims)
    {
        static_assert(sizeof...(IDims) <= GetNumOfDimension(),
                      "wrong! too many number of dimensions to be extracted");

        using extract_lengths = decltype(Lengths{}.Extract(extract_dims...));
        using extract_strides = decltype(Strides{}.Extract(extract_dims...));
        using extract_ranks   = decltype(MemoryRanks{}.Extract(extract_dims...));

#if 0
        using new_ranks = typename sequence_sort<extract_ranks>::Original2SortedType;
#else // WRONG! TODO:: implement sequence_sort
        using new_ranks = typename arithmetic_sequence_gen<0, sizeof...(IDims), 1>::SeqType;
#endif

        return ConstantTensorDescriptor<extract_lengths, extract_strides, new_ranks>{};
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto Extract(Sequence<IDims...>)
    {
        return Extract(Number<IDims>{}...);
    }

    template <class... Ts>
    __host__ __device__ static constexpr auto Embed(ConstantTensorDescriptor<Ts...>)
    {
        using leaf_tensor = ConstantTensorDescriptor<Ts...>;

        // memory rank is broken
        // TODO: remove memory rank info from tensor descritpor
        return ConstantTensorDescriptor<decltype(GetLengths().Append(leaf_tensor::GetLengths())),
                                        decltype(GetStrides().Append(leaf_tensor::GetStrides())),
                                        decltype(GetMemoryRanks().Append(
                                            leaf_tensor::GetMemoryRanks()))>{};
    }

    template <index_t IDim, index_t SliceLen>
    __host__ __device__ static constexpr auto Slice(Number<IDim>, Number<SliceLen>)
    {
        using slice_lengths = decltype(Lengths{}.Modify(Number<IDim>{}, Number<SliceLen>{}));

        return ConstantTensorDescriptor<slice_lengths, Strides, MemoryRanks>{};
    }

    template <index_t Threashold, index_t Delta>
    struct f_fold_impl
    {
        __host__ __device__ constexpr index_t operator()(index_t x) const
        {
            return x > Threashold ? x + Delta : x;
        }
    };

    template <index_t IDim, index_t... FoldIntervals>
    __host__ __device__ static constexpr auto Fold(Number<IDim>, Number<FoldIntervals>...)
    {
        constexpr auto fold_intervals = Sequence<FoldIntervals...>{};

        constexpr index_t fold_intervals_product =
            accumulate_on_sequence(fold_intervals, mod_conv::multiplies<index_t>{}, Number<1>{});

        constexpr auto unfold_length = GetLength(Number<IDim>{});
        constexpr auto unfold_stride = GetStride(Number<IDim>{});
        constexpr auto unfold_rank   = GetMemoryRank(Number<IDim>{});

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
            reverse_inclusive_scan_sequence(fold_intervals.PushBack(Number<1>{}),
                                            mod_conv::multiplies<index_t>{});

        // folded_ranks
        constexpr auto fold_ranks =
            typename arithmetic_sequence_gen<unfold_rank,
                                             unfold_rank + fold_intervals.GetSize() + 1,
                                             1>::SeqType{};

        // increase the ranks that are larger than unfold_rank
        constexpr auto tmp_ranks = transform_sequences(
            f_fold_impl<unfold_rank, fold_intervals.GetSize()>{}, GetMemoryRanks());

        // left and right
        constexpr auto left = typename arithmetic_sequence_gen<0, IDim, 1>::SeqType{};
        constexpr auto right =
            typename arithmetic_sequence_gen<IDim + 1, GetNumOfDimension(), 1>::SeqType{};

        constexpr auto new_lengths =
            GetLengths().Extract(left).Append(fold_lengths).Append(GetLengths().Extract(right));
        constexpr auto new_strides =
            GetStrides().Extract(left).Append(fold_strides).Append(GetStrides().Extract(right));
        constexpr auto new_ranks =
            tmp_ranks.Extract(left).Append(fold_ranks).Append(tmp_ranks.Extract(right));

        static_assert(new_ranks.GetSize() == new_lengths.GetSize(), "wrong!");
        static_assert(fold_ranks.GetSize() == fold_lengths.GetSize(), "wrong!");

        return ConstantTensorDescriptor<decltype(new_lengths),
                                        decltype(new_strides),
                                        decltype(new_ranks)>{};
    }

    template <index_t Threashold, index_t Delta>
    struct f_unfold_impl
    {
        __host__ __device__ constexpr index_t operator()(index_t x) const
        {
            return x > Threashold ? x - Delta : x;
        }
    };

    template <index_t FirstUnfoldDim, index_t LastUnfoldDim>
    __host__ __device__ static constexpr auto Unfold(Number<FirstUnfoldDim>, Number<LastUnfoldDim>)
    {
        static_assert(FirstUnfoldDim >= 0 && LastUnfoldDim < nDim &&
                          FirstUnfoldDim <= LastUnfoldDim,
                      "wrong! should have FirstUnfoldDim <= LastUnfoldDim!");

#if 0 // cannot compile: compiler complain about constexpr
        // dimensions to be unfold need to be in descending order (w.r.t. strides), and need to be
        // packed in memory, otherwise, unfolding is invalid
        static_for<FirstUnfoldDim, LastUnfoldDim, 1>{}([&](auto IDim_) {
            constexpr auto IDim    = decltype(IDim_){};
            constexpr auto IDim_p1 = IDim + Number<1>{};

            // check stride
            static_assert(
                GetStride(IDim) >= GetStride(IDim_p1),
                "wrong! dimensions to be unfolded need to be in descending order w.r.t strides");

            // check if packed
            static_assert(GetStride(IDim_p1) * GetLength(IDim_p1) == GetStride(IDim),
                          "wrong! dimensions to be unfolded need to be packed");

            // check ranks
            static_assert(GetMemoryRank(IDim_p1) == GetMemoryRank(IDim) + 1,
                          "wrong! ranks of dimensions to be unfolded need to be in increasing and "
                          "continuous ranks");
        });
#endif

        // left and right
        constexpr auto left = typename arithmetic_sequence_gen<0, FirstUnfoldDim, 1>::SeqType{};
        constexpr auto middle =
            typename arithmetic_sequence_gen<FirstUnfoldDim, LastUnfoldDim + 1, 1>::SeqType{};
        constexpr auto right =
            typename arithmetic_sequence_gen<LastUnfoldDim + 1, GetNumOfDimension(), 1>::SeqType{};

        // unfolded length, stride and rank
        constexpr index_t unfold_length = accumulate_on_sequence(
            GetLengths().Extract(middle), mod_conv::multiplies<index_t>{}, Number<1>{});

        constexpr index_t unfold_stride = GetStride(Number<LastUnfoldDim>{});

        constexpr index_t unfold_rank = GetMemoryRank(Number<FirstUnfoldDim>{});

        // decrease the ranks that are larger than the rank of LastUnfoldDim
        constexpr auto tmp_ranks =
            transform_sequences(f_unfold_impl<GetMemoryRank(Number<LastUnfoldDim>{}),
                                              LastUnfoldDim - FirstUnfoldDim + 1>{},
                                GetMemoryRanks());

        // new lengths, strides and ranks
        constexpr auto new_lengths = GetLengths()
                                         .Extract(left)
                                         .PushBack(Number<unfold_length>{})
                                         .Append(GetLengths().Extract(right));

        constexpr auto new_strides = GetStrides()
                                         .Extract(left)
                                         .PushBack(Number<unfold_stride>{})
                                         .Append(GetStrides().Extract(right));

        constexpr auto new_ranks = tmp_ranks.Extract(left)
                                       .PushBack(Number<unfold_rank>{})
                                       .Append(tmp_ranks.Extract(right));

        return ConstantTensorDescriptor<decltype(new_lengths),
                                        decltype(new_strides),
                                        decltype(new_ranks)>{};
    }

    template <class MapNew2Old>
    __host__ __device__ static constexpr auto ReorderGivenNew2Old(MapNew2Old)
    {
        return ConstantTensorDescriptor<decltype(Lengths{}.ReorderGivenNew2Old(MapNew2Old{})),
                                        decltype(Strides{}.ReorderGivenNew2Old(MapNew2Old{})),
                                        decltype(
                                            MemoryRanks{}.ReorderGivenNew2Old(MapNew2Old{}))>{};
    }

#if 0 // require sequence_sort, which is not implemented yet
    template <class MapOld2New>
    __host__ __device__ static constexpr auto ReorderGivenOld2New(MapOld2New)
    {
        return ConstantTensorDescriptor<decltype(Lengths{}.ReorderGivenOld2New(MapOld2New{})),
                                        decltype(Strides{}.ReorderGivenOld2New(MapOld2New{})),
                                        decltype(
                                            MemoryRanks{}.ReorderGivenOld2New(MapOld2New{}))>{};
    }
#endif
};

template <class Lengths>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor_default_rank_packed(Lengths)
{
    using Strides     = decltype(calculate_tensor_strides_default_rank_packed(Lengths{}));
    using MemoryRanks = typename arithmetic_sequence_gen<0, Lengths::GetSize(), 1>::SeqType;
    return ConstantTensorDescriptor<Lengths, Strides, MemoryRanks>{};
}

template <class Lengths, class Strides>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor_default_rank(Lengths, Strides)
{
    using MemoryRanks = typename arithmetic_sequence_gen<0, Lengths::GetSize(), 1>::SeqType;
    return ConstantTensorDescriptor<Lengths, Strides, MemoryRanks>{};
}

template <class Lengths, index_t Align>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor_default_rank_aligned(Lengths,
                                                                                      Number<Align>)
{
    using Strides =
        decltype(calculate_tensor_strides_default_rank_aligned(Lengths{}, Number<Align>{}));
    using MemoryRanks = typename arithmetic_sequence_gen<0, Lengths::GetSize(), 1>::SeqType;
    return ConstantTensorDescriptor<Lengths, Strides, MemoryRanks>{};
}

template <class TDesc>
__host__ __device__ void print_ConstantTensorDescriptor(TDesc, const char* s)
{
    constexpr index_t ndim = TDesc::GetNumOfDimension();

    static_assert(ndim >= 1 && ndim <= 10, "wrong!");

    static_if<ndim == 1>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u}, strides {%u}, ranks {%u}\n",
               s,
               desc.GetNumOfDimension(),
               desc.GetLength(I0),
               desc.GetStride(I0),
               desc.GetMemoryRank(I0));
    });

    static_if<ndim == 2>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u}, strides {%u %u}, ranks {%u %u}\n",
               s,
               desc.GetNumOfDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1));
    });

    static_if<ndim == 3>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u}, strides {%u %u %u}, ranks {%u %u %u}\n",
               s,
               desc.GetNumOfDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetLength(I2),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2));
    });

    static_if<ndim == 4>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u %u}, strides {%u %u %u %u}, ranks {%u %u %u %u}\n",
               s,
               desc.GetNumOfDimension(),
               desc.GetLength(I0),
               desc.GetLength(I1),
               desc.GetLength(I2),
               desc.GetLength(I3),
               desc.GetStride(I0),
               desc.GetStride(I1),
               desc.GetStride(I2),
               desc.GetStride(I3),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2),
               desc.GetMemoryRank(I3));
    });

    static_if<ndim == 5>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u %u %u}, strides {%u %u %u %u %u}, ranks {%u %u %u %u "
               "%u}\n",
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
               desc.GetStride(I4),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2),
               desc.GetMemoryRank(I3),
               desc.GetMemoryRank(I4));
    });

    static_if<ndim == 6>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u %u %u %u}, strides {%u %u %u %u %u %u}, ranks {%u %u "
               "%u %u %u %u}\n",
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
               desc.GetStride(I5),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2),
               desc.GetMemoryRank(I3),
               desc.GetMemoryRank(I4),
               desc.GetMemoryRank(I5));
    });

    static_if<ndim == 7>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u}, ranks "
               "{%u %u %u %u %u %u %u}\n",
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
               desc.GetStride(I6),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2),
               desc.GetMemoryRank(I3),
               desc.GetMemoryRank(I4),
               desc.GetMemoryRank(I5),
               desc.GetMemoryRank(I6));
    });

    static_if<ndim == 8>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u}, "
               "ranks {%u %u %u %u %u %u %u %u}\n",
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
               desc.GetStride(I7),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2),
               desc.GetMemoryRank(I3),
               desc.GetMemoryRank(I4),
               desc.GetMemoryRank(I5),
               desc.GetMemoryRank(I6),
               desc.GetMemoryRank(I7));
    });

    static_if<ndim == 9>{}([&](auto fwd) {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};
        constexpr auto I8 = Number<8>{};

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u "
               "%u}, ranks {%u %u %u %u %u %u %u %u %u}\n",
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
               desc.GetStride(I8),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2),
               desc.GetMemoryRank(I3),
               desc.GetMemoryRank(I4),
               desc.GetMemoryRank(I5),
               desc.GetMemoryRank(I6),
               desc.GetMemoryRank(I7),
               desc.GetMemoryRank(I8));
    });

    static_if<ndim == 10>{}([&](auto fwd) {
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

        constexpr auto desc = fwd(TDesc{});

        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u "
               "%u %u %u}, ranks {%u %u %u %u %u %u %u %u %u %u}\n",
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
               desc.GetStride(I9),
               desc.GetMemoryRank(I0),
               desc.GetMemoryRank(I1),
               desc.GetMemoryRank(I2),
               desc.GetMemoryRank(I3),
               desc.GetMemoryRank(I4),
               desc.GetMemoryRank(I5),
               desc.GetMemoryRank(I6),
               desc.GetMemoryRank(I7),
               desc.GetMemoryRank(I8),
               desc.GetMemoryRank(I9));
    });
}
