#ifndef CK_TENSOR_DESCRIPTOR_HPP
#define CK_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "dimension.hpp"
#include "multi_index_transform.hpp"

namespace ck {

template <typename... NativeDimensions>
struct NativeTensorDescriptor
{
    using type                        = NativeTensorDescriptor;
    static constexpr index_t nDim     = sizeof...(NativeDimensions);
    static constexpr auto mDimensions = make_tuple(NativeDimensions{}...);

    using Index = MultiIndex<nDim>;

    __host__ __device__ static constexpr auto GetNumOfDimension() { return Number<nDim>{}; }

    struct lambda_GetLength
    {
        template <typename IDim>
        __host__ __device__ constexpr auto operator()(IDim) const
        {
            return GetLength(IDim{});
        }
    };

    __host__ __device__ static constexpr auto GetLengths()
    {
        return typename sequence_gen<nDim, lambda_GetLength>::type{};
    }

    struct lambda_GetStride
    {
        template <typename IDim>
        __host__ __device__ constexpr auto operator()(IDim) const
        {
            return GetStride(IDim{});
        }
    };

    __host__ __device__ static constexpr auto GetStrides()
    {
        return typename sequence_gen<nDim, lambda_GetStride>::type{};
    }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetLength(Number<IDim>)
    {
        return mDimensions.At(Number<IDim>{}).GetLength();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetStride(Number<IDim>)
    {
        return mDimensions.At(Number<IDim>{}).GetStride();
    }

    __host__ __device__ static constexpr index_t GetOffset(const Index& idx)
    {
        index_t offset = 0;

        static_for<0, nDim, 1>{}([&](auto idim) { offset += idx[idim] * GetStride(idim); });

        return offset;
    }

    __host__ __device__ static constexpr index_t GetOffsetDiff(const Index& idx_diff)
    {
        index_t offset_diff = 0;

        static_for<0, nDim, 1>{}(
            [&](auto idim) { offset_diff += idx_diff[idim] * GetStride(idim); });

        return offset_diff;
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool IsLinearDimension(Number<IDim>)
    {
        return true;
    }

    __host__ __device__ static constexpr auto GetLinearDimensions()
    {
        return typename arithmetic_sequence_gen<0, nDim, 1>::type{};
    }

    __host__ __device__ static constexpr auto GetNonLinearDimensions() { return Sequence<>{}; }

    __host__ __device__ static constexpr auto GetNonLinearIndependentDimensionGroups()
    {
        return Tuple<>{};
    }
};

// LowerTensorDescriptor
// Transforms: Tuple<DimensionTransforms...>
// LowerDimensionIds: Tuple<Sequence<...>>
// UpperDimensionIds: Tuple<Sequence<...>>
template <typename LowTensorDescriptor,
          typename Transforms,
          typename LowDimensionIds,
          typename UpDimensionIds>
struct TransformedTensorDescriptor
{
    using type                          = TransformedTensorDescriptor;
    static constexpr index_t nTransform = Transforms::Size();

    struct lambda_merge_sequences
    {
        template <typename... Seqs>
        __host__ __device__ constexpr auto operator()(Seqs... seqs) const
        {
            return merge_sequences(seqs...);
        }
    };

    __host__ __device__ static constexpr auto GetNumOfLowerDimension()
    {
        // Here, we assume all lower-dimensions are active
        // TODO: sanity-check all lower-dimension are indeed active

        using duplicated_low_active_dims =
            decltype(unpack(lambda_merge_sequences{}, LowDimensionIds{}));

        using low_active_dims = typename sequence_unique_sort<duplicated_low_active_dims,
                                                              math::less<index_t>,
                                                              math::equal<index_t>>::type;

        return low_active_dims::Size();
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension()
    {
        using duplicated_up_active_dims =
            decltype(unpack(lambda_merge_sequences{}, UpDimensionIds{}));

        using up_active_dims = typename sequence_unique_sort<duplicated_up_active_dims,
                                                             math::less<index_t>,
                                                             math::equal<index_t>>::type;

        return up_active_dims::Size();
    }

    static constexpr index_t nDimUp  = GetNumOfUpperDimension();
    static constexpr index_t nDimLow = GetNumOfLowerDimension();

    using UpperIndex = MultiIndex<nDimUp>;
    using LowerIndex = MultiIndex<nDimLow>;

    __host__ __device__ constexpr TransformedTensorDescriptor()
    {
        static_assert(nTransform == Transforms::Size() && nTransform == LowDimensionIds::Size() &&
                          nTransform == UpDimensionIds::Size(),
                      "wrong! # of transformations not the same");

        // TODO: sanity check: LowDimensionIds should include all low-dimensions,
        //   UpDimensionIds should include all up-dimensions

        // TODO: sanity check: while a up-dimension could be associated with multille
        // transformation,
        //   a low-dimension should be associated with only one transformation
    }

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        return GetNumOfUpperDimension();
    }

#if 0
    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        struct lambda_get_upper_lengths
        {
            template <typename Transform>
            __host__ __device__ constexpr auto operator()(Transform tran) const
            {
                return tran.GetUpperLengths();
            }
        };

        constexpr auto tuple_of_upper_lengths =
            transform_tuple(Transforms, lambda_get_upper_lengths{});

        constexpr auto all_upper_lengths = merge_tuple_of_sequences(tuple_of_upper_lengths);

        constexpr auto all_upper_dimension_ids = merge_tuple_of_sequences(UpDimensionIds{});

        // TODO: sanity-check all_upper_dimension_ids contain all upper-dimensions
        // TODO: sanity-check all_upper_lengths have no conflicting upper-length

        using sort_dimension_ids =
            sequence_unique_sort<decltype(all_upper_dimension_ids), math::less<index_t>>;

        constexpr auto sorted_upper_dimension_ids = typename sort_dimension_ids::type;
        constexpr auto sorted2unsorted_map = typename sort_dimension_ids::sorted2unsorted_map_type;

        constexpr auto sorted_upper_lengths =
            sequence_element_pick(all_upper_lengths, sorted2unsorted_map);

        return sorted_upper_lengths;
    }

    __host__ __device__ static constexpr auto GetLengths() { return GetUpperLengths(); }
#endif

    __host__ __device__ static constexpr auto GetLowerTensorDescriptor()
    {
        return LowTensorDescriptor{};
    }

    __host__ __device__ static constexpr LowerIndex GetLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            auto idx_low_part      = pick_array_element(idx_low, LowDimensionIds{}.At(itran));
            const auto idx_up_part = pick_array_element(idx_up, UpDimensionIds{}.At(itran));

            // this assume each lower (single) index is only assocaited with one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_part = tran.GetLowerIndex(to_array(idx_up_part));
        });

        return idx_low;
    }

    __host__ __device__ static constexpr LowerIndex GetLowerIndexDiff(const UpperIndex& idx_up_diff,
                                                                      const LowerIndex& idx_low_old)
    {
        LowerIndex idx_low_diff;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms::At(itran);

            const auto idx_up_diff_part =
                pick_array_element(idx_up_diff, UpDimensionIds::At(itran));

            auto idx_low_diff_part = pick_array_element(idx_low_diff, LowDimensionIds::At(itran));

            const auto idx_low_old_part =
                pick_array_element(idx_low_old, LowDimensionIds::At(itran));

            // this assume each lower (single) index is associated with only one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_diff_part = tran.GetLowerIndex(idx_up_diff_part, idx_low_old_part);
        });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr index_t GetOffset(const UpperIndex& idx_up)
    {
        return GetLowerTensorDescriptor().GetOffset(GetLowerIndex(idx_up));
    }

#if 0
    template <index_t IDim>
    __host__ __device__ static constexpr bool IsLinearDimension(Number<IDim>)
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetLinearDimensions()
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetNonLinearDimensions()
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetNonLinearIndependentDimensionGroups()
    {
        // not implemented
    }
#endif
};

template <index_t... Lengths, index_t... Strides>
__host__ __device__ constexpr auto make_NativeTensorDescriptor(Sequence<Lengths...>,
                                                               Sequence<Strides...>)
{
    return NativeTensorDescriptor<NativeDimension<Lengths, Strides>...>{};
}

template <typename Lengths>
__host__ __device__ constexpr auto make_NativeTensorDescriptor_packed(Lengths)
{
    constexpr auto strides = reverse_inclusive_scan_sequence(
                                 Lengths::PopFront(), math::multiplies<index_t>{}, Number<1>{})
                                 .PushBack(Number<1>{});

    return make_NativeTensorDescriptor(Lengths{}, strides);
}

template <typename LowTensorDescriptor,
          typename Transforms,
          typename LowDimensionIds,
          typename UpDimensionIds>
__host__ __device__ constexpr auto
    transform_tensor_descriptor(LowTensorDescriptor, Transforms, LowDimensionIds, UpDimensionIds)
{
    return TransformedTensorDescriptor<LowTensorDescriptor,
                                       Transforms,
                                       LowDimensionIds,
                                       UpDimensionIds>{};
}

} // namespace ck
#endif
