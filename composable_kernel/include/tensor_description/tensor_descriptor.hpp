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

    template <index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Sequence<IDims...>)
    {
        return Sequence<GetLength(Number<IDims>{})...>{};
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto GetStrides(Sequence<IDims...>)
    {
        return Sequence<GetStride(Number<IDims>{})...>{};
    }

    template <index_t IDim, index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Number<IDim>, Number<IDims>...)
    {
        return GetLengths(Sequence<IDim, IDims...>{});
    }

    template <index_t IDim, index_t... IDims>
    __host__ __device__ static constexpr auto GetStrides(Number<IDim>, Number<IDims>...)
    {
        return GetStrides(Sequence<IDim, IDims...>{});
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        return GetLengths(typename arithmetic_sequence_gen<0, nDim, 1>::type{});
    }

    __host__ __device__ static constexpr auto GetStrides()
    {
        return GetStrides(typename arithmetic_sequence_gen<0, nDim, 1>::type{});
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return accumulate_on_sequence(GetLengths(), math::multiplies<index_t>{}, Number<1>{});
    }

    __host__ __device__ static constexpr index_t GetElementSpace()
    {
        return accumulate_on_sequence(
            (GetLengths() - Number<1>{}) * GetStrides(), math::plus<index_t>{}, Number<1>{});
    }

    // TODO: this cannot return constepxr because of use of lambda
    __host__ __device__ static constexpr index_t CalculateOffset(const Index& idx)
    {
        index_t offset = 0;

        static_for<0, nDim, 1>{}([&](auto idim) { offset += idx[idim] * GetStride(idim); });

        return offset;
    }

    // TODO: remove this
    __host__ __device__ static constexpr index_t GetOffsetFromMultiIndex(const Index& idx)
    {
        return CalculateOffset(idx);
    }

    __host__ __device__ static constexpr index_t CalculateOffsetDiff(const Index& idx_diff)
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

    // TODO: should this function be here? should it be specific for padding check?
    __host__ __device__ static constexpr bool IsUpperIndexInPaddingArea(const Index& /* idx */)
    {
        return false;
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
        //   transformation, a low-dimension should be associated with only one transformation

        // TODO: sanity-check: GetLowerLengths of each transform should be consistent with lengths
        //   of lower-tensor-descriptor
    }

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        return GetNumOfUpperDimension();
    }

    __host__ __device__ static constexpr auto GetLowerTensorDescriptor()
    {
        return LowTensorDescriptor{};
    }

    __host__ __device__ static constexpr auto GetLowerLengths()
    {
        return GetLowerTensorDescriptor().GetLengths();
    }

    struct lambda_GetUpperLengths
    {
        template <typename Transform>
        __host__ __device__ constexpr auto operator()(const Transform& tran) const
        {
            return tran.GetUpperLengths();
        }
    };

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        constexpr auto tuple_of_up_lengths =
            transform_tuple(lambda_GetUpperLengths{}, Transforms{});

        constexpr auto mingled_up_lengths = unpack(lambda_merge_sequences{}, tuple_of_up_lengths);

        constexpr auto mingled_up_dimension_ids =
            unpack(lambda_merge_sequences{}, UpDimensionIds{});

        // TODO: sanity-check mingled_up_dimension_ids contain all upper-dimensions
        // TODO: sanity-check mingled_up_lengths have no conflicting upper-length

        // sort by upper-dimension-ids
        using sort_up_dimension_ids = sequence_unique_sort<decltype(mingled_up_dimension_ids),
                                                           math::less<index_t>,
                                                           math::equal<index_t>>;

        // sanity-check sorted-upper-dimension-ids should be Sequence<0, 1, ... nDimUp-1>
        static_assert(is_same<typename sort_up_dimension_ids::type,
                              typename arithmetic_sequence_gen<0, nDimUp, 1>::type>{},
                      "wrong! UpDimensionIds is not configured correctly");

        constexpr auto sorted2unsorted_map = typename sort_up_dimension_ids::sorted2unsorted_map{};

        constexpr auto sorted_up_lengths =
            pick_sequence_elements(mingled_up_lengths, sorted2unsorted_map);

        return sorted_up_lengths;
    }

    __host__ __device__ static constexpr auto GetLengths() { return GetUpperLengths(); }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetLength(Number<IDim>)
    {
        return GetLengths()[IDim];
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Sequence<IDims...>)
    {
        return Sequence<GetLength(Number<IDims>{})...>{};
    }

    template <index_t IDim, index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Number<IDim>, Number<IDims>...)
    {
        return GetLengths(Sequence<IDim, IDims...>{});
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return accumulate_on_sequence(GetLengths(), math::multiplies<index_t>{}, Number<1>{});
    }

    __host__ __device__ static constexpr index_t GetElementSpace()
    {
        // TODO: Is this the correct definition for transformed tensor?
        return GetLowerTensorDescriptor().GetElementSpace();
    }

    // TODO: right now return value is constexpr because use of non-constepxr lambda
    __host__ __device__ static constexpr LowerIndex CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_part = pick_array_element(idx_up, UpDimensionIds{}.At(itran));
            auto idx_low_part      = pick_array_element(idx_low, LowDimensionIds{}.At(itran));

            // this assume each lower (single) index is only assocaited with one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_part = tran.CalculateLowerIndex(to_array(idx_up_part));
        });

        return idx_low;
    }

    // TODO: right now return value is constexpr because use of non-constepxr lambda
    __host__ __device__ static constexpr LowerIndex CalculateLowerIndexDiff(
        const UpperIndex& idx_up_diff, const UpperIndex& idx_up_old, const LowerIndex& idx_low_old)
    {
        LowerIndex idx_low_diff;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_diff_part =
                pick_array_element(idx_up_diff, UpDimensionIds{}.At(itran));

            const auto idx_up_old_part = pick_array_element(idx_up_old, UpDimensionIds{}.At(itran));

            const auto idx_low_old_part =
                pick_array_element(idx_low_old, LowDimensionIds{}.At(itran));

            auto idx_low_diff_part = pick_array_element(idx_low_diff, LowDimensionIds{}.At(itran));

            // this assume each lower (single) index is associated with only one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_diff_part = tran.CalculateLowerIndexDiff(
                to_array(idx_up_diff_part), to_array(idx_up_old_part), to_array(idx_low_old_part));
        });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr index_t CalculateOffset(const UpperIndex& idx_up)
    {
        return GetLowerTensorDescriptor().CalculateOffset(CalculateLowerIndex(idx_up));
    }

    // TODO: remove this
    __host__ __device__ static constexpr index_t GetOffsetFromMultiIndex(const UpperIndex& idx_up)
    {
        return CalculateOffset(idx_up);
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

    // TODO: should this function be here? should it be specific for padding check?
    __host__ __device__ static constexpr bool IsUpperIndexInPaddingArea(const UpperIndex& idx_up)
    {
        bool flag = false;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_part = pick_array_element(idx_up, UpDimensionIds{}.At(itran));

            flag = flag || tran.IsUpperIndexInPaddingArea(to_array(idx_up_part));
        });

        return flag;
    }
};

template <index_t... Lengths, index_t... Strides>
__host__ __device__ constexpr auto make_native_tensor_descriptor(Sequence<Lengths...>,
                                                                 Sequence<Strides...>)
{
    return NativeTensorDescriptor<NativeDimension<Lengths, Strides>...>{};
}

template <typename Lengths>
__host__ __device__ constexpr auto make_native_tensor_descriptor_packed(Lengths)
{
    constexpr auto strides = reverse_inclusive_scan_sequence(
                                 Lengths::PopFront(), math::multiplies<index_t>{}, Number<1>{})
                                 .PushBack(Number<1>{});

    return make_native_tensor_descriptor(Lengths{}, strides);
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
