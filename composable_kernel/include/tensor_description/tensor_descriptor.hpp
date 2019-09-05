#ifndef CK_TENSOR_DESCRIPTOR_HPP
#define CK_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "dimension.hpp"
#include "multi_index_transform.hpp"

namespace ck {

template <class... NativeDimensions>
struct NativeTensorDescriptor
{
    using type                        = NativeTensorDescriptor;
    static constexpr auto mDimensions = Tuple<NativeDimensions...>;
    static constexpr index_t nDim     = mDimensions::GetSize();

    using Index = MultiIndex<nDim>;

    __host__ __device__ static constexpr auto GetNumOfDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetLengths()
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetStrides()
    {
        // not implemented
    }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetLength(Number<IDim>)
    {
        return mDimensions.Get(Number<IDim>{}).GetLength();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetStride(Number<IDim>)
    {
        return mDimensions.Get(Number<IDim>{}).GetStride();
    }

    __host__ __device__ static constexpr index_t GetOffset(Index idx)
    {
        index_t offset = 0;

        static_for<0, nDim, 1>{}([&](auto idim) { offset += idx[idim] * GetStride(idim); });

        return offset;
    }

    __host__ __device__ static constexpr index_t GetOffsetDiff(Index idx_diff)
    {
        index_t offset_diff = 0;

        static_for<0, nDim, 1>{}(
            [&](auto idim) { offset_diff += idx_diff[idim] * GetStride(idim); });

        return offset_diff;
    }

    __host__ __device__ static constexpr auto AreUpperIndex2OffsetTransformLinear();
    {
        // TODO: re-implement "Sequence", so that it can take other data-type (including bool) as
        // element
        return uniform_sequence_gen<nDim, 1>{};
    }

    __host__ __device__ static constexpr auto GetIndependentDimensionGroups()
    {
        // not implemented, should return Tuple<Sequence<0>, Sequence<1>, ...>
        return xxx;
    }
};

// LowerTensorDescriptor
// Transforms: std::tuple<DimensionTransforms...>
// LowerDimensionIds: std::tuple<Sequence<...>>
// UpperDimensionIds: std::tuple<Sequence<...>>
template <class LowTensorDescriptor, class Transforms, class LowDimensionIds, class UpDimensionIds>
struct TransformedTensorDescriptor
{
    using type                       = TransformedTensorDescriptor;
    static constexpr index_t nDimUp  = GetUpperNumOfDimension();
    static constexpr index_t nDimLow = GetLowerNumOfDimension();

    static constexpr index_t nTransform = Transforms::GetSize();

    using UpperIndex = MultiIndex<nDimUp>;
    using LowerIndex = MultiIndex<nDimLow>;

    __host__ __device__ static constexpr TransformedTensorDescriptor()
    {
        static_assert(nTransform == Transforms::GetSize() &&
                          nTransform == LowDimensionIds::GetSize() &&
                          nTransform == UpDimensionIds::GetSize(),
                      "wrong! # of transformations not the same");

        // TODO: sanity check: LowDimensionIds should include all low-dimensions,
        //   UpDimensionIds should include all up-dimensions

        // TODO: sanity check: while a up-dimension could be associated with multille
        // transformation,
        //   a low-dimension should be associated with only one transformation
    }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension()
    {
        // Here, we assume all lower-dimensions are active
        // TODO: sanity-check all lower-dimension are indeed active
        constexpr auto low_active_dims = unique_sort_sequence(
            merge_tuple_of_sequences(LowDimensionIds{}), math::less<index_t>{});

        return low_active_dims.GetSize();
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension()
    {
        constexpr auto up_active_dims =
            unique_sort_sequence(merge_tuple_of_sequences(UpDimensionIds{}), math::less<index_t>{});
        return up_active_dims.GetSize();
    }

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        return GetNumOfUpperDimension();
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        struct lambda_get_upper_lengths
        {
            template <class Transform>
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

    __host__ __device__ static constexpr auto GetLowerTensorDescriptor()
    {
        return LowTensorDescriptor{};
    }

    __host__ __device__ static constexpr index_t GetLowerIndex(UpperIndex idx_up)
    {
        LowerIndex idx_low;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms::Get(itran);

            constexpr auto idx_low_part = pick_array_element(idx_low, LowDimensionIds::Get(itran));
            constexpr auto idx_up_part  = pick_array_element(idx_up, UpDimensionIds::Get(itran));

            // this assume each lower (single) index is only assocaited with one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_part = tran.GetLowerIndex(idx_up_part);
        });

        return idx_low;
    }

    __host__ __device__ static constexpr index_t GetLowerIndexDiff(UpperIndex idx_up_diff,
                                                                   LowerIndex idx_low_old)
    {
        LowerIndex idx_low_diff;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms::Get(itran);

            constexpr auto idx_up_diff_part =
                pick_array_element(idx_up_diff, UpDimensionIds::Get(itran));

            constexpr auto idx_low_diff_part =
                pick_array_element(idx_low_diff, LowDimensionIds::Get(itran));

            constexpr auto idx_low_old_part =
                pick_array_element(idx_low_old, LowDimensionIds::Get(itran));

            // this assume each lower (single) index is associated with only one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_diff_part = tran.GetLowerIndex(idx_up_diff_part, idx_low_old_part);
        });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr index_t GetOffset(UpperIndex idx_up)
    {
        return GetLowerTensorDescriptor().GetOffset(GetLowerIndex(idx_up));
    }

    __host__ __device__ static constexpr auto AreUpperIndex2OffsetTransformLinear();
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetIndependentDimensionGroups()
    {
        // not implemented
    }
};

} // namespace ck
#endif
