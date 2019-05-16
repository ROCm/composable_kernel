#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"

// TensorDesc: ConstantTensorDescriptor<...>
// MergedDimRanges: Sequence<FirstMergedDim, LastMergedDim>
template <class TensorDesc, class... MergedDimRanges>
struct ConstantMergedTensorDescriptor
{
    static constexpr index_t nOriginalDim = GetNumOfOriginalDimension();
    static constexpr index_t nDim         = GetNumOfDimension();

    template <class... Is>
    __host__ __device__ constexpr ConstantMergedTensorDescriptor()
    {
        constexpr auto merged_dim_ranges = std::make_tuple(MergedDimRanges{}...);

        static_for<0, sizeof...(MergedDimRanges), 1>{}([&](auto I) {
            constexpr index_t i             = I.Get();
            constexpr auto merged_dim_range = std::get<i>(merged_dim_ranges);

            static_assert(merged_dim_range.GetSize() == 2,
                          "wrong! should specify first and last dimension to be merged");
            static_assert(merged_dim_range.Get(Number<0>{}) < GetNumOfUnmergedDimension(),
                          "wrong!");
            static_assert(merged_dim_range.Get(Number<1>{}) < GetNumOfUnmergedDimension(),
                          "wrong!");
            static_assert(merged_dim_range.Get(Number<0>{}) <= merged_dim_range.Get(Number<1>{}),
                          "wrong!");
        });
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension()
    {
        constexpr auto merged_dim_ranges = std::make_tuple(MergedDimRanges...);

        struct f_calculate_num_of_lost_dim
        {
            __host__ __device__ constexpr index_t operator()(auto I) const
            {
                constexpr index_t i             = I.Get();
                constexpr auto merged_dim_range = std::get<i>(merged_dim_ranges);

                return merged_dim_range.Get(Number<1>{}) - merged_dim_range.Get(Number<0>{});
            }
        };

        constexpr index_t num_lost_dim = static_const_reduce_n<sizeof...(MergedDimRanges)>{}(
            f_calculate_num_of_lost_dim, std::plus<index_t>{});

        return TensorDesc::GetNumOfDimension() - num_lost_dim;
    }

    __host__ __device__ static constexpr index_t GetNumOfOriginalDimension()
    {
        return TensorDesc::GetNumOfDimension();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool IsMergedDimension(Number<IDim>)
    {
        // not implemented
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool GetLength(Number<IDim>)
    {
        // not implemented
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool GetStride(Number<IDim>)
    {
        static_assert(!IsMergedDimension(Number<IDim>{}, "wrong! stride of a merged dimension is undefined")
        // not implemented
    }

    template <class... Is>
    __host__ __device__ auto MultiIndex2OriginalMultiIndex(Is... is) const
    {
        // not implemented
    }

    template <class... Is>
    __host__ __device__ auto OriginalMultiIndex2MultiIndex(Is... is) const
    {
        // not implemented
    }
};

template <class TensorDesc, class... MergedDimRanges>
constexpr auto make_ConstantMergedTensorDescriptor(TensorDesc, MergedDimRanges...)
{
    return ConstantMergedTensorDescriptor<TensorDesc, MergedDimRanges...>{};
}
