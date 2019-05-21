#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"

// OriginalTensorDesc : ConstantTensorDescriptor<...>
//     it's the tensor whose dimensions are to be merged
// OriginalDimMergeSeqs : Sequence<...>...
//     each is a sequence of original dimensions (of OriginalTensorDesc) to be merged
template <class OriginalTensorDesc, class... OriginalDimMergeSeqs>
struct ConstantMergedTensorDescriptor
{
    static constexpr auto mOriginalDimMergeSeqs = std::tuple<OriginalDimMergeSeqs...>{};

    static constexpr index_t nDim         = std::tuple_size<mOriginalDimMergeSeqs>::value;
    static constexpr index_t nOriginalDim = OriginalDesc::GetNumOfDimension();

    __host__ __device__ constexpr ConstantMergedTensorDescriptor()
    {
        static_assert(nDim <= nOriginalDim, "wrong!");

        // TODO: check each of OriginalDimMergeSeqs contains at least 1, and at most
        // OriginalTensorDesc::nDim number of dimensions

        // TODO: check there is no duplication in OriginalDimMergeSeqs

        // TODO: check OriginalDimMergeSeqs contains all original dimensions
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return nDim; }

    __host__ __device__ static constexpr index_t GetNumOfOriginalDimension() { return nOriginalDim }

    template <index_t IDim>
    __host__ __device__ static constexpr bool ContainMultipleOriginalDimensions(Number<IDim>)
    {
        return (std::Get<IDIM>(mOriginalDimMergeSeqs).GetSize() > 1);
    }

    template <index_t IDim>
    __host__ __device__ static constexpr index_t GetLength(Number<IDim>)
    {
        constexpr auto original_dims_partial = std::Get<IDim>(mOriginalDimMergeSeqs);

        return OriginalTensorDesc::Extract(original_dims_partial).GetElementSize();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr index_t GetStride(Number<IDim>)
    {
        static_assert(!ContainMultipleOriginalDimensions(Number<IDim>{}),
                      "wrong! stride of a merged dimension is undefined");

        constexpr auto idim_original = std::Get<IDim>(mOriginalDimMergeSeqs).Front();

        return OriginalTensorDesc::GetStride(Number<idim_original>{});
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        return Sequence<OriginalTensorDesc::Extract(OriginalDimMergeSeqs).GetElementSize()...>{};
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return OriginalTensorDesc::GetElementSize();
    }

    __host__ __device__ static auto
    GetOriginalMultiIndexFromMultiIndex(Array<index_t, nDim> multi_id)
    {
        Array<index_t, nOriginalDim> original_multi_id;

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr index_t idim               = IDim.Get();
            constexpr auto original_dims_partial = std::get<idim>(mOriginalDimMergeSeqs);

            // get partial original-multi-id corresponding to this merged dimension
            constexpr auto original_multi_id_partial =
                OriginalTensorDesc::Extract(original_dims_partial)
                    .GetMultiIndexFrom1dIndex(multi_id[idim]);

            // make sure compiler unroll this loop and propagate all the constants
            for(index_t i = 0; i < original_dims_partial.GetSize(); ++i)
            {
                index_t idim_original = original_dims_partial[i];

                original_multi_id[idim_original] = original_multi_id_partial[i]
            }
        });

        return original_multi_id;
    }

    __host__ __device__ static index_t GetOffsetFromMultiIndex(Array<index_t, nDim> multi_id)
    {
        const auto original_multi_id = GetOriginalMultiIndexFromMultiIndex(multi_id);

        return OriginalTensorDesc::GetOffsetFromMultiIndex(orginal_multi_id);
    }

    template <index_t... Is>
    __host__ __device__ static index_t GetOffsetFromMultiIndex(Is... is)
    {
        return GetOffsetFromMultiIndex(Array<index_t, nDim>{is...});
    }

    __host__ __device__ static Array<index_t, nDim> GetMultiIndexFrom1dIndex(index_t id)
    {
        constexpr auto dummy_desc = make_packed_ConstantTensorDescriptor(GetLengths());

        return dummy_desc.GetMultiIndexFrom1dIndex(id);
    }
};

template <class OriginalTensorDesc, class... OriginalDimMergeSeqs>
constexpr auto make_ConstantMergedTensorDescriptor(OriginalTensorDesc, OriginalDimMergeSeqs...)
{
    return ConstantMergedTensorDescriptor<OriginalTensorDesc, OriginalDimMergeSeqs...>{};
}
