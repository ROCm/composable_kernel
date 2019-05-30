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

    static constexpr index_t nDim         = sizeof...(OriginalDimMergeSeqs);
    static constexpr index_t nOriginalDim = OriginalTensorDesc::GetNumOfDimension();

    __host__ __device__ constexpr ConstantMergedTensorDescriptor()
    {
        static_assert(nDim <= nOriginalDim, "wrong!");

        // TODO: check each of OriginalDimMergeSeqs contains at least 1, and at most
        // OriginalTensorDesc::nDim number of dimensions

        // TODO: check OriginalDimMergeSeqs contains all original dimensions

        // TODO: check there is no duplication in OriginalDimMergeSeqs
    }

    __host__ __device__ static constexpr auto GetOriginalTensorDescriptor()
    {
        return OriginalTensorDesc{};
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return nDim; }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetContainedOriginalDimensions(Number<IDim>)
    {
        return std::get<IDim>(mOriginalDimMergeSeqs);
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool ContainMultipleOriginalDimensions(Number<IDim>)
    {
        return (std::get<IDim>(mOriginalDimMergeSeqs).GetSize() > 1);
    }

    template <index_t IDim>
    __host__ __device__ static constexpr index_t GetLength(Number<IDim>)
    {
        constexpr auto original_dims_partial = std::get<IDim>(mOriginalDimMergeSeqs);

        return OriginalTensorDesc::Extract(original_dims_partial).GetElementSize();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr index_t GetStride(Number<IDim>)
    {
        static_assert(!ContainMultipleOriginalDimensions(Number<IDim>{}),
                      "wrong! stride of a merged dimension is undefined");

        constexpr auto idim_original = std::get<IDim>(mOriginalDimMergeSeqs).Front();

        return OriginalTensorDesc::GetStride(Number<idim_original>{});
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        return Sequence<OriginalTensorDesc::Extract(OriginalDimMergeSeqs{}).GetElementSize()...>{};
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
            const auto original_multi_id_partial =
                OriginalTensorDesc::Extract(original_dims_partial)
                    .GetMultiIndexFrom1dIndex(multi_id[idim]);

            static_for<0, original_dims_partial.GetSize(), 1>{}([&](auto I_) {
                constexpr auto I                = decltype(I_){};
                constexpr index_t idim_original = original_dims_partial.Get(I);

                original_multi_id[idim_original] = original_multi_id_partial[I.Get()];
            });
        });

        return original_multi_id;
    }

#if 0 // not needed
    __host__ __device__ static index_t
    GetOffsetFromOriginalMultiIndex(Array<index_t, nOriginalDim> original_multi_id)
    {
        return OriginalTensorDesc::GetOffsetFromMultiIndex(original_multi_id);
    }
#endif

    __host__ __device__ static index_t GetOffsetFromMultiIndexA(Array<index_t, nDim> multi_id)
    {
        const auto original_multi_id = GetOriginalMultiIndexFromMultiIndex(multi_id);

        return OriginalTensorDesc::GetOffsetFromMultiIndex(original_multi_id);
    }

    template <class... Is>
    __host__ __device__ static index_t GetOffsetFromMultiIndex(Is... is)
    {
        return GetOffsetFromMultiIndex(Array<index_t, nDim>{is...});
    }

    __host__ __device__ static Array<index_t, nDim> GetMultiIndexFrom1dIndex(index_t id)
    {
        constexpr auto dummy_desc = make_ConstantTensorDescriptor_default_rank_packed(GetLengths());

        return dummy_desc.GetMultiIndexFrom1dIndex(id);
    }

#if 0 // not needed
    template <index_t IDim>
    __host__ __device__ static index_t GetNewOriginalMultiIndexAfterMovingAlongOneDimension(
        Array<index_t, nOriginalDim> old_original_multi_id, Number<IDim>, index_t step_size)
    {
        auto new_original_multi_id = old_original_multi_id;

        // get partial-original-multi-id corresponding to this merged dimension
        constexpr auto original_partial_dims = std::get<IDim>(mOriginalDimMergeSeqs);

        constexpr auto original_partial_tensor_desc =
            OriginalTensorDesc::Extract(original_partial_dims);

        auto old_original_partial_multi_id =
            extract_array(old_original_mutli_id, original_paritial_dims);

        auto new_original_partial_multi_id =
            original_partial_tensor_desc.GetNewMultiIndexGivenStepSizeOf1dIndex(
                old_original_partial_multi_id, step_size);

        // update original-mutli-id
        static_for<0, original_dims_partial.GetSize(), 1>{}([&](auto I_) {
            constexpr auto I                = decltype(I_){};
            constexpr index_t idim_original = original_dims_partial.Get(I);

            new_original_multi_id[idim_original] = original_multi_id_partial[I.Get()];
        });

        return new_original_multi_id;
    }
#endif
};

template <class OriginalTensorDesc, class... OriginalDimMergeSeqs>
__host__ __device__ constexpr auto make_ConstantMergedTensorDescriptor(OriginalTensorDesc,
                                                                       OriginalDimMergeSeqs...)
{
    return ConstantMergedTensorDescriptor<OriginalTensorDesc, OriginalDimMergeSeqs...>{};
}

template <class TDesc>
__host__ __device__ void print_ConstantMergedTensorDescriptor(TDesc, const char* s)
{
    print_ConstantTensorDescriptor(TDesc::GetOriginalTensorDescriptor(), s);
}
