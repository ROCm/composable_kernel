#ifndef CK_TENSOR_VIEW_HPP
#define CK_TENSOR_VIEW_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "tensor_coordinate_deprecated.hpp"

namespace ck {

// TensorDesc is ConstantTensorDescriptor or ConstantMergedTensorDescriptor
template <class TensorDesc, class TData>
struct NormalTensorView
{
    using type             = NormalTensorView;
    using tensor_desc_type = TensorDesc;
    using coordinate_type  = typename NormalTensorCoordinate_deprecated<TensorDesc>::type;
    using data_type        = TData;

    static constexpr auto nDim = TensorDesc::GetNumOfDimension();

    __host__ __device__ constexpr NormalTensorView(TData* p_data) : mpData{p_data} {}

    __host__ __device__ constexpr NormalTensorView() : NormalTensorView{nullptr} {}

    __host__ __device__ static constexpr auto GetNumOfDimension() { return nDim; }

    __host__ __device__ static constexpr auto GetLengths() { return TensorDesc::GetLengths(); }

    __host__ __device__ const TData& operator[](coordinate_type coord) const
    {
        return mpData[coord.GetOffset()];
    }

    __host__ __device__ TData& operator()(coordinate_type coord) const
    {
        return mpData[coord.GetOffset()];
    }

    template <class IDim, class DataPerVector>
    __host__ __device__ static constexpr auto IsVectorizationAllowed(IDim, DataPerVector)
    {
        return TensorDesc::IsVectorizationAllowed(IDim{}, DataPerVector{});
    }

    template <class IDim, class DataPerVector>
    __host__ __device__ auto Vectorize(IDim idim, DataPerVector data_per_vector) const
    {
        static_assert(IsVectorizationAllowed(idim, data_per_vector), "wrong!");

        using vector_t = typename vector_type<TData, data_per_vector>::MemoryType;
        return NormalTensorView<decltype(TensorDesc::Vectorize(idim, data_per_vector)), vector_t>(
            reinterpret_cast<vector_t*>(mpData));
    }

    template <index_t... Is>
    __host__ __device__ auto Slice(coordinate_type slice_origin, Sequence<Is...> slice_lengths)
    {
        static_assert(slice_lengths.GetSize() == nDim, "wrong!");

        return NormalTensorView<decltype(TensorDesc::Slice(slice_lengths)), TData>(
            mpData + slice_origin.GetOffset());
    }

    template <class IDim, class SliceLen>
    __host__ __device__ auto
    Slice(coordinate_type slice_origin, IDim idim, SliceLen slice_len) const
    {
        return NormalTensorView<decltype(TensorDesc::Slice(idim, slice_len)), TData>(
            mpData + slice_origin.GetOffset());
    }

    // slice_window is a slicing window on "*this"
    template <class SliceWindow, class T, bool PositiveDirection>
    __device__ void MoveSliceWindow(SliceWindow& slice_window,
                                    T step_sizes,
                                    integral_constant<bool, PositiveDirection>)
    {
        if(PositiveDirection)
        {
            slice_window.mpData += coordinate_type{step_sizes}.GetOffset();
        }
        else
        {
            slice_window.mpData -= coordinate_type{step_sizes}.GetOffset();
        }
    }

    // private:
    data_type* mpData;
};

template <class... Xs, class TData>
__host__ __device__ constexpr auto make_TensorView(ConstantTensorDescriptor<Xs...>, TData* p_data)
{
    return NormalTensorView<ConstantTensorDescriptor<Xs...>, TData>{p_data};
}

} // namespace ck
#endif
