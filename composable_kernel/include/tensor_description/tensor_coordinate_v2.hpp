#ifndef CK_TENSOR_COORDINATE_V2_HPP
#define CK_TENSOR_COORDINATE_V2_HPP

#include "common_header.hpp"
#include "dimension.hpp"
#include "dimension_transform.hpp"
#include "tensor_descriptor.hpp"

namespace ck {

template <class NativeTensorDesc>
struct NativeTensorCoordinate
{
    using type             = NativeTensorCoordinate;
    using tensor_desc_type = NativeTensorDesc;
    using Index            = tensor_desc_type::Index;

    static constexpr index_t nDim = Index::GetSize();

    __host__ __device__ constexpr NativeTensorCoordinate(Index idx)
        : mOffset{GetTensorDesriptor().GetOffset(idx)}
    {
    }

    template <class... Xs>
    __host__ __device__ constexpr NativeTensorCoordinate(Xs... xs)
        : NativeTensorCoordinate(Index{xs...})
    {
    }

    template <index_t... Xs>
    __host__ __device__ constexpr NativeTensorCoordinate(Sequence<Xs...>)
        : NativeTensorCoordinate(Index{Xs...})
    {
    }

    __host__ __device__ static constexpr auto GetTensorDescriptor() { return tensor_desc_type{}; }

    __host__ __device__ constexpr index_t GetOffset() const { return mOffset; }

    __host__ __device__ type operator+=(Index idx_diff)
    {
        mOffset += tensor_desc_type::GetOffsetDiff(idx_diff);

        return *this;
    }

    __host__ __device__ type operator-=(Index idx_diff)
    {
        mOffset -= tensor_desc_type::GetOffsetFromMultiIndex(idx_diff);

        return *this;
    }

    __host__ __device__ constexpr type operator+(Index idx_diff) const
    {
        type coord = *this;
        coord += idx_diff;
        return coord;
    }

    __host__ __device__ constexpr type operator-(Index idx_diff) const
    {
        type coord = *this;
        coord -= idx_diff;
        return coord;
    }

    private:
    index_t mOffset;
};

template <class TransformedTensorDesc>
struct TransformedTensorCoordinate
{
    using type             = TransformedTensorCoordinate;
    using tensor_desc_type = TransformedTensorDesc;
    using Index            = tensor_desc_type::UpperIndex;

    using lower_coordinate_type =
        TensorCoordiante_v2<decltype(GetTensorDescriptor().GetLowerTensorDescriptor())>::type;

    static constexpr index_t nDim = Index::GetSize();

    __host__ __device__ constexpr TransformedTensorCoordinate(Index idx)
        : mIndex{idx}, mCoordLow{GetTensorDescriptor().GetLowerIndex(idx)}
    {
    }

    template <class... Xs>
    __host__ __device__ constexpr TransformedTensorCoordinate(Xs... xs)
        : TransformedTensorCoordinate(Index{xs...})
    {
    }

    template <index_t... Xs>
    __host__ __device__ constexpr TransformedTensorCoordinate(Sequence<Xs...>)
        : TransformedTensorCoordinate(Index{Xs...})
    {
    }

    __host__ __device__ static constexpr auto GetTensorDescriptor() { return tensor_desc_type{}; }

    __host__ __device__ constexpr index_t GetOffset() const { return mCoordLow.GetOffset(); }

    __host__ __device__ constexpr Index GetIndex() const { return mIndex; }

    __host__ __device__ type operator+=(Index idx_up_diff)
    {
        // For transformation of multi-index difference, not all transformation functions need to
        //   know the old lower-index or the old upper-index. We pass both of them to the
        //   transformation function. The transformation function itself decides to use them or not.
        mCoordLow +=
            tensor_desc_type::GetLowerIndexDiff(idx_up_diff, mIndexUp, mCoordLow.GetIndex());

        // mIndexUp is updated here, but some (or all) of its entries may never be used
        mIndexUp += idx_up_diff;

        return *this;
    }

    __host__ __device__ constexpr type operator+(Index idx_up_diff) const
    {
        type coord = *this;
        coord += idx_diff;
        return coord;
    }

    private:
    // mIndexUp may be calculated and update, however, the value of some (or all) of its entries may
    //   never be used. Compiler should be able to remove these entries as well as its calculation
    //   as dead code.
    // TODO: make sure compiler indeed remove these dead code
    Index mIndexUp;
    lower_coordinate_type mCoordLow;
};

template <class TensorDesc>
struct TensorCoordinate_v2
{
    private:
    template <class... Ts>
    __host__ __device__ static constexpr auto
    MakeDummyTensorCoordinate(NativeTensorDescriptor<Ts...>)
    {
        return NativeTensorCoordinate<NativeTensorDescriptor<Ts...>>();
    }

    template <class... Ts>
    __host__ __device__ static constexpr auto
    MakeDummyTensorCoordinate(TransformedTensorDescriptor<Ts...>)
    {
        return TransformedTensorCoordinate<TransformedTensorDescriptor<Ts...>>();
    }

    public:
    using type = decltype(MakeDummyTensorCoordinate(TensorDesc{}));
};
}
#endif
