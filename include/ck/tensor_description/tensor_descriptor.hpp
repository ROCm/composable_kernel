// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

// Transforms: Tuple<transforms...>
// LowerDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// UpperDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// TopDimensionHiddenIds> : Sequence<...>
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename TopDimensionHiddenIds,
          typename ElementSpaceSize,
          typename GuaranteedVectorLengths_,
          typename GuaranteedVectorSrides_>
struct TensorDescriptor : public TensorAdaptor<Transforms,
                                               LowerDimensionHiddenIdss,
                                               UpperDimensionHiddenIdss,
                                               Sequence<0>,
                                               TopDimensionHiddenIds>
{
    using Base = TensorAdaptor<Transforms,
                               LowerDimensionHiddenIdss,
                               UpperDimensionHiddenIdss,
                               Sequence<0>,
                               TopDimensionHiddenIds>;

    using ElementSpaceSizeType = ElementSpaceSize;

    constexpr static index_t ntransform_  = Base::GetNumOfTransform();
    constexpr static index_t ndim_hidden_ = Base::GetNumOfHiddenDimension();
    constexpr static index_t ndim_top_    = Base::GetNumOfTopDimension();

    using GuaranteedVectorLengths = GuaranteedVectorLengths_;
    using GuaranteedVectorStrides = GuaranteedVectorSrides_;

    static_assert(GuaranteedVectorLengths::Size() == ndim_hidden_ &&
                      GuaranteedVectorStrides::Size() == ndim_hidden_,
                  "wrong! inconsistent # of hidden dimensions");

    using TopIndex    = MultiIndex<ndim_top_>;
    using HiddenIndex = MultiIndex<ndim_hidden_>;

    public:
    __host__ __device__ constexpr TensorDescriptor() = default;

    __host__ __device__ constexpr TensorDescriptor(const Transforms& transforms,
                                                   ElementSpaceSize element_space_size)
        : Base{transforms}, element_space_size_{element_space_size}

    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionHiddenIdss::Size() == ntransform_ &&
                          UpperDimensionHiddenIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    // construct from TensorAdaptor base class
    __host__ __device__ constexpr TensorDescriptor(const Base& adaptor,
                                                   ElementSpaceSize element_space_size)
        : Base{adaptor}, element_space_size_{element_space_size}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension()
    {
        return Base::GetNumOfTopDimension();
    }

    template <index_t IDim>
    __host__ __device__ constexpr auto GetLength(Number<IDim> idim) const
    {
        return Base::GetTopDimensionLength(idim);
    }

    __host__ __device__ constexpr auto GetLengths() const { return Base::GetTopDimensionLengths(); }

    __host__ __device__ constexpr auto GetElementSpaceSize() const { return element_space_size_; }

    template <typename Idx>
    __host__ __device__ constexpr index_t CalculateOffset(const Idx& idx) const
    {
        return Base::CalculateBottomIndex(idx)[Number<0>{}];
    }

    // TODO make these private
    __host__ __device__ constexpr const auto& GetTransforms() const
    {
        return Base::GetTransforms();
    }

    __host__ __device__ static constexpr auto GetLowerDimensionHiddenIdss()
    {
        return Base::GetLowerDimensionHiddenIdss();
    }

    __host__ __device__ static constexpr auto GetUpperDimensionHiddenIdss()
    {
        return Base::GetUpperDimensionHiddenIdss();
    }

    __host__ __device__ static constexpr auto GetTopDimensionHiddenIds()
    {
        return Base::GetTopDimensionHiddenIds();
    }

    __host__ __device__ static constexpr bool IsStatic()
    {
        return Base::IsKnownAtCompileTime() && is_known_at_compile_time<ElementSpaceSize>::value;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime() { return IsStatic(); }

    __host__ __device__ static constexpr auto GetTopDimensionSafeVectorLengthStrides()
    {
        return Base::GetTopDimensionSafeVectorLengthStrides(
            to_array<index_t, ndim_hidden_>(GuaranteedVectorLengths{}),
            to_array<index_t, ndim_hidden_>(GuaranteedVectorStrides{}));
    }

    __host__ __device__ void Print() const
    {
        printf("TensorDescriptor{");

        // TensorAdaptor
        Base::Print();
        printf(", ");

        // element_space_size_
        printf("element_space_size_: ");
        print(element_space_size_);

        printf("}");
    }

    // TODO make these private
    ElementSpaceSize element_space_size_;
};

template <typename Adaptor, typename ElementSpaceSize>
__host__ __device__ constexpr auto
make_tensor_descriptor_from_adaptor(const Adaptor& adaptor,
                                    const ElementSpaceSize& element_space_size)
{
    constexpr index_t NDimHidden = Adaptor::GetNumOfHiddenDimension();

    return TensorDescriptor<remove_cvref_t<decltype(adaptor.GetTransforms())>,
                            remove_cvref_t<decltype(adaptor.GetLowerDimensionHiddenIdss())>,
                            remove_cvref_t<decltype(adaptor.GetUpperDimensionHiddenIdss())>,
                            remove_cvref_t<decltype(adaptor.GetTopDimensionHiddenIds())>,
                            remove_cvref_t<decltype(element_space_size)>,
                            typename uniform_sequence_gen<NDimHidden, -1>::type,
                            typename uniform_sequence_gen<NDimHidden, -1>::type>{
        adaptor, element_space_size};
}

template <typename OldTensorDescriptor,
          typename NewTransforms,
          typename NewLowerDimensionOldTopIdss,
          typename NewUpperDimensionNewTopIdss>
__host__ __device__ constexpr auto
transform_tensor_descriptor(const OldTensorDescriptor& old_tensor_desc,
                            const NewTransforms& new_transforms,
                            NewLowerDimensionOldTopIdss,
                            NewUpperDimensionNewTopIdss)
{
    const auto element_space_size = old_tensor_desc.GetElementSpaceSize();

    const auto new_tensor_adaptor = transform_tensor_adaptor(old_tensor_desc,
                                                             new_transforms,
                                                             NewLowerDimensionOldTopIdss{},
                                                             NewUpperDimensionNewTopIdss{});

    constexpr index_t NDimHiddenOld = OldTensorDescriptor::GetNumOfHiddenDimension();
    constexpr index_t NDimHiddenNew = decltype(new_tensor_adaptor)::GetNumOfHiddenDimension();

    using NewGuaranteedVectorLengths = typename sequence_merge<
        typename OldTensorDescriptor::GuaranteedVectorLengths,
        typename uniform_sequence_gen<NDimHiddenNew - NDimHiddenOld, -1>::type>::type;

    using NewGuaranteedVectorStrides = typename sequence_merge<
        typename OldTensorDescriptor::GuaranteedVectorStrides,
        typename uniform_sequence_gen<NDimHiddenNew - NDimHiddenOld, -1>::type>::type;

    return TensorDescriptor<
        remove_cvref_t<decltype(new_tensor_adaptor.GetTransforms())>,
        remove_cvref_t<decltype(new_tensor_adaptor.GetLowerDimensionHiddenIdss())>,
        remove_cvref_t<decltype(new_tensor_adaptor.GetUpperDimensionHiddenIdss())>,
        remove_cvref_t<decltype(new_tensor_adaptor.GetTopDimensionHiddenIds())>,
        remove_cvref_t<decltype(element_space_size)>,
        NewGuaranteedVectorLengths,
        NewGuaranteedVectorStrides>{new_tensor_adaptor, element_space_size};
}

} // namespace ck
