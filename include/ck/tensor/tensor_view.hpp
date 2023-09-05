// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/tensor_coordinate.hpp"

namespace ck {

template <typename BufferView_, typename TensorDesc_>
struct TensorView
{
    using BufferView  = remove_reference_t<BufferView_>;
    using DataType    = typename BufferView::type;
    using TensorDesc  = remove_cvref_t<TensorDesc_>;
    using TensorIndex = Array<index_t, TensorDesc::GetNumOfTopDimension()>;
    using TensorCoord = decltype(make_tensor_coordinate(TensorDesc{}, TensorIndex{}));

    __host__ __device__ constexpr TensorView() = default;

    __host__ __device__ constexpr TensorView(const BufferView& buffer_view, const TensorDesc& desc)
        : buf_{buffer_view}, desc_{desc}
    {
    }

    __host__ __device__ constexpr auto& GetTensorDescriptor() const { return desc_; }

    __host__ __device__ static constexpr index_t GetNumOfDimension()
    {
        return TensorDesc::GetNumOfTopDimension();
    }

    __host__ __device__ constexpr const auto& GetBufferView() const { return buf_; }

    __host__ __device__ constexpr auto& GetBufferView() { return buf_; }

    __host__ __device__ constexpr DataType GetElement(const TensorCoord& coord) const
    {
        return buf_.template Get<DataType>(
            coord.GetOffset(),
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord));
    }

    __host__ __device__ constexpr void SetElement(const TensorCoord& coord, const DataType& x)
    {
        buf_.template Set<DataType>(
            coord.GetOffset(),
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            x);
    }

    // X is vector of DataType.
    // "coord" is coordinate of DataType, not X. "coord" should be aligned to X
    template <typename X,
              typename enable_if<is_same_v<typename scalar_type<remove_cvref_t<X>>::type,
                                           typename scalar_type<remove_cvref_t<DataType>>::type>,
                                 bool>::type = false>
    __host__ __device__ constexpr remove_cvref_t<X>
    GetVectorizedElements(const TensorCoord& coord) const
    {
        return buf_.template Get<X>(
            coord.GetOffset(),
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord));
    }

    // X is vector of DataType.
    // "coord" is coordinate of DataType, not X. "coord" should be aligned to X
    template <typename X,
              typename enable_if<is_same_v<typename scalar_type<remove_cvref_t<X>>::type,
                                           typename scalar_type<remove_cvref_t<DataType>>::type>,
                                 bool>::type = false>
    __host__ __device__ constexpr void SetVectorizedElements(const TensorCoord& coord, const X& x)
    {
        buf_.template Set<X>(coord.GetOffset(),
                             coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
                             x);
    }

    __host__ __device__ void Print() const
    {
        printf("TensorView{");

        // buf_
        printf("buf_: ");
        print(buf_);
        printf(", ");

        // desc_
        printf("desc_: ");
        print(desc_);

        printf("}");
    }

    // member
    BufferView buf_;
    TensorDesc desc_;
};

template <AddressSpaceEnum BufferAddressSpace = AddressSpaceEnum::Generic,
          typename DataType,
          typename... Ts>
__host__ __device__ constexpr auto make_tensor_view(DataType* p,
                                                    const TensorDescriptor<Ts...>& desc)
{
    auto buffer_view = make_buffer_view<BufferAddressSpace>(p, desc.GetElementSpaceSize());

    return TensorView<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

template <AddressSpaceEnum BufferAddressSpace = AddressSpaceEnum::Generic,
          typename DataType,
          typename... Lengths,
          typename... Strides,
          index_t GuaranteedLastDimensionVectorLength                              = -1,
          index_t GuaranteedLastDimensionVectorStride                              = -1,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto
make_naive_tensor_view(DataType* p,
                       const Tuple<Lengths...>& lengths,
                       const Tuple<Strides...>& strides,
                       Number<GuaranteedLastDimensionVectorLength> = Number<-1>{},
                       Number<GuaranteedLastDimensionVectorStride> = Number<-1>{})
{
    auto desc = make_naive_tensor_descriptor(lengths,
                                             strides,
                                             Number<GuaranteedLastDimensionVectorLength>{},
                                             Number<GuaranteedLastDimensionVectorStride>{});

    auto buffer_view = make_buffer_view<BufferAddressSpace>(p, desc.GetElementSpaceSize());

    return TensorView<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

template <AddressSpaceEnum BufferAddressSpace = AddressSpaceEnum::Generic,
          typename DataType,
          typename... Lengths,
          index_t GuaranteedLastDimensionVectorLength = -1>
__host__ __device__ constexpr auto
make_naive_tensor_view_packed(DataType* p,
                              const Tuple<Lengths...>& lengths,
                              Number<GuaranteedLastDimensionVectorLength> = Number<-1>{})
{
    auto desc =
        make_naive_tensor_descriptor_packed(lengths, Number<GuaranteedLastDimensionVectorLength>{});

    auto buffer_view = make_buffer_view<BufferAddressSpace>(p, desc.GetElementSpaceSize());

    return TensorView<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

template <typename OldTensorView,
          typename NewTransforms,
          typename NewLowerDimensionOldVisibleIdss,
          typename NewUpperDimensionNewVisibleIdss>
__host__ __device__ constexpr auto transform_tensor_view(const OldTensorView& old_tensor_view,
                                                         const NewTransforms& new_transforms,
                                                         NewLowerDimensionOldVisibleIdss,
                                                         NewUpperDimensionNewVisibleIdss)
{
    auto new_desc = transform_tensor_descriptor(old_tensor_view.desc_,
                                                new_transforms,
                                                NewLowerDimensionOldVisibleIdss{},
                                                NewUpperDimensionNewVisibleIdss{});

    return TensorView<typename OldTensorView::BufferView, remove_cvref_t<decltype(new_desc)>>{
        old_tensor_view.buf_, new_desc};
}

} // namespace ck
