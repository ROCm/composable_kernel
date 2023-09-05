// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"

namespace ck {

template <index_t NDimHidden, typename TopDimensionHiddenIds>
struct TensorCoordinate
    : public TensorAdaptorCoordinate<NDimHidden, Sequence<0>, TopDimensionHiddenIds>
{
    using Base = TensorAdaptorCoordinate<NDimHidden, Sequence<0>, TopDimensionHiddenIds>;

    // TODO make these private
    static constexpr index_t ndim_top_ = TopDimensionHiddenIds::Size();

    using HiddenIndex = MultiIndex<NDimHidden>;
    using TopIndex    = MultiIndex<ndim_top_>;

    public:
    __host__ __device__ constexpr TensorCoordinate() = default;

    __host__ __device__ constexpr TensorCoordinate(const HiddenIndex& idx_hidden) : Base{idx_hidden}
    {
    }

    // construct from TensorAdaptorCoordinte base class
    __host__ __device__ constexpr TensorCoordinate(const Base& adaptor_coord) : Base{adaptor_coord}
    {
    }

    __host__ __device__ constexpr auto GetIndex() const { return Base::GetTopIndex(); }

    __host__ __device__ constexpr index_t GetOffset() const
    {
        return Base::GetBottomIndex()[Number<0>{}];
    }

    __host__ __device__ constexpr const auto& GetHiddenIndex() const
    {
        return Base::GetHiddenIndex();
    }

    __host__ __device__ auto& GetHiddenIndex() { return Base::GetHiddenIndex(); }
};

template <typename TensorDesc, typename TopIndex>
__host__ __device__ constexpr auto make_tensor_coordinate(const TensorDesc& tensor_desc,
                                                          const TopIndex& idx_top)
{
    const auto adaptor_coord = make_tensor_adaptor_coordinate(tensor_desc, idx_top);

    return TensorCoordinate<TensorDesc::GetNumOfHiddenDimension(),
                            remove_cvref_t<decltype(TensorDesc::GetTopDimensionHiddenIds())>>{
        adaptor_coord};
}

template <bool JudgeDoTransforms = true, typename TensorDesc, typename TensorCoord, typename Index>
__host__ __device__ constexpr void
move_tensor_coordinate(const TensorDesc& tensor_desc, TensorCoord& coord, const Index& coord_step)
{
    move_tensor_adaptor_coordinate(tensor_desc, coord, coord_step);
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool
coordinate_has_valid_offset_assuming_top_index_is_valid(const TensorDesc& tensor_desc,
                                                        const TensorCoord& coord)
{
    return adaptor_coordinate_is_valid_assuming_top_index_is_valid(tensor_desc, coord);
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord)
{
    return adaptor_coordinate_is_valid(tensor_desc, coord);
}

} // namespace ck
