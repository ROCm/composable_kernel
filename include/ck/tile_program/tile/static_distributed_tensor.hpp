// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {

template <typename DataType_, typename StaticTileDistribution_>
struct StaticDistributedTensor
{
    using DataType               = remove_cvref_t<DataType_>;
    using StaticTileDistribution = remove_cvref_t<StaticTileDistribution_>;

    static_assert(StaticTileDistribution::IsStatic(),
                  "wrong! StaticTileDistribution should be known at compile tile");

    using ThreadTensorDesc = remove_cvref_t<decltype(StaticTileDistribution{}.GetYs2DDescriptor())>;

    static constexpr index_t kThreadElementSpaceSize = ThreadTensorDesc{}.GetElementSpaceSize();

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        return StaticTileDistribution::GetNumOfDimensionX();
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        return StaticTileDistribution::GetLengths();
    }

    __host__ __device__ static constexpr auto GetTileDistribution()
    {
        return StaticTileDistribution{};
    }

    __host__ __device__ static constexpr auto GetDistributedSpans()
    {
        return StaticTileDistribution::GetDistributedSpans();
    }

    __host__ __device__ void Initialize(const DataType& x) { thread_buf_.Initialize(x); }

    __host__ __device__ constexpr const auto& GetThreadBuffer() const { return thread_buf_; }

    __host__ __device__ constexpr auto& GetThreadBuffer() { return thread_buf_; }

    __host__ __device__ static constexpr index_t GetThreadBufferSize()
    {
        return kThreadElementSpaceSize;
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths>
    __host__ __device__ auto GetYSlicedThreadData(Sequence<YSliceOrigins...>,
                                                  Sequence<YSliceLengths...>) const
    {
        static_assert(sizeof...(YSliceOrigins) == StaticTileDistribution::NDimY &&
                          sizeof...(YSliceLengths) == StaticTileDistribution::NDimY,
                      "wrong!");

        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     DataType,
                     sliced_thread_tensor_desc.GetElementSpaceSize(),
                     true>
            sliced_thread_data;

        static_ford<Sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + Sequence<YSliceOrigins...>{};

            sliced_thread_data(Number<sliced_thread_tensor_desc.CalculateOffset(idx)>{}) =
                thread_buf_[Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}];
        });

        return sliced_thread_data;
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths, index_t NSlicedData>
    __host__ __device__ void SetYSlicedThreadData(
        Sequence<YSliceOrigins...>,
        Sequence<YSliceLengths...>,
        const StaticBuffer<AddressSpaceEnum::Vgpr, DataType, NSlicedData, true>& sliced_thread_data)
    {
        static_assert(sizeof...(YSliceOrigins) == StaticTileDistribution::NDimY &&
                          sizeof...(YSliceLengths) == StaticTileDistribution::NDimY,
                      "wrong!");

        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        static_ford<Sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + Sequence<YSliceOrigins...>{};

            thread_buf_(Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}) =
                sliced_thread_data[Number<sliced_thread_tensor_desc.CalculateOffset(idx)>{}];
        });
    }

    template <typename TileDistributedIndices>
    __host__ __device__ constexpr const DataType& operator[](TileDistributedIndices) const
    {
        static_assert(is_static_v<TileDistributedIndices>,
                      "wrong! Tile Distributed Indices should be static");

        constexpr auto y_idx =
            GetTileDistribution().GetYIndicesFromDistributedIndices(TileDistributedIndices{});

        return thread_buf_[Number<ThreadTensorDesc{}.CalculateOffset(y_idx)>{}];
    }

    template <typename TileDistributedIndices>
    __host__ __device__ constexpr DataType& operator()(TileDistributedIndices)
    {
        static_assert(is_static_v<TileDistributedIndices>,
                      "wrong! Tile Distributed Indices should be static");

        constexpr auto y_idx =
            GetTileDistribution().GetYIndicesFromDistributedIndices(TileDistributedIndices{});

        return thread_buf_(Number<ThreadTensorDesc{}.CalculateOffset(y_idx)>{});
    }

#if 0
    template <index_t... Ys>
    __host__ __device__ auto GetElementFromYsIndex(Sequence<Ys...> idx_ys) const
    {
        return thread_buf_[Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}];
    }

    template <index_t... Ys>
    __host__ __device__ void SetElementFromYsIndex(Sequence<Ys...> idx_ys, const DataType& v)
    {
        thread_buf_(Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}) = v;
    }
    template <typename TileDistributedIndices>
    __host__ __device__ auto GetElementFromTileDistributedIndices(TileDistributedIndices) const
    {
        static_assert(is_static_v<TileDistributedIndices>, "wrong!");

        constexpr auto y_idx =
            GetTileDistribution().GetYIndicesFromDistributedIndices(TileDistributedIndices{});

        return GetElementFromYsIndex(y_idx);
    }

    template <typename TileDistributedIndices>
    __host__ __device__ void SetElementFromTileDistributedIndices(TileDistributedIndices,
                                                                  const DataType& v)
    {
        static_assert(is_static_v<TileDistributedIndices>, "wrong!");

        constexpr auto y_idx =
            GetTileDistribution().GetYIndicesFromDistributedIndices(TileDistributedIndices{});

        return SetElementFromYsIndex(y_idx, v);
    }
#endif

    //
    StaticBuffer<AddressSpaceEnum::Vgpr, DataType, kThreadElementSpaceSize, true> thread_buf_;
};

template <typename DataType, typename StaticTileDistribution>
__host__ __device__ constexpr auto make_static_distributed_tensor(const StaticTileDistribution&)
{
    return StaticDistributedTensor<remove_cvref_t<DataType>,
                                   remove_cvref_t<StaticTileDistribution>>{};
}

} // namespace tile_program
} // namespace ck
