// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0, typename = void>
struct ComputePtrOffsetOfStridedBatch
{
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      ck::enable_if_t<(NumATensor > 1 || NumBTensor > 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(Array<ck::index_t, NumATensor>& BatchStrideAs,
                                   Array<ck::index_t, NumBTensor>& BatchStrideBs,
                                   Array<ck::index_t, NumDTensor>& BatchStrideDs,
                                   index_t BatchStrideE)
        : BatchStrideA_(BatchStrideAs),
          BatchStrideB_(BatchStrideBs),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr auto GetAsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumATensor> as_offset;
        static_for<0, NumATensor, 1>{}(
            [&](auto i) { as_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideA_[i]); });
        return as_offset;
    }

    __host__ __device__ constexpr auto GetBsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumBTensor> bs_offset;
        static_for<0, NumBTensor, 1>{}(
            [&](auto i) { bs_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideB_[i]); });
        return bs_offset;
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideDs_[i]); });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    Array<ck::index_t, NumATensor> BatchStrideA_;
    Array<ck::index_t, NumBTensor> BatchStrideB_;
    Array<ck::index_t, NumDTensor> BatchStrideDs_;
    index_t BatchStrideE_;
    index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      ck::enable_if_t<(NumATensor == 1 && NumBTensor == 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                   index_t BatchStrideB,
                                   Array<ck::index_t, NumDTensor> BatchStrideDs,
                                   index_t BatchStrideE)
        : BatchStrideA_(BatchStrideA),
          BatchStrideB_(BatchStrideB),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideA_);
    }

    __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideB_);
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideDs_[i]); });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    ck::index_t BatchStrideA_;
    ck::index_t BatchStrideB_;
    Array<ck::index_t, NumDTensor> BatchStrideDs_;
    index_t BatchStrideE_;
    index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <bool isTuple, typename Tensors>
constexpr static auto GetNumABTensors()
{
    if constexpr(isTuple)
    {
        return Number<Tensors::Size()>{};
    }
    else
    {
        return Number<1>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetAGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::AsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetBGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::BsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename Id, typename Type>
constexpr static auto UnpackDataType()
{
    if constexpr(isTuple)
    {
        // unpack if tuple
        return tuple_element_t<Id{}, Type>{};
    }
    else
    {
        // if no, return Type
        return Type{};
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
