// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct ComputePtrOffsetOfStridedBatch
{
    static constexpr bool isMultiAB = NumATensor > 1 || NumBTensor > 1;

    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                   index_t BatchStrideB,
                                   Array<ck::index_t, NumDTensor> BatchStrideDs,
                                   index_t BatchStrideE)
        : BatchStrideA_(),
          BatchStrideB_(),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
        if constexpr(!isMultiAB)
        {
            BatchStrideA_ = BatchStrideA;
            BatchStrideB_ = BatchStrideB;
        }
        else
        {
            static_assert("Invalid constructor for multiple A or B");
        }
    }

    ComputePtrOffsetOfStridedBatch(Array<ck::index_t, NumATensor> BatchStrideAs,
                                   Array<ck::index_t, NumBTensor> BatchStrideBs,
                                   Array<ck::index_t, NumDTensor> BatchStrideDs,
                                   index_t BatchStrideE)
        : BatchStrideA_(),
          BatchStrideB_(),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
        if constexpr(isMultiAB)
        {
            BatchStrideA_ = BatchStrideAs;
            BatchStrideB_ = BatchStrideBs;
        }
        else
        {
            static_assert("Invalid constructor for single A and B");
        }
    }

    __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
    {
        if constexpr(!isMultiAB)
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }
        else
        {
            static_assert("Invalid function for multiple A or B");
            return 0;
        }
    }

    __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
    {
        if constexpr(!isMultiAB)
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }
        else
        {
            static_assert("Invalid function for multiple A or B");
            return 0;
        }
    }

    __host__ __device__ constexpr auto GetAsPtrOffset(index_t g_idx) const
    {
        if constexpr(isMultiAB)
        {
            Array<long_index_t, NumATensor> as_offset;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideA_[i]);
            });
            return as_offset;
        }
        else
        {
            static_assert("Invalid function for single A and B");
            return BatchStrideA_;
        }
    }

    __host__ __device__ constexpr auto GetBsPtrOffset(index_t g_idx) const
    {
        if constexpr(isMultiAB)
        {
            Array<long_index_t, NumBTensor> bs_offset;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideB_[i]);
            });
            return bs_offset;
        }
        else
        {
            static_assert("Invalid function for single A and B");
            return BatchStrideB_;
        }
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

    // If multiAB use Array
    using BatchStrideAType =
        std::conditional_t<isMultiAB, Array<ck::index_t, NumATensor>, ck::index_t>;
    using BatchStrideBType =
        std::conditional_t<isMultiAB, Array<ck::index_t, NumBTensor>, ck::index_t>;

    BatchStrideAType BatchStrideA_;
    BatchStrideBType BatchStrideB_;
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
