// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>
#include <vector>

#include "ck/ck.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

#include "batchnorm_common.hpp"

template <typename InOutDataType,
          typename AccDataType,
          ck::index_t Rank,
          ck::index_t NumBatchNormReduceDim,
          bool fastest_dim_is_reduced = false>
int bnorm_infer(
    bool time_kernel,
    const std::array<int, NumBatchNormReduceDim> reduceDims,
    const std::array<ck::index_t, Rank> xyLengths,
    const std::array<ck::index_t, Rank> xStrides,
    const std::array<ck::index_t, Rank> yStrides,
    const std::array<ck::index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
    const std::array<ck::index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarStrides,
    const void* p_x,
    const void* p_scale,
    const void* p_bias,
    double epsilon,
    const void* p_estimatedMean,
    const void* p_estimatedVariance,
    void* p_y)
{
    (void)bnScaleBiasMeanVarLengths;

    static_assert(NumBatchNormReduceDim < Rank,
                  "Invalid number of reduced dimensions for batchnorm!");

    using DeviceNormalizeInstance = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<InOutDataType, AccDataType, AccDataType, AccDataType, AccDataType>, // x, mean,
                                                                                      // variance,
                                                                                      // scale,
                                                                                      // bias,
        ck::Tuple<InOutDataType>,                                                     // y
        NormalizeInInfer,
        Rank,
        2,                           // MPerthread
        ck::Sequence<1, 1, 1, 1, 1>, // x, mean, variance, scale, bias
        ck::Sequence<1>>;            // scalarPerVector: y

    auto invariantDims = get_invariant_dims<Rank, NumBatchNormReduceDim>(reduceDims);
    std::array<ck::index_t, Rank> aligned_scaleBiasMeanVarStrides{0};

    int i = 0;
    for(auto dim : invariantDims)
    {
        assert(xyLengths[dim] == bnScaleBiasMeanVarLengths[i]);

        aligned_scaleBiasMeanVarStrides[dim] = bnScaleBiasMeanVarStrides[i];
        i++;
    };

    int32_t reduceLength = 1;

    for(auto dim : reduceDims)
        reduceLength *= xyLengths[dim];

    int32_t invariantLength = 1;

    for(auto dim : invariantDims)
        invariantLength *= xyLengths[dim];

    size_t total_length = static_cast<size_t>(invariantLength) * reduceLength;

    float avg_time        = 0.0f;
    std::size_t num_bytes = 0;

    auto dev_normalize = DeviceNormalizeInstance{};

    auto argument_ptr1 = dev_normalize.MakeArgumentPointer(
        xyLengths,
        {xStrides,
         aligned_scaleBiasMeanVarStrides,
         aligned_scaleBiasMeanVarStrides,
         aligned_scaleBiasMeanVarStrides,
         aligned_scaleBiasMeanVarStrides},
        {yStrides},
        {p_x, p_estimatedMean, p_estimatedVariance, p_scale, p_bias},
        {p_y},
        NormalizeInInfer{epsilon});

    if(!dev_normalize.IsSupportedArgument(argument_ptr1.get()))
    {
        std::cout << "The runtime parameters seems not supported by the Devic, exiting!"
                  << std::endl;

        return (-1);
    };

    auto invoker_ptr1 = dev_normalize.MakeInvokerPointer();

    avg_time += invoker_ptr1->Run(argument_ptr1.get(), StreamConfig{nullptr, time_kernel});

    num_bytes += (total_length * (1 * sizeof(InOutDataType) + 4 * sizeof(AccDataType)) +
                  total_length * sizeof(InOutDataType));

    if(time_kernel)
    {
        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    };

    return (0);
};
