// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>
#include <vector>

#include "ck/ck.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_multiple_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/device_binary_elementwise.hpp"
#include "ck/tensor_operation/gpu/device/device_5ary_elementwise.hpp"

#include "batchnorm_common.hpp"

namespace batchnorm {

template <typename InOutDataType,
          typename AccDataType,
          ck::index_t Rank,
          ck::index_t NumBatchNormReduceDim,
          bool fastest_dim_is_reduced = false>
int bnorm_fwd(bool time_kernel,
              bool saveMeanAndInvVariance,
              bool updateMovingAverage,
              const std::vector<int> reduceDims,
              const std::vector<ck::index_t> xyLengths,
              const std::vector<ck::index_t> xStrides,
              const std::vector<ck::index_t> yStrides,
              const std::vector<ck::index_t> bnScaleBiasMeanVarLengths,
              const std::vector<ck::index_t> bnScaleBiasMeanVarStrides,
              const void* p_x,
              const void* p_scale,
              const void* p_bias,
              void* p_y,
              double exponentialAverageFactor,
              void* p_runningMean,
              void* p_runningVariance,
              double epsilon,
              void* p_savedMean,
              void* p_savedInvVariance,
              void* p_workspace)
{
    static_assert(NumBatchNormReduceDim < Rank,
                  "Invalid number of reduced dimensions for batchnorm!");

    constexpr ck::index_t NumScaleBiasMeanVarDim = Rank - NumBatchNormReduceDim;

    using InElementwiseOperation_Mean  = ck::tensor_operation::element_wise::PassThrough;
    using AccElementwiseOperation_Mean = ck::tensor_operation::element_wise::UnaryDivide;

    using InElementwiseOperation_Meansquare  = ck::tensor_operation::element_wise::UnarySquare;
    using AccElementwiseOperation_Meansquare = ck::tensor_operation::element_wise::UnaryDivide;

    using DeviceMeanAndMeansquareInstance =
        ck::tensor_operation::device::DeviceMultipleReduceMultiBlock<
            2,
            InOutDataType,
            AccDataType,
            ck::Tuple<AccDataType*, AccDataType*>,
            Rank,
            NumBatchNormReduceDim,
            ck::reduce::Add,
            ck::Tuple<InElementwiseOperation_Mean, InElementwiseOperation_Meansquare>,
            ck::Tuple<AccElementwiseOperation_Mean, AccElementwiseOperation_Meansquare>,
            ck::InMemoryDataOperationEnum::Set,
            false, // PropagateNan
            256,
            16,
            16,
            1,
            1,
            fastest_dim_is_reduced ? 1 : 0,
            1,
            1>;

    using NormalizeOp = ck::tensor_operation::element_wise::Normalize;

    using DeviceNormalizeInstance =
        ck::tensor_operation::device::Device5AryElementwise<InOutDataType, // x
                                                            AccDataType,   // mean
                                                            AccDataType,   // meansquare
                                                            AccDataType,   // Scale
                                                            AccDataType,   // Bias
                                                            InOutDataType, // y
                                                            AccDataType,   //
                                                            NormalizeOp,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: x
                                                            1,  // scalarPerVector: mean
                                                            1,  // scalarPerVector: meansquare
                                                            1,  // scalarPerVector: Scale
                                                            1,  // scalarPerVector: Bias
                                                            1>; // scalarPerVector: y

    using InvVariance = ck::tensor_operation::element_wise::InvVariance;

    using DeviceInvVarianceInstance =
        ck::tensor_operation::device::DeviceBinaryElementwise<AccDataType, // mean
                                                              AccDataType, // meansquare
                                                              AccDataType, // InvVariance
                                                              AccDataType,
                                                              InvVariance,
                                                              NumScaleBiasMeanVarDim,
                                                              4,
                                                              1,  // scalerPerVector: mean
                                                              1,  // scalerPerVector: meansquare
                                                              1>; // scalerPerVector: InvVariance

    using DeviceVarianceInstance =
        ck::tensor_operation::device::DeviceBinaryElementwise<AccDataType, // mean
                                                              AccDataType, // meansquare
                                                              AccDataType, // Variance
                                                              AccDataType,
                                                              detail::Variance,
                                                              NumScaleBiasMeanVarDim,
                                                              4,
                                                              1,  // scalerPerVector: mean
                                                              1,  // scalerPerVector: meansquare
                                                              1>; // scalerPerVector: variance

    using DeviceMovingAverageInstance = ck::tensor_operation::device::DeviceBinaryElementwise<
        AccDataType, // old moving mean/variance
        AccDataType, // new mean/variance
        AccDataType, // updated moving mean/variance
        AccDataType,
        detail::MovingAverage,
        NumScaleBiasMeanVarDim,
        4,
        1,  // scalerPerVector: old moving mean/variance
        1,  // scalerPerVector: new mean/variance
        1>; // scalerPerVector: updated moving mean/variance

    auto invariantDims = get_invariant_dims<Rank, NumBatchNormReduceDim>(reduceDims);
    std::vector<ck::index_t> aligned_scaleBiasMeanVarStrides(Rank, 0);

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

    void* p_meansquare =
        (saveMeanAndInvVariance && updateMovingAverage) ? p_workspace : p_savedInvVariance;

    float avg_time        = 0.0f;
    std::size_t num_bytes = 0;

    auto dev_mean_and_meansquare = DeviceMeanAndMeansquareInstance{};

    auto argument_ptr1 = dev_mean_and_meansquare.MakeArgumentPointer(
        xyLengths,
        xStrides,
        bnScaleBiasMeanVarLengths,
        bnScaleBiasMeanVarStrides,
        reduceDims,
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        p_x,
        {p_savedMean, p_meansquare},
        ck::make_tuple(InElementwiseOperation_Mean{}, InElementwiseOperation_Meansquare{}),
        ck::make_tuple(AccElementwiseOperation_Mean{reduceLength},
                       AccElementwiseOperation_Meansquare{reduceLength}));

    auto dev_normalize = DeviceNormalizeInstance{};

    auto argument_ptr2 = dev_normalize.MakeArgumentPointer(p_x,
                                                           p_savedMean,  // mean
                                                           p_meansquare, // meansquare
                                                           p_scale,      // scale
                                                           p_bias,       // bias
                                                           p_y,
                                                           xyLengths,
                                                           xStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           yStrides,
                                                           NormalizeOp{epsilon});

    if(!dev_mean_and_meansquare.IsSupportedArgument(argument_ptr1.get()) ||
       !dev_normalize.IsSupportedArgument(argument_ptr2.get()))
    {
        std::cout << "The runtime parameters seems not supported by the Devic, exiting!"
                  << std::endl;

        return (-1);
    };

    auto invoker_ptr1 = dev_mean_and_meansquare.MakeInvokerPointer();
    auto invoker_ptr2 = dev_normalize.MakeInvokerPointer();

    avg_time += invoker_ptr1->Run(argument_ptr1.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr2->Run(argument_ptr2.get(), StreamConfig{nullptr, time_kernel});

    num_bytes +=
        (total_length * sizeof(InOutDataType) + invariantLength * 2 * sizeof(AccDataType)) + // No.1
        (total_length * (1 * sizeof(InOutDataType) + 4 * sizeof(AccDataType)) +
         total_length * sizeof(InOutDataType)); // No.2

    if(saveMeanAndInvVariance)
    {
        auto dev_inv_variance = DeviceInvVarianceInstance{};
        auto argument_ptr3    = dev_inv_variance.MakeArgumentPointer(p_savedMean,
                                                                  p_meansquare,
                                                                  p_savedInvVariance,
                                                                  bnScaleBiasMeanVarLengths,
                                                                  bnScaleBiasMeanVarStrides,
                                                                  bnScaleBiasMeanVarStrides,
                                                                  bnScaleBiasMeanVarStrides,
                                                                  InvVariance{epsilon});

        if(!dev_inv_variance.IsSupportedArgument(argument_ptr3.get()))
        {
            std::cout << "Runtime parameters not supported by the Device, exiting!" << std::endl;

            return (-1);
        };

        auto invoker_ptr3 = dev_inv_variance.MakeInvokerPointer();

        avg_time += invoker_ptr3->Run(argument_ptr3.get(), StreamConfig{nullptr, time_kernel});

        num_bytes += invariantLength * (2 + 1) * sizeof(AccDataType);
    };

    if(updateMovingAverage)
    {
        auto dev_variance       = DeviceVarianceInstance{};
        auto dev_moving_average = DeviceMovingAverageInstance{};

        auto argument_ptr4 = dev_variance.MakeArgumentPointer(p_savedMean,
                                                              p_meansquare,
                                                              p_meansquare, // variance
                                                              bnScaleBiasMeanVarLengths,
                                                              bnScaleBiasMeanVarStrides,
                                                              bnScaleBiasMeanVarStrides,
                                                              bnScaleBiasMeanVarStrides,
                                                              detail::Variance{});

        auto argument_ptr5 =
            dev_moving_average.MakeArgumentPointer(p_savedMean,
                                                   p_runningMean,
                                                   p_runningMean,
                                                   bnScaleBiasMeanVarLengths,
                                                   bnScaleBiasMeanVarStrides,
                                                   bnScaleBiasMeanVarStrides,
                                                   bnScaleBiasMeanVarStrides,
                                                   detail::MovingAverage{exponentialAverageFactor});

        auto argument_ptr6 =
            dev_moving_average.MakeArgumentPointer(p_meansquare, // variance
                                                   p_runningVariance,
                                                   p_runningVariance,
                                                   bnScaleBiasMeanVarLengths,
                                                   bnScaleBiasMeanVarStrides,
                                                   bnScaleBiasMeanVarStrides,
                                                   bnScaleBiasMeanVarStrides,
                                                   detail::MovingAverage{exponentialAverageFactor});

        if(!dev_variance.IsSupportedArgument(argument_ptr4.get()) ||
           !dev_moving_average.IsSupportedArgument(argument_ptr5.get()) ||
           !dev_moving_average.IsSupportedArgument(argument_ptr6.get()))
        {
            std::cout << "Runtime parameters not supported by the Device, exiting!" << std::endl;

            return (-1);
        };

        auto invoker_ptr4  = dev_variance.MakeInvokerPointer();
        auto invoker_ptr56 = dev_moving_average.MakeInvokerPointer();

        avg_time += invoker_ptr4->Run(argument_ptr4.get(), StreamConfig{nullptr, time_kernel});
        avg_time += invoker_ptr56->Run(argument_ptr5.get(), StreamConfig{nullptr, time_kernel});
        avg_time += invoker_ptr56->Run(argument_ptr6.get(), StreamConfig{nullptr, time_kernel});

        num_bytes += invariantLength * (2 + 1) * sizeof(AccDataType) + // No.4
                     invariantLength * (2 + 1) * sizeof(AccDataType) + // No.5
                     invariantLength * (2 + 1) * sizeof(AccDataType);  // No.6
    };

    if(time_kernel)
    {
        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    };

    return (0);
};

}; // end of namespace batchnorm
