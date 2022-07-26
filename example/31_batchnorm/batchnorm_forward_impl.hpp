// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>
#include <vector>

#include "ck/ck.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_multiple_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

#include "batchnorm_common.hpp"

template <typename InOutDataType,
          typename AccDataType,
          ck::index_t Rank,
          ck::index_t NumBatchNormReduceDim,
          bool fastest_dim_is_reduced = false>
int bnorm_fwd(bool time_kernel,
              bool updateMovingAverage,
              bool saveMeanAndInvVariance,
              const std::array<int, NumBatchNormReduceDim> reduceDims,
              const std::array<ck::index_t, Rank> xyLengths,
              const std::array<ck::index_t, Rank> xStrides,
              const std::array<ck::index_t, Rank> yStrides,
              const std::array<ck::index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
              const std::array<ck::index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarStrides,
              const void* p_x,
              const void* p_scale,
              const void* p_bias,
              void* p_y,
              double exponentialAverageFactor,
              void* p_runningMean,
              void* p_runningVariance,
              double epsilon,
              void* p_saveMean,
              void* p_saveInvVariance,
              void* p_tmp_mean,
              void* p_tmp_meansquare)
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
            ck::Tuple<AccDataType, AccDataType>,
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
            ck::Sequence<1, 1>>;

    using DeviceNormalizeInstance = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<InOutDataType, AccDataType, AccDataType, AccDataType, AccDataType>, // x, mean,
                                                                                      // meansquare,
                                                                                      // scale, bias
        ck::Tuple<InOutDataType>,                                                     // y
        NormalizeInForward,
        Rank,
        2,                           // MPerthread
        ck::Sequence<1, 1, 1, 1, 1>, // scalarPerVector: x, mean, meansquare, scale, bias
        ck::Sequence<1>>;            // scalarPerVector: y

    using DeviceInvVarianceInstance = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<AccDataType, AccDataType>, // mean, meansquare
        ck::Tuple<AccDataType>,              // invVariance
        InvVariance,
        NumScaleBiasMeanVarDim,
        2,                  // MPerthread
        ck::Sequence<1, 1>, // scalarPerVector: mean, meansquare
        ck::Sequence<1>>;   // scalarPerVector: invVariance

    using DeviceMovingAverageInstance = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<AccDataType, AccDataType, AccDataType, AccDataType>, // old moving mean, new mean,
                                                                       // old moving variance, new
                                                                       // meansquare
        ck::Tuple<AccDataType, AccDataType>, // updated moving mean, updated moving variance
        MovingAverage,
        NumScaleBiasMeanVarDim,
        4,                        // MPerthread
        ck::Sequence<1, 1, 1, 1>, // scalarPerVector: old moving mean, new mean, old moving
                                  // variance, new meansquare
        ck::Sequence<1, 1>>;      // scalarPerVector: updated moving mean, updated moving variance

    using DeviceMovingAverageAndInvVarianceInstance =
        ck::tensor_operation::device::DeviceElementwise<
            ck::Tuple<AccDataType, AccDataType, AccDataType, AccDataType>, // old moving mean, new
                                                                           // mean, old moving
                                                                           // variance, new
                                                                           // meansquare
            ck::Tuple<AccDataType, AccDataType, AccDataType>, // updated moving mean, updated moving
                                                              // variancem, invVariance
            MovingAverageAndInvVariance,
            NumScaleBiasMeanVarDim,
            4,                        // MPerthread
            ck::Sequence<1, 1, 1, 1>, // scalarPerVector: old moving mean, new mean, old moving
                                      // variance, new meansquare
            ck::Sequence<1, 1, 1>>; // scalarPerVector: updated moving mean, updated moving variance

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

    auto dev_mean_and_meansquare = DeviceMeanAndMeansquareInstance{};

    void* p_mean = saveMeanAndInvVariance ? p_saveMean : p_tmp_mean;

    const AccDataType alpha = ck::type_convert<AccDataType>(1.0f);
    const AccDataType beta  = ck::type_convert<AccDataType>(0.0f);

    auto argument_ptr1 = dev_mean_and_meansquare.MakeArgumentPointer(
        xyLengths,
        xStrides,
        bnScaleBiasMeanVarLengths,
        {bnScaleBiasMeanVarStrides, bnScaleBiasMeanVarStrides},
        reduceDims,
        {&alpha, &alpha},
        {&beta, &beta},
        p_x,
        {p_mean, p_tmp_meansquare},
        ck::make_tuple(InElementwiseOperation_Mean{}, InElementwiseOperation_Meansquare{}),
        ck::make_tuple(AccElementwiseOperation_Mean{reduceLength},
                       AccElementwiseOperation_Meansquare{reduceLength}));

    auto dev_normalize = DeviceNormalizeInstance{};

    auto argument_ptr2 =
        dev_normalize.MakeArgumentPointer(xyLengths,
                                          {xStrides,
                                           aligned_scaleBiasMeanVarStrides,
                                           aligned_scaleBiasMeanVarStrides,
                                           aligned_scaleBiasMeanVarStrides,
                                           aligned_scaleBiasMeanVarStrides},
                                          {yStrides},
                                          {p_x, p_mean, p_tmp_meansquare, p_scale, p_bias},
                                          {p_y},
                                          NormalizeInForward{epsilon});

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

    if(saveMeanAndInvVariance && updateMovingAverage)
    {
        auto dev_moving_average_inv_variance = DeviceMovingAverageAndInvVarianceInstance{};

        auto argument_ptr3 = dev_moving_average_inv_variance.MakeArgumentPointer(
            bnScaleBiasMeanVarLengths,
            {bnScaleBiasMeanVarStrides,
             bnScaleBiasMeanVarStrides,
             bnScaleBiasMeanVarStrides,
             bnScaleBiasMeanVarStrides},
            {bnScaleBiasMeanVarStrides, bnScaleBiasMeanVarStrides, bnScaleBiasMeanVarStrides},
            {p_mean, p_runningMean, p_tmp_meansquare, p_runningVariance},
            {p_runningMean, p_runningVariance, p_saveInvVariance},
            MovingAverageAndInvVariance{epsilon, exponentialAverageFactor});

        if(!dev_moving_average_inv_variance.IsSupportedArgument(argument_ptr3.get()))
        {
            std::cout << "Runtime parameters not supported by the Device, exiting!" << std::endl;

            return (-1);
        };

        auto invoker_ptr3 = dev_moving_average_inv_variance.MakeInvokerPointer();

        avg_time += invoker_ptr3->Run(argument_ptr3.get(), StreamConfig{nullptr, time_kernel});

        num_bytes += invariantLength * (4 + 3) * sizeof(AccDataType) * 2; // No.5
    }
    else if(saveMeanAndInvVariance)
    {
        auto dev_inv_variance = DeviceInvVarianceInstance{};
        auto argument_ptr3    = dev_inv_variance.MakeArgumentPointer(
            bnScaleBiasMeanVarLengths,
            {bnScaleBiasMeanVarStrides, bnScaleBiasMeanVarStrides},
            {bnScaleBiasMeanVarStrides},
            {p_mean, p_tmp_meansquare},
            {p_saveInvVariance},
            InvVariance{epsilon});

        if(!dev_inv_variance.IsSupportedArgument(argument_ptr3.get()))
        {
            std::cout << "Runtime parameters not supported by the Device, exiting!" << std::endl;

            return (-1);
        };

        auto invoker_ptr3 = dev_inv_variance.MakeInvokerPointer();

        avg_time += invoker_ptr3->Run(argument_ptr3.get(), StreamConfig{nullptr, time_kernel});

        num_bytes += invariantLength * (2 + 1) * sizeof(AccDataType);
    }
    else if(updateMovingAverage)
    {
        auto dev_moving_average = DeviceMovingAverageInstance{};

        auto argument_ptr3 = dev_moving_average.MakeArgumentPointer(
            bnScaleBiasMeanVarLengths,
            {bnScaleBiasMeanVarStrides,
             bnScaleBiasMeanVarStrides,
             bnScaleBiasMeanVarStrides,
             bnScaleBiasMeanVarStrides},
            {bnScaleBiasMeanVarStrides, bnScaleBiasMeanVarStrides},
            {p_mean, p_runningMean, p_tmp_meansquare, p_runningVariance},
            {p_runningMean, p_runningVariance},
            MovingAverage{exponentialAverageFactor});

        if(!dev_moving_average.IsSupportedArgument(argument_ptr3.get()))
        {
            std::cout << "Runtime parameters not supported by the Device, exiting!" << std::endl;

            return (-1);
        };

        auto invoker_ptr3 = dev_moving_average.MakeInvokerPointer();

        avg_time += invoker_ptr3->Run(argument_ptr3.get(), StreamConfig{nullptr, time_kernel});

        num_bytes += invariantLength * (4 + 2) * sizeof(AccDataType) * 2; // No.5
    };

    if(time_kernel)
    {
        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    };

    return (0);
};
