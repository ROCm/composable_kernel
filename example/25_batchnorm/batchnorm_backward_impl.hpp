#pragma once

#include <cassert>
#include <vector>

#include "ck/ck.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_multiple_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/device_binary_elementwise.hpp"
#include "ck/tensor_operation/gpu/device/device_5ary_elementwise.hpp"
#include "ck/tensor_operation/gpu/device/device_4ary_elementwise.hpp"

#include "batchnorm_common.hpp"

namespace batchnorm {

template <typename InOutDataType,
          typename AccDataType,
          ck::index_t Rank,
          ck::index_t NumBatchNormReduceDim,
          bool fastest_dim_is_reduced = false>
int bnorm_bwd_use_saved_mean_inv_variance(bool time_kernel,
                                          const std::vector<int> reduceDims,
                                          const std::vector<ck::index_t> xyLengths,
                                          const std::vector<ck::index_t> xStrides,
                                          const std::vector<ck::index_t> dyStrides,
                                          const std::vector<ck::index_t> dxStrides,
                                          const std::vector<ck::index_t> bnScaleBiasLengths,
                                          const std::vector<ck::index_t> bnScaleBiasStrides,
                                          const void* p_x,
                                          const void* p_dy,
                                          const void* p_scale,
                                          const void* p_savedMean,
                                          const void* p_savedInvVariance,
                                          void* p_dx,
                                          void* p_scaleDiff,
                                          void* p_biasDiff)
{
    static_assert(NumBatchNormReduceDim < Rank,
                  "Invalid number of reduced dimensions for batchnorm!");

    using DeviceNormalizeAndMultiplyInstance1 =
        ck::tensor_operation::device::Device4AryElementwise<InOutDataType, // x
                                                            AccDataType,   // mean
                                                            AccDataType,   // invVariance
                                                            InOutDataType, // val4 (for dy)
                                                            InOutDataType, // z
                                                            AccDataType,   //
                                                            detail::NormalizeAndMultiply,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: x
                                                            1,  // scalarPerVector: mean
                                                            1,  // scalarPerVector: invVariance
                                                            1,  // scalarPerVector: val4
                                                            1>; // scalarPerVector: z

    using DeviceNormalizeAndMultiplyInstance2 =
        ck::tensor_operation::device::Device4AryElementwise<InOutDataType, // x
                                                            AccDataType,   // mean
                                                            AccDataType,   // invVariance
                                                            AccDataType,   // val4 (for scaleDiff)
                                                            InOutDataType, // y
                                                            AccDataType,   //
                                                            detail::NormalizeAndMultiply,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: x
                                                            1,  // scalarPerVector: mean
                                                            1,  // scalarPerVector: invVariance
                                                            1,  // scalarPerVector: val4
                                                            1>; // scalarPerVector: z

    using DeviceFinalDiffXInstance =
        ck::tensor_operation::device::Device5AryElementwise<InOutDataType, // dy
                                                            AccDataType,   // invVariance
                                                            AccDataType,   // scale
                                                            AccDataType,   // biasDiff
                                                            InOutDataType, // val5
                                                            InOutDataType, // z
                                                            AccDataType,   //
                                                            detail::FinalDiffX,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: invVariance
                                                            1,  // scalarPerVector: scale
                                                            1,  // scalarPerVector: dy
                                                            1,  // scalarPerVector: biasDiff
                                                            1,  // scalarPerVector: val5
                                                            1>; // scalarPerVector: 5

    using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

    using DeviceReduceInstance =
        ck::tensor_operation::device::DeviceReduceMultiBlock<InOutDataType,
                                                             AccDataType,
                                                             AccDataType,
                                                             Rank,
                                                             NumBatchNormReduceDim,
                                                             ck::reduce::Add,
                                                             PassThroughOp,
                                                             PassThroughOp,
                                                             ck::InMemoryDataOperationEnum::Set,
                                                             false, // PropagateNan
                                                             false, // OutputIndex
                                                             false, // HaveIndexInputIfOutputIndex
                                                             256,
                                                             4,
                                                             64,
                                                             1,
                                                             1,
                                                             fastest_dim_is_reduced ? 1 : 0,
                                                             1,
                                                             1>;

    auto invariantDims = get_invariant_dims<Rank, NumBatchNormReduceDim>(reduceDims);
    std::vector<ck::index_t> aligned_scaleBiasStrides(Rank, 0);

    int i = 0;
    for(auto dim : invariantDims)
    {
        assert(xyLengths[dim] == bnScaleBiasLengths[i]);

        aligned_scaleBiasStrides[dim] = bnScaleBiasStrides[i];
        i++;
    };

    int32_t reduceLength = 1;

    for(auto dim : reduceDims)
        reduceLength *= xyLengths[dim];

    int32_t invariantLength = 1;

    for(auto dim : invariantDims)
        invariantLength *= xyLengths[dim];

    size_t total_length = static_cast<size_t>(invariantLength) * reduceLength;

    auto normalize_and_multiply1 = DeviceNormalizeAndMultiplyInstance1{};
    auto normalize_and_multiply2 = DeviceNormalizeAndMultiplyInstance2{};
    auto reduce                  = DeviceReduceInstance{};
    auto final_diff_x            = DeviceFinalDiffXInstance{};

    auto argument_ptr1 =
        normalize_and_multiply1.MakeArgumentPointer(p_x,
                                                    p_savedMean,        // mean
                                                    p_savedInvVariance, // invVariance
                                                    p_dy,               // val4
                                                    p_dx,               // z
                                                    xyLengths,
                                                    xStrides,
                                                    aligned_scaleBiasStrides, // mean
                                                    aligned_scaleBiasStrides, // invVariance
                                                    dyStrides,                // val4
                                                    dxStrides,                // z
                                                    detail::NormalizeAndMultiply{});

    auto argument_ptr2 = reduce.MakeArgumentPointer(xyLengths,
                                                    dyStrides,
                                                    bnScaleBiasLengths,
                                                    bnScaleBiasStrides,
                                                    reduceDims,
                                                    1.0f,
                                                    0.0f,
                                                    p_dy,
                                                    nullptr,
                                                    p_biasDiff, // biasDiff (output)
                                                    nullptr,
                                                    PassThroughOp{},
                                                    PassThroughOp{});

    auto argument_ptr3 = reduce.MakeArgumentPointer(xyLengths,
                                                    dxStrides,
                                                    bnScaleBiasLengths,
                                                    bnScaleBiasStrides,
                                                    reduceDims,
                                                    1.0f,
                                                    0.0f,
                                                    p_dx,
                                                    nullptr,
                                                    p_scaleDiff, // scaleDiff (output)
                                                    nullptr,
                                                    PassThroughOp{},
                                                    PassThroughOp{});

    auto argument_ptr4 =
        normalize_and_multiply2.MakeArgumentPointer(p_x,                // last z
                                                    p_savedMean,        // mean
                                                    p_savedInvVariance, // invVariance
                                                    p_scaleDiff,        // val4
                                                    p_dx,               // z
                                                    xyLengths,
                                                    xStrides,
                                                    aligned_scaleBiasStrides, // mean
                                                    aligned_scaleBiasStrides, // invVariance
                                                    aligned_scaleBiasStrides, // val4
                                                    dxStrides,                // z
                                                    detail::NormalizeAndMultiply{});

    auto argument_ptr5 = final_diff_x.MakeArgumentPointer(p_dy,               // dy
                                                          p_savedInvVariance, // invVariance
                                                          p_scale,            // gamma
                                                          p_biasDiff,         // diffBeta
                                                          p_dx,               // val5
                                                          p_dx,               // z, dx (output)
                                                          xyLengths,
                                                          dyStrides,                // dy
                                                          aligned_scaleBiasStrides, // invVariance
                                                          aligned_scaleBiasStrides, // gamma
                                                          aligned_scaleBiasStrides, // diffBeta
                                                          dxStrides,                // val5
                                                          dxStrides,                // z
                                                          detail::FinalDiffX{reduceLength});

    if(!normalize_and_multiply1.IsSupportedArgument(argument_ptr1.get()) ||
       !reduce.IsSupportedArgument(argument_ptr2.get()) ||
       !reduce.IsSupportedArgument(argument_ptr3.get()) ||
       !normalize_and_multiply2.IsSupportedArgument(argument_ptr4.get()) ||
       !final_diff_x.IsSupportedArgument(argument_ptr5.get()))
    {
        std::cout << "The runtime parameters seems not supported by the Devic instances, exiting!"
                  << std::endl;

        return (-1);
    };

    auto invoker_ptr1 = normalize_and_multiply1.MakeInvokerPointer();
    auto invoker_ptr2 = reduce.MakeInvokerPointer();
    auto invoker_ptr3 = reduce.MakeInvokerPointer();
    auto invoker_ptr4 = normalize_and_multiply2.MakeInvokerPointer();
    auto invoker_ptr5 = final_diff_x.MakeInvokerPointer();

    float avg_time = 0.0f;

    avg_time += invoker_ptr1->Run(argument_ptr1.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr2->Run(argument_ptr2.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr3->Run(argument_ptr3.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr4->Run(argument_ptr4.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr5->Run(argument_ptr5.get(), StreamConfig{nullptr, time_kernel});

    std::size_t num_bytes =
        (total_length * (2 * sizeof(InOutDataType) + 2 * sizeof(AccDataType)) +
         total_length * sizeof(InOutDataType)) +                                         // No.1
        (total_length * sizeof(InOutDataType) + invariantLength * sizeof(AccDataType)) + // No.2
        (total_length * sizeof(InOutDataType) + invariantLength * sizeof(AccDataType)) + // No.3
        (total_length * (2 * sizeof(InOutDataType) + 2 * sizeof(AccDataType)) +
         total_length * sizeof(InOutDataType)) + // No.4
        (total_length * (2 * sizeof(InOutDataType) + 3 * sizeof(AccDataType)) +
         total_length * sizeof(InOutDataType)); // No.5

    if(time_kernel)
    {
        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    };

    return (0);
};

template <typename InOutDataType,
          typename AccDataType,
          ck::index_t Rank,
          ck::index_t NumBatchNormReduceDim,
          bool fastest_dim_is_reduced = false>
int bnorm_bwd_without_saved_mean_inv_variance(bool time_kernel,
                                              const std::vector<int> reduceDims,
                                              const std::vector<ck::index_t> xyLengths,
                                              const std::vector<ck::index_t> xStrides,
                                              const std::vector<ck::index_t> dyStrides,
                                              const std::vector<ck::index_t> dxStrides,
                                              const std::vector<ck::index_t> bnScaleBiasDiffLengths,
                                              const std::vector<ck::index_t> bnScaleBiasDiffStrides,
                                              const void* p_x,
                                              const void* p_dy,
                                              const void* p_scale,
                                              double epsilon,
                                              void* p_dx,
                                              void* p_scaleDiff,
                                              void* p_biasDiff,
                                              void* p_workspace)
{
    static_assert(NumBatchNormReduceDim < Rank,
                  "Invalid number of reduced dimensions for batchnorm!");

    constexpr ck::index_t NumScaleBiasDiffDim = Rank - NumBatchNormReduceDim;

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

    using InvVariance = ck::tensor_operation::element_wise::InvVariance;

    using DeviceInvVarianceInstance =
        ck::tensor_operation::device::DeviceBinaryElementwise<AccDataType, // mean
                                                              AccDataType, // meansquare
                                                              AccDataType, // InvVariance
                                                              AccDataType,
                                                              InvVariance,
                                                              NumScaleBiasDiffDim,
                                                              4,
                                                              1,  // scalerPerVector: mean
                                                              1,  // scalerPerVector: meansquare
                                                              1>; // scalerPerVector: InvVariance

    using DeviceNormalizeAndMultiplyInstance1 =
        ck::tensor_operation::device::Device4AryElementwise<InOutDataType, // x
                                                            AccDataType,   // mean
                                                            AccDataType,   // invVariance
                                                            InOutDataType, // val4 (for dy)
                                                            InOutDataType, // z
                                                            AccDataType,   //
                                                            detail::NormalizeAndMultiply,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: x0
                                                            1,  // scalarPerVector: mean
                                                            1,  // scalarPerVector: invVariance
                                                            1,  // scalarPerVector: x1
                                                            1>; // scalarPerVector: y

    using DeviceNormalizeAndMultiplyInstance2 =
        ck::tensor_operation::device::Device4AryElementwise<InOutDataType, // x
                                                            AccDataType,   // mean
                                                            AccDataType,   // invVariance
                                                            AccDataType,   // val4 (for scaleDiff)
                                                            InOutDataType, // z
                                                            AccDataType,   //
                                                            detail::NormalizeAndMultiply,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: x0
                                                            1,  // scalarPerVector: mean
                                                            1,  // scalarPerVector: invVariance
                                                            1,  // scalarPerVector: x1
                                                            1>; // scalarPerVector: y

    using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

    using DeviceReduceInstance =
        ck::tensor_operation::device::DeviceReduceMultiBlock<InOutDataType,
                                                             AccDataType,
                                                             AccDataType,
                                                             Rank,
                                                             NumBatchNormReduceDim,
                                                             ck::reduce::Add,
                                                             PassThroughOp,
                                                             PassThroughOp,
                                                             ck::InMemoryDataOperationEnum::Set,
                                                             false, // PropagateNan
                                                             false, // OutputIndex
                                                             false, // HaveIndexInputIfOutputIndex
                                                             256,
                                                             4,
                                                             64,
                                                             1,
                                                             1,
                                                             fastest_dim_is_reduced ? 1 : 0,
                                                             1,
                                                             1>;

    using DeviceFinalDiffXInstance =
        ck::tensor_operation::device::Device5AryElementwise<InOutDataType, // dy
                                                            AccDataType,   // invVariance
                                                            AccDataType,   // scale
                                                            AccDataType,   // biasDiff
                                                            InOutDataType, // val5
                                                            InOutDataType, // z
                                                            AccDataType,   //
                                                            detail::FinalDiffX,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: invVariance
                                                            1,  // scalarPerVector: scale
                                                            1,  // scalarPerVector: dy
                                                            1,  // scalarPerVector: biasDiff
                                                            1,  // scalarPerVector: val5
                                                            1>; // scalarPerVector: z

    auto invariantDims = get_invariant_dims<Rank, NumBatchNormReduceDim>(reduceDims);
    std::vector<ck::index_t> aligned_scaleBiasDiffStrides(Rank, 0);

    int i = 0;
    for(auto dim : invariantDims)
    {
        aligned_scaleBiasDiffStrides[dim] = bnScaleBiasDiffStrides[i];
        i++;
    };

    int32_t reduceLength = 1;

    for(auto dim : reduceDims)
        reduceLength *= xyLengths[dim];

    int32_t invariantLength = 1;

    for(auto dim : invariantDims)
        invariantLength *= xyLengths[dim];

    size_t total_length = static_cast<size_t>(invariantLength) * reduceLength;

    auto mean_and_meansquare     = DeviceMeanAndMeansquareInstance{};
    auto inv_variance            = DeviceInvVarianceInstance{};
    auto normalize_and_multiply1 = DeviceNormalizeAndMultiplyInstance1{};
    auto normalize_and_multiply2 = DeviceNormalizeAndMultiplyInstance2{};
    auto reduce                  = DeviceReduceInstance{};
    auto final_diff_x            = DeviceFinalDiffXInstance{};

    auto argument_ptr1 = mean_and_meansquare.MakeArgumentPointer(
        xyLengths,
        xStrides,
        bnScaleBiasDiffLengths,
        bnScaleBiasDiffStrides,
        reduceDims,
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        p_x,
        {p_biasDiff, p_workspace}, // mean, meansquare
        ck::make_tuple(InElementwiseOperation_Mean{}, InElementwiseOperation_Meansquare{}),
        ck::make_tuple(AccElementwiseOperation_Mean{reduceLength},
                       AccElementwiseOperation_Meansquare{reduceLength}));

    auto argument_ptr2 = inv_variance.MakeArgumentPointer(p_biasDiff,  // mean
                                                          p_workspace, // meansquare
                                                          p_workspace, // invVariance
                                                          bnScaleBiasDiffLengths,
                                                          bnScaleBiasDiffStrides,
                                                          bnScaleBiasDiffStrides,
                                                          bnScaleBiasDiffStrides,
                                                          InvVariance{epsilon});

    auto argument_ptr3 =
        normalize_and_multiply1.MakeArgumentPointer(p_x,
                                                    p_biasDiff,  // mean
                                                    p_workspace, // invVariance
                                                    p_dy,        // val4 (for dy)
                                                    p_dx,        // z
                                                    xyLengths,
                                                    xStrides,
                                                    aligned_scaleBiasDiffStrides, // mean
                                                    aligned_scaleBiasDiffStrides, // invVariance
                                                    dyStrides,                    // val4
                                                    dxStrides,                    // z
                                                    detail::NormalizeAndMultiply{});

    auto argument_ptr4 = reduce.MakeArgumentPointer(xyLengths,
                                                    dxStrides,
                                                    bnScaleBiasDiffLengths,
                                                    bnScaleBiasDiffStrides,
                                                    reduceDims,
                                                    1.0f,
                                                    0.0f,
                                                    p_dx, // dy * norm_x
                                                    nullptr,
                                                    p_scaleDiff, // scaleDiff (output)
                                                    nullptr,
                                                    PassThroughOp{},
                                                    PassThroughOp{});

    auto argument_ptr5 =
        normalize_and_multiply2.MakeArgumentPointer(p_x,
                                                    p_biasDiff,  // mean
                                                    p_workspace, // invVariance
                                                    p_scaleDiff, // val4 (for scaleDiff)
                                                    p_dx,        // z
                                                    xyLengths,
                                                    xStrides,
                                                    aligned_scaleBiasDiffStrides, // mean
                                                    aligned_scaleBiasDiffStrides, // invVariance
                                                    aligned_scaleBiasDiffStrides, // val4
                                                    dxStrides,                    // z
                                                    detail::NormalizeAndMultiply{});

    auto argument_ptr6 = reduce.MakeArgumentPointer(xyLengths,
                                                    dyStrides,
                                                    bnScaleBiasDiffLengths,
                                                    bnScaleBiasDiffStrides,
                                                    reduceDims,
                                                    1.0f,
                                                    0.0f,
                                                    p_dy,
                                                    nullptr,
                                                    p_biasDiff, // biasDiff (output)
                                                    nullptr,
                                                    PassThroughOp{},
                                                    PassThroughOp{});

    auto argument_ptr7 =
        final_diff_x.MakeArgumentPointer(p_dy,        // dy
                                         p_workspace, // invVariance
                                         p_scale,     // scale
                                         p_biasDiff,  // biasDiff
                                         p_dx,        // val5
                                         p_dx,        // z, dx (output)
                                         xyLengths,
                                         dyStrides,                    // dy
                                         aligned_scaleBiasDiffStrides, // invVariance
                                         aligned_scaleBiasDiffStrides, // scale
                                         aligned_scaleBiasDiffStrides, // biasDiff
                                         dxStrides,                    // val5
                                         dxStrides,                    // z
                                         detail::FinalDiffX{reduceLength});

    if(!mean_and_meansquare.IsSupportedArgument(argument_ptr1.get()) ||
       !inv_variance.IsSupportedArgument(argument_ptr2.get()) ||
       !normalize_and_multiply1.IsSupportedArgument(argument_ptr3.get()) ||
       !reduce.IsSupportedArgument(argument_ptr4.get()) ||
       !normalize_and_multiply2.IsSupportedArgument(argument_ptr5.get()) ||
       !reduce.IsSupportedArgument(argument_ptr6.get()) ||
       !final_diff_x.IsSupportedArgument(argument_ptr7.get()))
    {
        std::cout << "The runtime parameters seems not supported by the Device instances, exiting!"
                  << std::endl;

        return (-1);
    };

    auto invoker_ptr1  = mean_and_meansquare.MakeInvokerPointer();
    auto invoker_ptr2  = inv_variance.MakeInvokerPointer();
    auto invoker_ptr3  = normalize_and_multiply1.MakeInvokerPointer();
    auto invoker_ptr5  = normalize_and_multiply2.MakeInvokerPointer();
    auto invoker_ptr46 = reduce.MakeInvokerPointer();
    auto invoker_ptr7  = final_diff_x.MakeInvokerPointer();

    float avg_time = 0.0f;

    avg_time += invoker_ptr1->Run(argument_ptr1.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr2->Run(argument_ptr2.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr3->Run(argument_ptr3.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr46->Run(argument_ptr4.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr5->Run(argument_ptr5.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr46->Run(argument_ptr6.get(), StreamConfig{nullptr, time_kernel});
    avg_time += invoker_ptr7->Run(argument_ptr7.get(), StreamConfig{nullptr, time_kernel});

    std::size_t num_bytes =
        (total_length * sizeof(InOutDataType) + invariantLength * 2 * sizeof(AccDataType)) + // No.1
        (invariantLength * 2 * sizeof(AccDataType) +
         invariantLength * sizeof(AccDataType)) + // No.2
        (total_length * (2 * sizeof(InOutDataType) + 2 * sizeof(AccDataType)) +
         total_length * sizeof(InOutDataType)) +                                         // No.3
        (total_length * sizeof(InOutDataType) + invariantLength * sizeof(AccDataType)) + // No.4
        (total_length * (2 * sizeof(InOutDataType) + 2 * sizeof(AccDataType)) +
         total_length * sizeof(InOutDataType)) +                                         // No.5
        (total_length * sizeof(InOutDataType) + invariantLength * sizeof(AccDataType)) + // No.6
        (total_length * (2 * sizeof(InOutDataType) + 3 * sizeof(AccDataType)) +
         total_length * sizeof(InOutDataType)); // No.7

    if(time_kernel)
    {
        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    };

    return (0);
};

}; // end of namespace batchnorm
