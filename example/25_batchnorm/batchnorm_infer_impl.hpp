#pragma once

#include <cassert>

#include "reduction_operator.hpp"
#include "device_5ary_elementwise.hpp"
#include "batchnorm_common.hpp"

namespace batchnorm {

template <typename InOutDataType,
          typename AccDataType,
          ck::index_t Rank,
          ck::index_t NumBatchNormReduceDim,
          bool fastest_dim_is_reduced = false>
int bnorm_infer(bool time_kernel,
                const std::vector<int> reduceDims,
                const std::vector<ck::index_t> xyLengths,
                const std::vector<ck::index_t> xStrides,
                const std::vector<ck::index_t> yStrides,
                const std::vector<ck::index_t> bnScaleBiasMeanVarLengths,
                const std::vector<ck::index_t> bnScaleBiasMeanVarStrides,
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

    using NormalizeOp = detail::NormalizeInInfer;

    using DeviceNormalizeInstance =
        ck::tensor_operation::device::Device5AryElementwise<InOutDataType, // x
                                                            AccDataType,   // mean
                                                            AccDataType,   // variance
                                                            AccDataType,   // Scale
                                                            AccDataType,   // Bias
                                                            InOutDataType, // y
                                                            AccDataType,   //
                                                            NormalizeOp,
                                                            Rank,
                                                            2,
                                                            1,  // scalarPerVector: x
                                                            1,  // scalarPerVector: mean
                                                            1,  // scalarPerVector: variance
                                                            1,  // scalarPerVector: Scale
                                                            1,  // scalarPerVector: Bias
                                                            1>; // scalarPerVector: y

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

    float avg_time        = 0.0f;
    std::size_t num_bytes = 0;

    auto dev_normalize = DeviceNormalizeInstance{};

    auto argument_ptr1 = dev_normalize.MakeArgumentPointer(p_x,
                                                           p_estimatedMean,     // mean
                                                           p_estimatedVariance, // meansquare
                                                           p_scale,             // scale
                                                           p_bias,              // bias
                                                           p_y,
                                                           xyLengths,
                                                           xStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           yStrides,
                                                           NormalizeOp{epsilon});

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

}; // end of namespace batchnorm
