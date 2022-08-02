// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_softmax.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_buffer_value.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceSoftmax : public DeviceNormalization
{
    static constexpr index_t kRank         = Rank;
    static constexpr index_t kNumReduceDim = NumReduceDim;

    virtual index_t GetRank() const override { return kRank; }

    virtual index_t GetNumReduceDim() const override { return kNumReduceDim; }

    using PassThrough = tensor_operation::element_wise::PassThrough;

    // Used for freeloading of some handy functions from DeviceReduceMultiBlock
    using Reduction = DeviceReduceMultiBlock<InDataType,
                                             AccDataType,
                                             OutDataType,
                                             Rank,
                                             NumReduceDim,
                                             reduce::Add,
                                             PassThrough, // InElementwiseOperation
                                             PassThrough, // AccElementwiseOperation
                                             InMemoryDataOperationEnum::Set,
                                             false, // PropagateNan
                                             false, // OutputIndex
                                             false, // HaveIndexInputIfOutputIndex
                                             BlockSize,
                                             MThreadClusterSize,
                                             KThreadClusterSize,
                                             MThreadSliceSize,
                                             KThreadSliceSize,
                                             InSrcVectorDim,
                                             InSrcVectorSize,
                                             1>; // OutDstVectorSize

    using GridDesc_M_K = decltype(Reduction::MakeSrc2dDescriptor({1}, {1}, 1, 1));

    using GridwiseSoftmaxGeneric = GridwiseSoftmax_mk_to_mk<InDataType,
                                                            OutDataType,
                                                            AccDataType,
                                                            GridDesc_M_K,
                                                            BlockSize,
                                                            MThreadClusterSize,
                                                            KThreadClusterSize,
                                                            MThreadSliceSize,
                                                            KThreadSliceSize,
                                                            InSrcVectorDim,
                                                            InSrcVectorSize,
                                                            OutDstVectorSize,
                                                            false>;

    using GridwiseSoftmaxSweepOnce = GridwiseSoftmax_mk_to_mk<InDataType,
                                                              OutDataType,
                                                              AccDataType,
                                                              GridDesc_M_K,
                                                              BlockSize,
                                                              MThreadClusterSize,
                                                              KThreadClusterSize,
                                                              MThreadSliceSize,
                                                              KThreadSliceSize,
                                                              InSrcVectorDim,
                                                              InSrcVectorSize,
                                                              OutDstVectorSize,
                                                              true>;

    struct Argument : public Reduction::Argument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> inStrides,
                 const std::vector<index_t> reduceDims,
                 AccDataType alpha,
                 AccDataType beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev)
            : Reduction::Argument(inLengths,
                                  inStrides,
                                  {},
                                  {},
                                  reduceDims,
                                  0.0f, // alpha
                                  0.0f, // beta
                                  in_dev,
                                  nullptr,
                                  out_dev,
                                  nullptr,
                                  PassThrough{},
                                  PassThrough{}),
              // FIXME: The base class DeviceReduceMultiBlock::Argument only supports alpha/beta of
              // float32 precision. Make it support any data type so the fields can be removed.
              alpha_(alpha),
              beta_(beta)
        {
            // std::cout << "blkGroupSize= " << this->blkGroupSize
            //           << ", numBlockTileIteration= " << this->numBlockTileIteration
            //           << ", gridSize=" << this->gridSize
            //           << ", invariant_total_length=" << this->invariant_total_length <<
            //           std::endl;
        }

        AccDataType alpha_;
        AccDataType beta_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto in_grid_desc_m_k = Reduction::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto out_grid_desc_m_k = Reduction::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);

            bool sweep_once =
                in_grid_desc_m_k.GetLength(Number<1>{}) <= KThreadClusterSize * KThreadSliceSize;

            const auto kernel_main = sweep_once ? kernel_softmax<GridwiseSoftmaxSweepOnce,
                                                                 InDataType,
                                                                 OutDataType,
                                                                 AccDataType,
                                                                 GridDesc_M_K>
                                                : kernel_softmax<GridwiseSoftmaxGeneric,
                                                                 InDataType,
                                                                 OutDataType,
                                                                 AccDataType,
                                                                 GridDesc_M_K>;

            float avg_time = 0;

            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize),
                                               dim3(BlockSize),
                                               0,
                                               in_grid_desc_m_k,
                                               out_grid_desc_m_k,
                                               arg.blkGroupSize,
                                               arg.numBlockTileIteration,
                                               arg.alpha_,
                                               arg.in_dev_,
                                               arg.beta_,
                                               arg.out_dev_);

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        if(!Reduction::IsSupportedArgument(p_arg_))
        {
            return false;
        }

        if(p_arg_->inLengths_[Rank - 1] % OutDstVectorSize != 0)
        {
            return false;
        }

        return true;
    };

    // inLengths: input tensor extent(s) from high to low dimension
    // inStrides: input tensor stride(s) from high to low dimension
    // reduceDims: the dimension(s) the softmax normalization operate on
    // alpha: typeless pointer in host memory storing the alpha scaling value as type AccDataType
    // beta: typeless pointer in host memory storing the beta scaling value as type AccDataType
    // in_dev: typeless const pointer in device memory storing the input tensor
    // out_dev: typeless pointer in device memory storing the output tensor
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<index_t> inLengths,
                                                      const std::vector<index_t> inStrides,
                                                      const std::vector<int> reduceDims,
                                                      const void* alpha,
                                                      const void* beta,
                                                      const void* in_dev,
                                                      void* out_dev) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          reduceDims,
                                          *static_cast<const AccDataType*>(alpha),
                                          *static_cast<const AccDataType*>(beta),
                                          static_cast<const InDataType*>(in_dev),
                                          static_cast<OutDataType*>(out_dev));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceReduceSoftmax<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "InSrcVectorDim_" << InSrcVectorDim << "_InSrcVectorSize_" << InSrcVectorSize << "_OutDstVectorSize_" << OutDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
