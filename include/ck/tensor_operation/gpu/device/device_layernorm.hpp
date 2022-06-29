// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_layernorm.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_buffer_value.hpp"
#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          index_t Rank,
          index_t NumReduceDim,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t AffineSrcVectorDim,
          index_t AffineSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceLayernorm : public BaseOperator
{
    static_assert(
        ((AffineSrcVectorDim == 0 && MThreadSliceSize % AffineSrcVectorSize == 0) ||
         (AffineSrcVectorDim == 1 && KThreadSliceSize % AffineSrcVectorSize == 0)),
        "Invalid thread slice sizes and/or affine vector sizes configuration, please check!");

    using PassThrough = tensor_operation::element_wise::PassThrough;

    // Used for freeloading of some handy functions from DeviceReduceMultiBlock
    using Reduction = DeviceReduceMultiBlock<XDataType,
                                             AccDataType,
                                             YDataType,
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

    using GridwiseReduce = GridwiseLayernorm_mk_to_mk<XDataType,
                                                      GammaDataType,
                                                      BetaDataType,
                                                      YDataType,
                                                      AccDataType,
                                                      GridDesc_M_K,
                                                      BlockSize,
                                                      MThreadClusterSize,
                                                      KThreadClusterSize,
                                                      MThreadSliceSize,
                                                      KThreadSliceSize,
                                                      InSrcVectorDim,
                                                      InSrcVectorSize,
                                                      AffineSrcVectorDim,
                                                      AffineSrcVectorSize,
                                                      OutDstVectorSize,
                                                      false>;

    struct Argument : public Reduction::Argument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> inStrides,
                 const std::vector<index_t> affineStrides,
                 const std::vector<index_t> reduceDims,
                 AccDataType epsilon,
                 const XDataType* p_x,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 YDataType* p_y)
            : Reduction::Argument(inLengths,
                                  inStrides,
                                  {},
                                  {},
                                  reduceDims,
                                  0.0f, // alpha
                                  0.0f, // beta
                                  p_x,
                                  nullptr,
                                  p_y,
                                  nullptr,
                                  PassThrough{},
                                  PassThrough{}),
              epsilon_(epsilon),
              p_gamma_(p_gamma),
              p_beta_(p_beta)
        {
            affineStrides_ =
                shuffle_tensor_dimensions<Rank, NumReduceDim>(affineStrides, reduceDims);
        }

        AccDataType epsilon_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        std::vector<index_t> affineStrides_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto in_grid_desc_m_k = Reduction::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto gamma_grid_desc_m_k = Reduction::MakeSrc2dDescriptor(
                arg.inLengths_, arg.affineStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto beta_grid_desc_m_k = Reduction::MakeSrc2dDescriptor(
                arg.inLengths_, arg.affineStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto out_grid_desc_m_k = Reduction::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);

            const auto kernel_main = kernel_layernorm<GridwiseReduce,
                                                      XDataType,
                                                      GammaDataType,
                                                      BetaDataType,
                                                      YDataType,
                                                      AccDataType,
                                                      GridDesc_M_K>;

            float avg_time = 0;

            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize),
                                               dim3(BlockSize),
                                               0,
                                               in_grid_desc_m_k,
                                               gamma_grid_desc_m_k,
                                               beta_grid_desc_m_k,
                                               out_grid_desc_m_k,
                                               arg.blkGroupSize,
                                               arg.numBlockTileIteration,
                                               arg.epsilon_,
                                               arg.in_dev_,
                                               arg.p_gamma_,
                                               arg.p_beta_,
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
        // TODO - Check AffineSrcVectorDim and AffineSrcVectorSize

        return true;
    };

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<index_t> inLengths,
                                                      const std::vector<index_t> inStrides,
                                                      const std::vector<index_t> affineStrides,
                                                      const std::vector<int> reduceDims,
                                                      AccDataType epsilon,
                                                      const void* p_x,
                                                      const void* p_gamma,
                                                      const void* p_beta,
                                                      void* p_y)
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          affineStrides,
                                          reduceDims,
                                          epsilon,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const GammaDataType*>(p_gamma),
                                          static_cast<const BetaDataType*>(p_beta),
                                          static_cast<YDataType*>(p_y));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() { return std::make_unique<Invoker>(); };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceLayernorm<" << BlockSize << ",";
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
