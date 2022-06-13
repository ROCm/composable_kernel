#ifndef DEVICE_SOFTMAX_HPP
#define DEVICE_SOFTMAX_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_reduce.hpp"
#include "device_reduce_multiblock.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_softmax.hpp"
#include "gridwise_set_buffer_value.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// TODO ANT: refactor so can reuse most of existing reduction codes
// TODO ANT: compose, not inherit, DeviceReduceMultiblock because softmax is MK in MK out
template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          bool PropagateNan,
          typename InElementwiseOperation, // TODO ANT: remove
          typename AccElementwiseOperation, // TODO ANT: remove
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceSoftmax : public DeviceReduceMultiBlock<InDataType,
                                                     AccDataType,
                                                     OutDataType,
                                                     Rank,
                                                     NumReduceDim,
                                                     reduce::Add<AccDataType>, // TODO ANT: make poisoned
                                                     InElementwiseOperation,
                                                     AccElementwiseOperation,
                                                     InMemoryDataOperationEnum::Set,
                                                     PropagateNan,
                                                     false, // OutputIndex
                                                     false, // HaveIndexInputIfOutputIndex
                                                     BlockSize,
                                                     MThreadClusterSize,
                                                     KThreadClusterSize,
                                                     MThreadSliceSize,
                                                     KThreadSliceSize,
                                                     InSrcVectorDim,
                                                     InSrcVectorSize,
                                                     OutDstVectorSize,
                                                     false> // MultiBlockReduction
{
    using Base = DeviceReduceMultiBlock<InDataType,
                                        AccDataType,
                                        OutDataType,
                                        Rank,
                                        NumReduceDim,
                                        reduce::Add<AccDataType>,
                                        InElementwiseOperation,
                                        AccElementwiseOperation,
                                        InMemoryDataOperationEnum::Set,
                                        PropagateNan,
                                        false, // OutputIndex
                                        false, // HaveIndexInputIfOutputIndex
                                        BlockSize,
                                        MThreadClusterSize,
                                        KThreadClusterSize,
                                        MThreadSliceSize,
                                        KThreadSliceSize,
                                        InSrcVectorDim,
                                        InSrcVectorSize,
                                        OutDstVectorSize,
                                        false>; // MultiBlockReduction


    using IndexDataType = int32_t;

    using Base::HaveIndexInput;

    using Base::NumInvariantDim;

    using Base::numSrcDim;
    using Base::numDstDim;
    using Base::reduceAllDim;

    using Base::use_multiblock;

    static_assert(!use_multiblock,
                  "softmax kernel requires reduction op be done by single workgroup");

    using Base::K_BlockTileSize;
    using Base::M_BlockTileSize;

    using InGridDesc_M_K = decltype(Base::MakeSrc2dDescriptor({1}, {1}, 1, 1));
    using OutGridDesc_M  = decltype(Base::MakeDst1dDescriptor({1}, {1})); // TODO ANT: M_K

    using GridwiseReduce = GridwiseSoftmax_mk_to_m<InDataType,
                                                   OutDataType,
                                                   AccDataType,
                                                   InGridDesc_M_K,
                                                   OutGridDesc_M,
                                                   InElementwiseOperation,
                                                   AccElementwiseOperation,
                                                   PropagateNan,
                                                   BlockSize,
                                                   MThreadClusterSize,
                                                   KThreadClusterSize,
                                                   MThreadSliceSize,
                                                   KThreadSliceSize,
                                                   InSrcVectorDim,
                                                   InSrcVectorSize,
                                                   OutDstVectorSize>;
    struct Argument : public Base::Argument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> inStrides,
                 const std::vector<index_t> outLengths,
                 const std::vector<index_t> outStrides,
                 const std::vector<int> reduceDims,
                 float alpha,
                 float beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev,
                 const InElementwiseOperation in_elementwise_op,
                 const AccElementwiseOperation acc_elementwise_op)
            : Base::Argument(inLengths,
                             inStrides,
                             outLengths,
                             outStrides,
                             reduceDims,
                             alpha,
                             beta,
                             in_dev,
                             nullptr,
                             out_dev,
                             nullptr,
                             in_elementwise_op,
                             acc_elementwise_op)
        {
            std::cout << "blkGroupSize= " << this->blkGroupSize
                      << ", numBlockTileIteration= " << this->numBlockTileIteration
                      << ", gridSize=" << this->gridSize
                      << ", M_BlockTileSize=" << M_BlockTileSize
                      << ", invariant_total_length=" << this->invariant_total_length
                      << std::endl;
        }
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto in_grid_desc_m_k = Base::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto out_grid_desc_m =
                Base::MakeDst1dDescriptor(arg.outLengths_, arg.outStrides_);

            const auto kernel_main = kernel_reduce_multiblock<GridwiseReduce,
                                                              false, // TODO ANT: remove
                                                              false, // TODO ANT: remove
                                                              InDataType,
                                                              OutDataType,
                                                              AccDataType,
                                                              int32_t,
                                                              InGridDesc_M_K,
                                                              OutGridDesc_M,
                                                              InElementwiseOperation,
                                                              AccElementwiseOperation>;

            float avg_time = 0;

            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize),
                                               dim3(BlockSize),
                                               0,
                                               in_grid_desc_m_k,
                                               out_grid_desc_m,
                                               arg.in_elementwise_op_,
                                               arg.acc_elementwise_op_,
                                               arg.blkGroupSize,
                                               arg.numBlockTileIteration,
                                               arg.alpha_,
                                               arg.in_dev_,
                                               arg.in_index_dev_,
                                               arg.beta_,
                                               arg.out_dev_,
                                               arg.out_index_dev_);

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
        return Base::IsSupportedArgument(p_arg);

        // TODO ANT: softmax specific checks
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<index_t> outLengths,
                        const std::vector<index_t> outStrides,
                        const std::vector<int> reduceDims,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        const void* /* in_index_dev */,
                        void* out_dev,
                        void* /* out_index_dev */,
                        const InElementwiseOperation in_elementwise_op,
                        const AccElementwiseOperation acc_elementwise_op) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          reduceDims,
                                          alpha,
                                          beta,
                                          static_cast<const InDataType*>(in_dev),
                                          static_cast<OutDataType*>(out_dev),
                                          in_elementwise_op,
                                          acc_elementwise_op);
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
#endif // DEVICE_SOFTMAX_HPP
