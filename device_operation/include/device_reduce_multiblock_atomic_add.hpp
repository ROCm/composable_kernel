#ifndef DEVICE_REDUCE_MULTIBLOCK_ATOMIC_ADD_HPP
#define DEVICE_REDUCE_MULTIBLOCK_ATOMIC_ADD_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_reduce.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_2d_reduction_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          typename InnerDims,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          bool PropagateNan,
          bool NeedIndices,
          int BlockSize,
          int MThreadClusterSize,
          int KThreadClusterSize,
          int MThreadSliceSize,
          int KThreadSliceSize,
          int InSrcVectorDim,
          int InSrcVectorSize,
          int OutDstVectorSize>
struct DeviceReduceMultiBlockAtomicAdd
    : public DeviceReduce<InElementwiseOperation, AccElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    using OuterDims = decltype(get_outer_dims<Rank, InnerDims>());

    static constexpr index_t srcDims    = Rank;
    static constexpr index_t dstDims    = (OuterDims::Size() == 0) ? 1 : OuterDims::Size();
    static constexpr bool reduceAllDims = (OuterDims::Size() == 0);

    static constexpr bool support_AtomicAdd =
        std::is_same<OutDataType, float>::value || std::is_same<OutDataType, double>::value;

    static_assert(!NeedIndices && support_AtomicAdd,
                  "MultiBlockAtomicAdd method can only be used with non-indiced operation and when "
                  "having float/double output type!");

    static constexpr int M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr int K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides,
                                    int blkGroupSize,
                                    int kBlockTileIterations)
    {
        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<srcDims>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<srcDims>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in2dDesc = [&]() {
            if constexpr(reduceAllDims)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, srcDims, 1>::type{}),
                    make_tuple(Sequence<0>{}));

                return transform_tensor_descriptor(one_dim_inDesc,
                                                   make_tuple(make_unmerge_transform(make_tuple(
                                                       1, one_dim_inDesc.GetLength(Number<0>{})))),
                                                   make_tuple(Sequence<0>{}),
                                                   make_tuple(Sequence<0, 1>{}));
            }
            else
            {
                const auto toReduceDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, InnerDims{});
                const auto invariantDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, OuterDims{});

                return transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(toReduceDimLengths)),
                    make_tuple(OuterDims{}, InnerDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto outerLen = in2dDesc.GetLength(Number<0>{});
        const auto innerLen = in2dDesc.GetLength(Number<1>{});

        const int reduceSizePerBlock = K_BlockTileSize * kBlockTileIterations;
        const auto inPad_M = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;
        const auto inPad_K = reduceSizePerBlock * blkGroupSize - innerLen;

        auto in2dDesc_M_K =
            transform_tensor_descriptor(in2dDesc,
                                        make_tuple(make_right_pad_transform(outerLen, inPad_M),
                                                   make_right_pad_transform(innerLen, inPad_K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in2dDesc_M_K);
    };

    static auto MakeDst1dDescriptor(const std::vector<int>& outLengths,
                                    const std::vector<int>& outStrides)
    {
        const auto tupleDstLengths = make_tuple_from_array(outLengths, Number<dstDims>{});
        const auto tupleDstStrides = make_tuple_from_array(outStrides, Number<dstDims>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out1dDesc = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto outerLen = out1dDesc.GetLength(Number<0>{});

        const auto outPad = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;

        auto out1dDesc_M =
            transform_tensor_descriptor(out1dDesc,
                                        make_tuple(make_right_pad_transform(outerLen, outPad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (out1dDesc_M);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<int>& inLengths,
                 const std::vector<int>& inStrides,
                 const std::vector<int>& outLengths,
                 const std::vector<int>& outStrides,
                 float alpha,
                 float beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev,
                 int* out_indices_dev,
                 AccDataType* workspace_dev,
                 const InElementwiseOperation& in_elementwise_op,
                 const AccElementwiseOperation& acc_elementwise_op)
            : in_dev_{in_dev}, out_dev_{out_dev}
        {
            (void)out_indices_dev;
            (void)workspace_dev;

            inLengths_  = inLengths;
            inStrides_  = inStrides;
            outLengths_ = outLengths;
            outStrides_ = outStrides;

            in_elementwise_op_  = in_elementwise_op;
            acc_elementwise_op_ = acc_elementwise_op;

            alpha_ = static_cast<AccDataType>(alpha);
            beta_  = static_cast<OutDataType>(beta);

            std::tie(outer_total_length, inner_total_length) =
                get_2d_lengths<Rank, InnerDims>(inLengths);

            if constexpr(OuterDims::Size() == 0)
                outer_lowest_length = 1;
            else
                outer_lowest_length = inLengths[OuterDims::At(OuterDims::Size() - 1)];

            inner_lowest_length = inLengths[InnerDims::At(InnerDims::Size() - 1)];

            int iterations = 1;
            while(true)
            {
                int test_blkGroupSize = (inner_total_length + (K_BlockTileSize * iterations) - 1) /
                                        (K_BlockTileSize * iterations);

                // we want the blkGroupSize be not more than 128
                if(test_blkGroupSize <= 128)
                    break;

                iterations++;
            };

            blkGroupSize = (inner_total_length + (K_BlockTileSize * iterations) - 1) /
                           (K_BlockTileSize * iterations);

            kBlockTileIterations = iterations;

            gridSize = math::integer_least_multiple(outer_total_length, M_BlockTileSize) /
                       M_BlockTileSize * blkGroupSize;

            gridSize_pre = math::integer_least_multiple(outer_total_length, BlockSize) / BlockSize;
        }

        std::vector<int> inLengths_;
        std::vector<int> inStrides_;
        std::vector<int> outLengths_;
        std::vector<int> outStrides_;

        AccDataType alpha_;
        OutDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;

        InElementwiseOperation in_elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

        int outer_lowest_length;
        int inner_lowest_length;
        size_t outer_total_length;
        size_t inner_total_length;

        int blkGroupSize;
        int kBlockTileIterations;
        size_t gridSize;

        size_t gridSize_pre;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto in2dDesc = DeviceReduceMultiBlockAtomicAdd::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.kBlockTileIterations);
            const auto out1dDesc = DeviceReduceMultiBlockAtomicAdd::MakeDst1dDescriptor(
                arg.outLengths_, arg.outStrides_);
            using In2dDescType  = decltype(in2dDesc);
            using Out1dDescType = decltype(out1dDesc);

            using GridwiseReduce =
                GridwiseReduction_xy_to_x_multiblock_atomic_add<InDataType,
                                                                OutDataType,
                                                                AccDataType,
                                                                In2dDescType,
                                                                Out1dDescType,
                                                                ReduceOperation,
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

            float avg_time = 0;

            KernelTimer timer;

            const auto kernel_pre  = kernel_buffer_set_value<BlockSize, OutDataType, Out1dDescType>;
            const auto kernel_main = kernel_reduce_multiblock_atocmi_add<GridwiseReduce,
                                                                         InDataType,
                                                                         OutDataType,
                                                                         AccDataType,
                                                                         In2dDescType,
                                                                         Out1dDescType,
                                                                         InElementwiseOperation,
                                                                         AccElementwiseOperation>;

            printf("launch_and_time_kernel: grid_dim {%ld, 1, 1}, block_dim {%d, 1, 1} \n",
                   arg.gridSize,
                   BlockSize);
            printf("Warm up\n");

            for(int i = 0; i < nrepeat + 1; i++)
            {
                if(i == 1)
                    timer.Start();

                launch_kernel(kernel_pre,
                              dim3(arg.gridSize_pre),
                              dim3(BlockSize),
                              0,
                              out1dDesc,
                              arg.out_dev_,
                              static_cast<OutDataType>(0.0f));

                launch_kernel(kernel_main,
                              dim3(arg.gridSize),
                              dim3(BlockSize),
                              0,
                              in2dDesc,
                              out1dDesc,
                              arg.in_elementwise_op_,
                              arg.acc_elementwise_op_,
                              arg.blkGroupSize,
                              arg.kBlockTileIterations,
                              arg.alpha_,
                              arg.in_dev_,
                              arg.out_dev_);
            };

            timer.End();

            avg_time = timer.GetElapsedTime() / nrepeat;

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if constexpr(InSrcVectorDim == 0)
        {
            if constexpr(OuterDims::Size() == 0)
                return (false);

            if(pArg->inStrides_[OuterDims::At(OuterDims::Size() - 1)] != 1)
                return (false);

            if(pArg->outer_lowest_length % InSrcVectorSize != 0)
                return (false);
        }
        else
        {
            if(pArg->inStrides_[InnerDims::At(InnerDims::Size() - 1)] != 1)
                return (false);

            if(pArg->inner_lowest_length % InSrcVectorSize != 0)
                return (false);
        };

        if(static_cast<float>(pArg->beta_) != 0.0f)
            return (false);

        // To improve
        if(pArg->outer_lowest_length % OutDstVectorSize != 0)
            return (false);

        // cases with small inner_total_length should be handled by the BlockWise method
        if(pArg->inner_total_length <= BlockSize * KThreadSliceSize)
            return (false);

        // This is very strong restriction, but needed to avoid some failure
        if(pArg->outer_lowest_length % M_BlockTileSize != 0)
            return (false);

        return (true);
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<int>& inLengths,
                        const std::vector<int>& inStrides,
                        const std::vector<int>& outLengths,
                        const std::vector<int>& outStrides,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        void* out_dev,
                        void* out_indices_dev,
                        void* workspace_dev,
                        const InElementwiseOperation& in_elementwise_op,
                        const AccElementwiseOperation& acc_elementwise_op) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          alpha,
                                          beta,
                                          static_cast<const InDataType*>(in_dev),
                                          static_cast<OutDataType*>(out_dev),
                                          static_cast<int*>(out_indices_dev),
                                          static_cast<AccDataType*>(workspace_dev),
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
        str << "DeviceReduceMultiBlockAtomicAdd<" << BlockSize << ",";
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
#endif
