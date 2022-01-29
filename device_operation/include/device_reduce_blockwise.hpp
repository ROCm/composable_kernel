#ifndef DEVICE_REDUCE_BLOCKWISE_HPP
#define DEVICE_REDUCE_BLOCKWISE_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_reduce.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_2d_reduction_blockwise.hpp"

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
          int VectorDim,
          int VectorSize>
struct DeviceReduceBlockWise : public DeviceReduce<InElementwiseOperation, AccElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static constexpr bool BetaIsZero = NeedIndices;

    using OuterDims = decltype(get_outer_dims<Rank, InnerDims>());

    static constexpr index_t srcDims    = Rank;
    static constexpr index_t dstDims    = (OuterDims::Size() == 0) ? 1 : OuterDims::Size();
    static constexpr bool reduceAllDims = (OuterDims::Size() == 0);

    static constexpr int M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr int K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides)
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

        const auto inPad_M = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;
        const auto inPad_K = math::integer_least_multiple(innerLen, K_BlockTileSize) - innerLen;

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

        const auto inPad = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;

        auto out1dDesc_M =
            transform_tensor_descriptor(out1dDesc,
                                        make_tuple(make_right_pad_transform(outerLen, inPad)),
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
                 const InElementwiseOperation& inElementwiseOp,
                 const AccElementwiseOperation& accElementwiseOp)
            : in_dev_{in_dev}, out_dev_{out_dev}, out_indices_dev_{out_indices_dev}
        {
            (void)workspace_dev;

            inLengths_  = inLengths;
            inStrides_  = inStrides;
            outLengths_ = outLengths;
            outStrides_ = outStrides;

            inElementwiseOp_  = inElementwiseOp;
            accElementwiseOp_ = accElementwiseOp;

            alpha_ = static_cast<AccDataType>(alpha);
            beta_  = static_cast<OutDataType>(beta);

            std::tie(outer_total_length, inner_total_length) =
                get_2d_lengths<Rank, InnerDims>(inLengths);

            if constexpr(OuterDims::Size() == 0)
                outer_lowest_length = 1;
            else
                outer_lowest_length = inLengths[OuterDims::At(OuterDims::Size() - 1)];

            inner_lowest_length = inLengths[InnerDims::At(InnerDims::Size() - 1)];

            gridSize =
                math::integer_least_multiple(outer_total_length, M_BlockTileSize) / M_BlockTileSize;
        }

        std::vector<int> inLengths_;
        std::vector<int> inStrides_;
        std::vector<int> outLengths_;
        std::vector<int> outStrides_;

        AccDataType alpha_;
        OutDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;
        int* out_indices_dev_;

        InElementwiseOperation inElementwiseOp_;
        AccElementwiseOperation accElementwiseOp_;

        int outer_lowest_length;
        int inner_lowest_length;
        size_t outer_total_length;
        size_t inner_total_length;

        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto in2dDesc =
                DeviceReduceBlockWise::MakeSrc2dDescriptor(arg.inLengths_, arg.inStrides_);
            const auto out1dDesc =
                DeviceReduceBlockWise::MakeDst1dDescriptor(arg.outLengths_, arg.outStrides_);
            using In2dDescType  = decltype(in2dDesc);
            using Out1dDescType = decltype(out1dDesc);

            using GridwiseReduce = GridwiseReduction_xy_to_x_blockwise<InDataType,
                                                                       OutDataType,
                                                                       AccDataType,
                                                                       In2dDescType,
                                                                       Out1dDescType,
                                                                       ReduceOperation,
                                                                       InElementwiseOperation,
                                                                       AccElementwiseOperation,
                                                                       PropagateNan,
                                                                       BetaIsZero,
                                                                       BlockSize,
                                                                       MThreadClusterSize,
                                                                       KThreadClusterSize,
                                                                       MThreadSliceSize,
                                                                       KThreadSliceSize,
                                                                       VectorDim,
                                                                       VectorSize>;

            float avg_time = 0;

            const auto kernel = kernel_reduce_blockwise<GridwiseReduce,
                                                        NeedIndices,
                                                        InDataType,
                                                        OutDataType,
                                                        AccDataType,
                                                        In2dDescType,
                                                        Out1dDescType,
                                                        InElementwiseOperation,
                                                        AccElementwiseOperation>;

            avg_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(arg.gridSize),
                                              dim3(BlockSize),
                                              0,
                                              in2dDesc,
                                              out1dDesc,
                                              arg.inElementwiseOp_,
                                              arg.accElementwiseOp_,
                                              arg.alpha_,
                                              arg.in_dev_,
                                              arg.beta_,
                                              arg.out_dev_,
                                              nullptr,
                                              arg.out_indices_dev_);

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

        if constexpr(VectorDim == 0)
        {
            if constexpr(OuterDims::Size() == 0)
                return (false);

            if(pArg->inStrides_[OuterDims::At(OuterDims::Size() - 1)] != 1)
                return (false);

            if(pArg->outer_lowest_length % VectorSize != 0)
                return (false);
        }
        else
        {
            if(pArg->inStrides_[InnerDims::At(InnerDims::Size() - 1)] != 1)
                return (false);

            if(pArg->inner_lowest_length % VectorSize != 0)
                return (false);
        };

        // cases with very small inner_total_length should be handled by the ThreadWise method
        if(pArg->inner_total_length / KThreadSliceSize < 2)
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
                        const InElementwiseOperation& inElementwiseOp,
                        const AccElementwiseOperation& accElementwiseOp) override
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
                                          inElementwiseOp,
                                          accElementwiseOp);
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        str << "DeviceReduceBlockWise<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ">";

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
