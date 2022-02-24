#ifndef DEVICE_REDUCE_MULTIBLOCK_PARTIAL_REDUCE_HPP
#define DEVICE_REDUCE_MULTIBLOCK_PARTIAL_REDUCE_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_reduce.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_2d_reduction_multiblock_partial_reduce.hpp"

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
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceReduceMultiBlockPartialReduce
    : public DeviceReduce<InElementwiseOperation, AccElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert(OutDstVectorSize == 1, "OutDstVectorSize must be 1 for MultiBlockPartialReduce!");

    using OuterDims = decltype(get_outer_dims<Rank, InnerDims>());

    static constexpr index_t srcDims    = Rank;
    static constexpr index_t dstDims    = (OuterDims::Size() == 0) ? 1 : OuterDims::Size();
    static constexpr bool reduceAllDims = (OuterDims::Size() == 0);

    static constexpr int M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr int K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    size_t getWorkspaceSizeInBytes(const std::vector<int>& inLengths) override
    {
        size_t outer_total_length;
        size_t inner_total_length;

        std::tie(outer_total_length, inner_total_length) =
            get_2d_lengths<Rank, InnerDims>(inLengths);

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

        int blkGroupSize = (inner_total_length + (K_BlockTileSize * iterations) - 1) /
                           (K_BlockTileSize * iterations);

        size_t workspace_size = outer_total_length * blkGroupSize;

        size_t wsSizeInBytes =
            !NeedIndices ? workspace_size * sizeof(AccDataType)
                         : workspace_size * (sizeof(AccDataType) + sizeof(int)) + 64 + sizeof(int);

        return (wsSizeInBytes);
    };

    bool hasFurtherCall() override { return (true); };

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides,
                                    int blkGroupSize,
                                    int kBlockTileIterations)
    {
        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<srcDims>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<srcDims>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
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

        const auto outerLen = in_grid_desc_m_k.GetLength(Number<0>{});
        const auto innerLen = in_grid_desc_m_k.GetLength(Number<1>{});

        const int reduceSizePerBlock = K_BlockTileSize * kBlockTileIterations;
        const auto inPad_M = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;
        const auto inPad_K = reduceSizePerBlock * blkGroupSize - innerLen;

        auto in_grid_desc_m_k_padded =
            transform_tensor_descriptor(in_grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(outerLen, inPad_M),
                                                   make_right_pad_transform(innerLen, inPad_K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

    static auto MakeWorkspace2dDescriptor(int outerLen, int blkGroupSize)
    {
        auto ws_desc_m_k = make_naive_tensor_descriptor_packed(make_tuple(outerLen, blkGroupSize));

        const auto wsPad = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;

        auto ws_desc_m_k_padded =
            transform_tensor_descriptor(ws_desc_m_k,
                                        make_tuple(make_right_pad_transform(outerLen, wsPad),
                                                   make_pass_through_transform(blkGroupSize)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (ws_desc_m_k_padded);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t>& inLengths,
                 const std::vector<index_t>& inStrides,
                 const std::vector<index_t>& outLengths,
                 const std::vector<index_t>& outStrides,
                 float alpha,
                 float beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev,
                 index_t* out_indices_dev,
                 AccDataType* workspace_dev,
                 const InElementwiseOperation& in_elementwise_op,
                 const AccElementwiseOperation& acc_elementwise_op)
            : in_dev_{in_dev},
              out_dev_{out_dev},
              out_indices_dev_{out_indices_dev},
              workspace_dev_{workspace_dev}
        {
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

            size_t ws_buf2_bytes_offset = math::integer_least_multiple(
                outer_total_length * blkGroupSize * sizeof(AccDataType), 64);

            if constexpr(NeedIndices)
                workspace_indices_dev_ = reinterpret_cast<int*>(
                    reinterpret_cast<char*>(workspace_dev_) + ws_buf2_bytes_offset);
            else
                workspace_indices_dev_ = nullptr;
        }

        std::vector<int> inLengths_;
        std::vector<int> inStrides_;
        std::vector<int> outLengths_;
        std::vector<int> outStrides_;

        AccDataType alpha_;
        OutDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;
        index_t* out_indices_dev_;
        AccDataType* workspace_dev_;
        index_t* workspace_indices_dev_;

        InElementwiseOperation in_elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

        int outer_lowest_length;
        int inner_lowest_length;
        size_t outer_total_length;
        size_t inner_total_length;

        index_t blkGroupSize;
        index_t kBlockTileIterations;
        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto in_grid_desc_m_k = DeviceReduceMultiBlockPartialReduce::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.kBlockTileIterations);
            const auto ws_desc_m_k = DeviceReduceMultiBlockPartialReduce::MakeWorkspace2dDescriptor(
                arg.outer_total_length, arg.blkGroupSize);
            using InGridDesc_M_K    = decltype(in_grid_desc_m_k);
            using WorkspaceDesc_M_K = decltype(ws_desc_m_k);

            using GridwiseReduce =
                GridwiseReduction_mk_to_mk_multiblock_partial_reduce<InDataType,
                                                                     OutDataType,
                                                                     AccDataType,
                                                                     InGridDesc_M_K,
                                                                     WorkspaceDesc_M_K,
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

            const auto kernel = kernel_partial_reduce_multiblock<GridwiseReduce,
                                                                 NeedIndices,
                                                                 InDataType,
                                                                 AccDataType,
                                                                 InGridDesc_M_K,
                                                                 WorkspaceDesc_M_K,
                                                                 InElementwiseOperation,
                                                                 AccElementwiseOperation>;

            avg_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(arg.gridSize),
                                              dim3(BlockSize),
                                              0,
                                              in_grid_desc_m_k,
                                              ws_desc_m_k,
                                              arg.in_elementwise_op_,
                                              arg.acc_elementwise_op_,
                                              arg.blkGroupSize,
                                              arg.kBlockTileIterations,
                                              arg.in_dev_,
                                              arg.workspace_dev_,
                                              arg.workspace_indices_dev_);

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

        if constexpr(OutDstVectorSize != 1)
            return (false);

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

        // cases with small inner_total_length should be handled by the BlockWise method
        if(pArg->inner_total_length <= BlockSize * KThreadSliceSize)
            return (false);

        return (true);
    };

    std::vector<int> getWorkspace2dLengths(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        return (std::vector<int>{static_cast<int>(pArg->outer_total_length), pArg->blkGroupSize});
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
        str << "DeviceReduceMultiBlockPartialReduce<" << BlockSize << ",";
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
