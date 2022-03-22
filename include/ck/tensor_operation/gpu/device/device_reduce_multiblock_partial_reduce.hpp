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
          index_t Rank,
          index_t NumReduceDim,
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

    static_assert((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                      (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert(OutDstVectorSize == 1, "OutDstVectorSize must be 1 for MultiBlockPartialReduce!");

    using IndexDataType = int32_t;

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t numSrcDim = Rank;
    static constexpr index_t numDstDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim = (NumInvariantDim == 0);

    static constexpr int M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr int K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static constexpr int MaxBlockGroupSize = 256;

    long_index_t GetWorkspaceSizeInBytes(const std::vector<int> inLengths,
                                         const std::vector<int> reduceDims) override
    {
        size_t invariant_total_length;
        size_t reduce_total_length;

        auto inLengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inLengths, reduceDims);

        std::tie(invariant_total_length, reduce_total_length) =
            get_2d_lengths<Rank, NumReduceDim>(inLengths_);

        int iterations = 1;
        while(true)
        {
            int testBlkGroupSize = (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                                   (K_BlockTileSize * iterations);

            if(testBlkGroupSize <= MaxBlockGroupSize)
                break;

            iterations++;
        };

        int blkGroupSize = (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                           (K_BlockTileSize * iterations);

        long_index_t workspace_size = invariant_total_length * blkGroupSize;

        long_index_t wsSizeInBytes =
            !NeedIndices
                ? workspace_size * sizeof(AccDataType)
                : workspace_size * (sizeof(AccDataType) + sizeof(int32_t)) + 64 + sizeof(int);

        return (wsSizeInBytes);
    };

    bool HasFurtherCall() override { return (true); };

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides,
                                    int blkGroupSize,
                                    int kBlockTileIterations)
    {
        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<numSrcDim>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<numSrcDim>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            if constexpr(reduceAllDim)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, numSrcDim, 1>::type{}),
                    make_tuple(Sequence<0>{}));

                return transform_tensor_descriptor(one_dim_inDesc,
                                                   make_tuple(make_unmerge_transform(make_tuple(
                                                       1, one_dim_inDesc.GetLength(Number<0>{})))),
                                                   make_tuple(Sequence<0>{}),
                                                   make_tuple(Sequence<0, 1>{}));
            }
            else
            {
                using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
                using ReduceDims = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

                const auto reduceDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, ReduceDims{});
                const auto invariantDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, InvariantDims{});

                return transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(reduceDimLengths)),
                    make_tuple(InvariantDims{}, ReduceDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto invariantLength = in_grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = in_grid_desc_m_k.GetLength(Number<1>{});

        const int reduceSizePerBlock = K_BlockTileSize * kBlockTileIterations;
        const auto inPad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto inPad_K = reduceSizePerBlock * blkGroupSize - reduceLength;

        auto in_grid_desc_m_k_padded = transform_tensor_descriptor(
            in_grid_desc_m_k,
            make_tuple(make_right_pad_transform(invariantLength, inPad_M),
                       make_right_pad_transform(reduceLength, inPad_K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

    static auto MakeWorkspace2dDescriptor(int invariantLength, int blkGroupSize)
    {
        auto ws_desc_m_k =
            make_naive_tensor_descriptor_packed(make_tuple(invariantLength, blkGroupSize));

        const auto wsPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto ws_desc_m_k_padded =
            transform_tensor_descriptor(ws_desc_m_k,
                                        make_tuple(make_right_pad_transform(invariantLength, wsPad),
                                                   make_pass_through_transform(blkGroupSize)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (ws_desc_m_k_padded);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<int> inLengths,
                 const std::vector<int> inStrides,
                 const std::vector<int> outLengths,
                 const std::vector<int> outStrides,
                 const std::vector<int> reduceDims,
                 float alpha,
                 float beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev,
                 IndexDataType* out_indices_dev,
                 AccDataType* workspace_dev,
                 const InElementwiseOperation in_elementwise_op,
                 const AccElementwiseOperation acc_elementwise_op)
            : outLengths_{outLengths},
              outStrides_{outStrides},
              in_dev_{in_dev},
              out_dev_{out_dev},
              out_indices_dev_{out_indices_dev},
              workspace_dev_{workspace_dev},
              in_elementwise_op_{in_elementwise_op},
              acc_elementwise_op_{acc_elementwise_op}
        {
            inLengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inLengths, reduceDims);
            inStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inStrides, reduceDims);

            alpha_ = type_convert<AccDataType>(alpha);
            beta_  = type_convert<AccDataType>(beta);

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, NumReduceDim>(inLengths_);

            if constexpr(NumInvariantDim == 0)
                invariant_lowest_length = 1;
            else
                invariant_lowest_length = inLengths_[NumInvariantDim - 1];

            reduce_lowest_length = inLengths_[Rank - 1];

            int iterations = 1;
            while(true)
            {
                int testBlkGroupSize = (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                                       (K_BlockTileSize * iterations);

                if(testBlkGroupSize <= MaxBlockGroupSize)
                    break;

                iterations++;
            };

            blkGroupSize = (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                           (K_BlockTileSize * iterations);

            kBlockTileIterations = iterations;

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize * blkGroupSize;

            size_t ws_buf2_bytes_offset = math::integer_least_multiple(
                invariant_total_length * blkGroupSize * sizeof(AccDataType), 64);

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
        AccDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;
        IndexDataType* out_indices_dev_;
        AccDataType* workspace_dev_;
        IndexDataType* workspace_indices_dev_;

        InElementwiseOperation in_elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

        int invariant_lowest_length;
        int reduce_lowest_length;
        size_t invariant_total_length;
        size_t reduce_total_length;

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
                arg.invariant_total_length, arg.blkGroupSize);
            using InGridDesc_M_K    = decltype(in_grid_desc_m_k);
            using WorkspaceDesc_M_K = decltype(ws_desc_m_k);

            using GridwiseReduce =
                GridwiseReduction_mk_to_mk_multiblock_partial_reduce<InDataType,
                                                                     AccDataType,
                                                                     IndexDataType,
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
                                                                 IndexDataType,
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
            if constexpr(NumInvariantDim == 0)
            {
                return (false);
            }
            else
            {
                if(pArg->inStrides_[NumInvariantDim - 1] != 1)
                    return (false);

                if(pArg->invariant_lowest_length % InSrcVectorSize != 0)
                    return (false);
            };
        }
        else
        {
            if(pArg->inStrides_[Rank - 1] != 1)
                return (false);

            if(pArg->reduce_lowest_length % InSrcVectorSize != 0)
                return (false);
        };

        // cases with small reduce_total_length should be handled by the BlockWise method
        if(pArg->reduce_total_length <= BlockSize * KThreadSliceSize)
            return (false);

        return (true);
    };

    std::vector<int> GetWorkspace2dLengths(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        return (
            std::vector<int>{static_cast<int>(pArg->invariant_total_length), pArg->blkGroupSize});
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<int> inLengths,
                        const std::vector<int> inStrides,
                        const std::vector<int> outLengths,
                        const std::vector<int> outStrides,
                        const std::vector<int> reduceDims,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        void* out_dev,
                        void* out_indices_dev,
                        void* workspace_dev,
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
                                          static_cast<IndexDataType*>(out_indices_dev),
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
