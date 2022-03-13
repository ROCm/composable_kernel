#ifndef DEVICE_REDUCE_MULTIBLOCK_ATOMIC_ADD_HPP
#define DEVICE_REDUCE_MULTIBLOCK_ATOMIC_ADD_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_reduce.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_2d_reduction_multiblock_atomic_add.hpp"
#include "gridwise_set_buffer_value.hpp"

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
struct DeviceReduceMultiBlockAtomicAdd
    : public DeviceReduce<InElementwiseOperation, AccElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (MThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    using IndexDataType = int32_t;

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t numSrcDim = Rank;
    static constexpr index_t numDstDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim = (NumInvariantDim == 0);

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

    static auto MakeDst1dDescriptor(const std::vector<int>& outLengths,
                                    const std::vector<int>& outStrides)
    {
        const auto tupleDstLengths = make_tuple_from_array(outLengths, Number<numDstDim>{});
        const auto tupleDstStrides = make_tuple_from_array(outStrides, Number<numDstDim>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out_grid_desc_m = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, numDstDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto invariantLength = out_grid_desc_m.GetLength(Number<0>{});

        const auto outPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto out_grid_desc_m_padded = transform_tensor_descriptor(
            out_grid_desc_m,
            make_tuple(make_right_pad_transform(invariantLength, outPad)),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0>{}));
        return (out_grid_desc_m_padded);
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
              in_elementwise_op_{in_elementwise_op},
              acc_elementwise_op_{acc_elementwise_op}
        {
            (void)out_indices_dev;
            (void)workspace_dev;

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

                // we want the blkGroupSize be not more than 128
                if(testBlkGroupSize <= 128)
                    break;

                iterations++;
            };

            blkGroupSize = (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                           (K_BlockTileSize * iterations);

            kBlockTileIterations = iterations;

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize * blkGroupSize;

            gridSize_pre =
                math::integer_least_multiple(invariant_total_length, BlockSize) / BlockSize;
        }

        std::vector<int> inLengths_;
        std::vector<int> inStrides_;
        std::vector<int> outLengths_;
        std::vector<int> outStrides_;

        AccDataType alpha_;
        AccDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;

        InElementwiseOperation in_elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

        int invariant_lowest_length;
        int reduce_lowest_length;
        size_t invariant_total_length;
        size_t reduce_total_length;

        index_t blkGroupSize;
        index_t kBlockTileIterations;
        size_t gridSize;

        size_t gridSize_pre;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto in_grid_desc_m_k = DeviceReduceMultiBlockAtomicAdd::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.kBlockTileIterations);
            const auto out_grid_desc_m = DeviceReduceMultiBlockAtomicAdd::MakeDst1dDescriptor(
                arg.outLengths_, arg.outStrides_);
            using InGridDesc_M_K = decltype(in_grid_desc_m_k);
            using OutGridDesc_M  = decltype(out_grid_desc_m);

            using GridwiseReduce =
                GridwiseReduction_mk_to_m_multiblock_atomic_add<InDataType,
                                                                OutDataType,
                                                                AccDataType,
                                                                InGridDesc_M_K,
                                                                OutGridDesc_M,
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

            const auto kernel_pre  = kernel_buffer_set_value<BlockSize, OutDataType, OutGridDesc_M>;
            const auto kernel_main = kernel_reduce_multiblock_atocmi_add<GridwiseReduce,
                                                                         InDataType,
                                                                         OutDataType,
                                                                         AccDataType,
                                                                         InGridDesc_M_K,
                                                                         OutGridDesc_M,
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
                              out_grid_desc_m,
                              arg.out_dev_,
                              static_cast<OutDataType>(0.0f));

                launch_kernel(kernel_main,
                              dim3(arg.gridSize),
                              dim3(BlockSize),
                              0,
                              in_grid_desc_m_k,
                              out_grid_desc_m,
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

        if(static_cast<float>(pArg->beta_) != 0.0f)
            return (false);

        // To improve
        if(pArg->invariant_lowest_length % OutDstVectorSize != 0)
            return (false);

        // cases with small reduce_total_length should be handled by the BlockWise method
        if(pArg->reduce_total_length <= BlockSize * KThreadSliceSize)
            return (false);

        // This is very strong restriction, but needed to avoid some failure
        if(pArg->invariant_lowest_length % M_BlockTileSize != 0)
            return (false);

        return (true);
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
