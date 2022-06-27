#ifndef DEVICE_MULTIPLE_REDUCE_MULTIBLOCK_HPP
#define DEVICE_MULTIPLE_REDUCE_MULTIBLOCK_HPP

#include <iostream>
#include <sstream>

#include "ck/utility/sequence.hpp"
#include "ck/utility/reduction_operator.hpp"

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_multiple_reduce.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_2d_multiple_reduction_multiblock.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_multiple_buffer_value.hpp"

#include "ck/device_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumReduction,
          typename InDataType,
          typename AccDataType,
          typename OutDataTypePointerTuple,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple,
          InMemoryDataOperationEnum OutMemoryDataOperation,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceMultipleReduceMultiBlock : public DeviceMultipleReduce<NumReduction,
                                                                    InElementwiseOperationTuple,
                                                                    AccElementwiseOperationTuple>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (MThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert(NumReduction == OutDataTypePointerTuple::Size() &&
                      NumReduction == InElementwiseOperationTuple::Size() &&
                      NumReduction == AccElementwiseOperationTuple::Size(),
                  "All tuple should have the same size as the number of Reductions!");

    using IndexDataType = int32_t;

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t numSrcDim = Rank;
    static constexpr index_t numDstDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim = (NumInvariantDim == 0);

    // So far, only AtomicAdd is considered, other Atomic Operation like AtomicMax can be added
    // later
    static constexpr bool use_multiblock =
        (OutMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd);

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides,
                                    int blkGroupSize,
                                    int numBlockTileIteration)
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

        const int reduceSizePerBlock = K_BlockTileSize * numBlockTileIteration;
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

    static auto MakeDst1dDescriptor(const std::vector<index_t>& outLengths,
                                    const std::vector<index_t>& outStrides)
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

    static auto MakeDst1dDescriptorForBufferSet(const std::vector<index_t>& outLengths,
                                                const std::vector<index_t>& outStrides)
    {
        const auto tupleDstLengths = make_tuple_from_array(outLengths, Number<numDstDim>{});
        const auto tupleDstStrides = make_tuple_from_array(outStrides, Number<numDstDim>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out_grid_desc_m = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, numDstDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto length = out_grid_desc_m.GetLength(Number<0>{});

        const auto pad = math::integer_least_multiple(length, BlockSize) - length;

        auto out_grid_desc_m_padded =
            transform_tensor_descriptor(out_grid_desc_m,
                                        make_tuple(make_right_pad_transform(length, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (out_grid_desc_m_padded);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> inStrides,
                 const std::vector<index_t> outLengths,
                 const std::vector<index_t> outStrides,
                 const std::vector<int> reduceDims,
                 const std::array<float, NumReduction> alpha_values,
                 const std::array<float, NumReduction> beta_values,
                 const void* in_dev,
                 const std::array<void*, NumReduction> out_dev_buffers,
                 const InElementwiseOperationTuple in_elementwise_op_tuple,
                 const AccElementwiseOperationTuple acc_elementwise_op_tuple)
            : outLengths_{outLengths},
              outStrides_{outStrides},
              in_elementwise_op_tuple_{in_elementwise_op_tuple},
              acc_elementwise_op_tuple_{acc_elementwise_op_tuple}
        {
            inLengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inLengths, reduceDims);
            inStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inStrides, reduceDims);

            for(size_t i = 0; i < alpha_values.size(); i++)
            {
                alpha_values_(i) = type_convert<AccDataType>(alpha_values[i]);
                beta_values_(i)  = type_convert<AccDataType>(beta_values[i]);
            };

            in_dev_ = static_cast<const InDataType*>(in_dev);

            out_dev_buffers_ = generate_tuple(
                [&](auto iR) {
                    using OutDataTypePointer =
                        remove_cvref_t<decltype(OutDataTypePointerTuple{}[iR])>;
                    using OutDataType = remove_cvref_t<remove_pointer_t<OutDataTypePointer>>;
                    return static_cast<OutDataType*>(out_dev_buffers[iR]);
                },
                Number<NumReduction>{});

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, NumReduceDim>(inLengths_);

            if constexpr(NumInvariantDim == 0)
                invariant_lowest_length = 1;
            else
                invariant_lowest_length = inLengths_[NumInvariantDim - 1];

            reduce_lowest_length = inLengths_[Rank - 1];

            if constexpr(use_multiblock)
            {

                int iterations = 1;
                while(true)
                {
                    int testBlkGroupSize =
                        (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                        (K_BlockTileSize * iterations);

                    // we want the blkGroupSize be not more than 128
                    if(testBlkGroupSize <= 128)
                        break;

                    iterations++;
                };

                blkGroupSize = (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                               (K_BlockTileSize * iterations);

                numBlockTileIteration = iterations;
            }
            else
            {
                blkGroupSize = 1;
                numBlockTileIteration =
                    (reduce_total_length + K_BlockTileSize - 1) / K_BlockTileSize;
            };

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize * blkGroupSize;

            gridSize_pre =
                math::integer_least_multiple(invariant_total_length, BlockSize) / BlockSize;
        }

        std::vector<index_t> inLengths_;
        std::vector<index_t> inStrides_;
        std::vector<index_t> outLengths_;
        std::vector<index_t> outStrides_;

        Array<AccDataType, NumReduction> alpha_values_;
        Array<AccDataType, NumReduction> beta_values_;

        const InDataType* in_dev_;
        OutDataTypePointerTuple out_dev_buffers_;

        InElementwiseOperationTuple in_elementwise_op_tuple_;
        AccElementwiseOperationTuple acc_elementwise_op_tuple_;

        index_t invariant_lowest_length;
        index_t reduce_lowest_length;
        long_index_t invariant_total_length;
        long_index_t reduce_total_length;

        int blkGroupSize;
        int numBlockTileIteration;
        size_t gridSize;

        size_t gridSize_pre;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto in_grid_desc_m_k = DeviceMultipleReduceMultiBlock::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto out_grid_desc_m = DeviceMultipleReduceMultiBlock::MakeDst1dDescriptor(
                arg.outLengths_, arg.outStrides_);
            const auto out_grid_desc_m_2 =
                DeviceMultipleReduceMultiBlock::MakeDst1dDescriptorForBufferSet(arg.outLengths_,
                                                                                arg.outStrides_);

            using InGridDesc_M_K  = decltype(in_grid_desc_m_k);
            using OutGridDesc_M   = decltype(out_grid_desc_m);
            using OutGridDesc_M_2 = decltype(out_grid_desc_m_2);

            using GridwiseMultipleReduce =
                GridwiseMultipleReduction_mk_to_m_multiblock<NumReduction,
                                                             InDataType,
                                                             OutDataTypePointerTuple,
                                                             AccDataType,
                                                             IndexDataType,
                                                             InGridDesc_M_K,
                                                             OutGridDesc_M,
                                                             ReduceOperation,
                                                             InElementwiseOperationTuple,
                                                             AccElementwiseOperationTuple,
                                                             OutMemoryDataOperation,
                                                             PropagateNan,
                                                             BlockSize,
                                                             MThreadClusterSize,
                                                             KThreadClusterSize,
                                                             MThreadSliceSize,
                                                             KThreadSliceSize,
                                                             InSrcVectorDim,
                                                             InSrcVectorSize,
                                                             OutDstVectorSize>;

            const auto kernel_main =
                kernel_multiple_reduce_multiblock<GridwiseMultipleReduce,
                                                  NumReduction,
                                                  InDataType,
                                                  OutDataTypePointerTuple,
                                                  AccDataType,
                                                  IndexDataType,
                                                  InGridDesc_M_K,
                                                  OutGridDesc_M,
                                                  InElementwiseOperationTuple,
                                                  AccElementwiseOperationTuple>;

            float avg_time = 0;

            if constexpr(use_multiblock)
            {
                auto identity_values = generate_tuple(
                    [&](auto iR) {
                        using OutDataTypePointer =
                            remove_cvref_t<decltype(OutDataTypePointerTuple{}[iR])>;
                        using OutDataType = remove_cvref_t<remove_pointer_t<OutDataTypePointer>>;
                        return ck::reduce::GetIdentityValueForInMemoryDataOperation<OutDataType>(
                            OutMemoryDataOperation);
                    },
                    Number<NumReduction>{});

                using OutDataTypeTuple = decltype(identity_values);

                const auto kernel_pre = kernel_multiple_buffer_set_value<OutGridDesc_M_2,
                                                                         NumReduction,
                                                                         BlockSize,
                                                                         OutDataTypePointerTuple,
                                                                         OutDataTypeTuple>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_pre,
                                                   dim3(arg.gridSize_pre),
                                                   dim3(BlockSize),
                                                   0,
                                                   out_grid_desc_m_2,
                                                   arg.out_dev_buffers_,
                                                   identity_values);
            };

            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize),
                                               dim3(BlockSize),
                                               0,
                                               in_grid_desc_m_k,
                                               out_grid_desc_m,
                                               arg.in_elementwise_op_tuple_,
                                               arg.acc_elementwise_op_tuple_,
                                               arg.blkGroupSize,
                                               arg.numBlockTileIteration,
                                               arg.alpha_values_,
                                               arg.in_dev_,
                                               arg.beta_values_,
                                               arg.out_dev_buffers_);

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
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if constexpr(use_multiblock)
        {
            for(size_t i = 0; i < pArg->beta_values_.Size(); i++)
                if(pArg->beta_values_[i] != 0.0f)
                    return (false);

            bool out_types_match_atomic_op = true;

            static_for<0, NumReduction, 1>{}([&](auto iR) {
                using OutDataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[iR])>;
                using OutDataType        = remove_cvref_t<remove_pointer_t<OutDataTypePointer>>;

                out_types_match_atomic_op =
                    out_types_match_atomic_op && (std::is_same<OutDataType, float>::value ||
                                                  std::is_same<OutDataType, double>::value);
            });

            if(!out_types_match_atomic_op)
                return false;
        };

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

        // To improve
        if(pArg->invariant_lowest_length % OutDstVectorSize != 0)
            return (false);

        if constexpr(use_multiblock)
        {
            // blkGroupSize of 1 should be handled by Blockwise path using
            // InMemoryDataOperationEnum::Set
            if(pArg->blkGroupSize == 1)
                return (false);

            // This is very strong restriction, but needed to avoid some failure
            if(pArg->invariant_lowest_length % M_BlockTileSize != 0)
                return (false);
        }
        else
        {
            // cases with very small reduce_total_length should be handled by ThreadWise kernel
            if(pArg->reduce_total_length / KThreadSliceSize < 2)
                return (false);
        };

        return (true);
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<index_t> outLengths,
                        const std::vector<index_t> outStrides,
                        const std::vector<int> reduceDims,
                        const std::array<float, NumReduction> alpha_values,
                        const std::array<float, NumReduction> beta_values,
                        const void* in_dev,
                        const std::array<void*, NumReduction> out_dev_buffers,
                        const InElementwiseOperationTuple in_elementwise_op_tuple,
                        const AccElementwiseOperationTuple acc_elementwise_op_tuple) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          reduceDims,
                                          alpha_values,
                                          beta_values,
                                          in_dev,
                                          out_dev_buffers,
                                          in_elementwise_op_tuple,
                                          acc_elementwise_op_tuple);
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceMultipleReduceMultiBlockAtomicAdd<" << BlockSize << ",";
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
