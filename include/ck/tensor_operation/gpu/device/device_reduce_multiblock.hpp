// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_2d_reduction_multiblock.hpp"
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
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          InMemoryDataOperationEnum OutMemoryDataOperation,
          bool PropagateNan,
          bool OutputIndex,
          bool HaveIndexInputIfOutputIndex,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceReduceMultiBlock : public DeviceReduce<InElementwiseOperation, AccElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (MThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    using IndexDataType = int32_t;

    static constexpr bool HaveIndexInput = OutputIndex && HaveIndexInputIfOutputIndex;

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t numSrcDim = Rank;
    static constexpr index_t numDstDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim = (NumInvariantDim == 0);

    // So far, only AtomicAdd is considered, other Atomic Operation like AtomicMax can be added
    // later
    static constexpr bool use_multiblock =
        (OutMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd);

    static_assert(ck::reduce::InMemoryDataOperatonSupportedOnDataType<OutMemoryDataOperation,
                                                                      OutDataType>::value,
                  "The OutDataType must support the specified OutMemoryDataOperation!");

    static_assert(!use_multiblock || (use_multiblock && !OutputIndex),
                  "MultiBlock reduction can only be used when outputing index is not required");

    static_assert(
        ReduceOperation::IsCompatibleInMemoryDataOperation(OutMemoryDataOperation),
        "The reduction accumulation operation must be compatible with the OutMemoryDataOperation!");

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
                 float alpha,
                 float beta,
                 const InDataType* in_dev,
                 const IndexDataType* in_index_dev,
                 OutDataType* out_dev,
                 IndexDataType* out_index_dev,
                 const InElementwiseOperation in_elementwise_op,
                 const AccElementwiseOperation acc_elementwise_op)
            : outLengths_{outLengths},
              outStrides_{outStrides},
              in_dev_{in_dev},
              in_index_dev_{in_index_dev},
              out_dev_{out_dev},
              out_index_dev_{out_index_dev},
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

        AccDataType alpha_;
        AccDataType beta_;

        const InDataType* in_dev_;
        const IndexDataType* in_index_dev_;
        OutDataType* out_dev_;
        IndexDataType* out_index_dev_;

        InElementwiseOperation in_elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

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
            const auto in_grid_desc_m_k = DeviceReduceMultiBlock::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto out_grid_desc_m =
                DeviceReduceMultiBlock::MakeDst1dDescriptor(arg.outLengths_, arg.outStrides_);
            const auto out_grid_desc_m_2 = DeviceReduceMultiBlock::MakeDst1dDescriptorForBufferSet(
                arg.outLengths_, arg.outStrides_);

            using InGridDesc_M_K  = decltype(in_grid_desc_m_k);
            using OutGridDesc_M   = decltype(out_grid_desc_m);
            using OutGridDesc_M_2 = decltype(out_grid_desc_m_2);

            using GridwiseReduce = GridwiseReduction_mk_to_m_multiblock<InDataType,
                                                                        OutDataType,
                                                                        AccDataType,
                                                                        IndexDataType,
                                                                        InGridDesc_M_K,
                                                                        OutGridDesc_M,
                                                                        ReduceOperation,
                                                                        InElementwiseOperation,
                                                                        AccElementwiseOperation,
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

            const auto kernel_main = kernel_reduce_multiblock<GridwiseReduce,
                                                              OutputIndex,
                                                              HaveIndexInput,
                                                              InDataType,
                                                              OutDataType,
                                                              AccDataType,
                                                              int32_t,
                                                              InGridDesc_M_K,
                                                              OutGridDesc_M,
                                                              InElementwiseOperation,
                                                              AccElementwiseOperation>;

            float avg_time = 0;

            if constexpr(use_multiblock)
            {
                const auto identityVal =
                    ck::reduce::GetIdentityValueForInMemoryDataOperation<OutDataType>(
                        OutMemoryDataOperation);

                const auto kernel_pre =
                    kernel_buffer_set_value<BlockSize, OutDataType, OutGridDesc_M_2>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_pre,
                                                   dim3(arg.gridSize_pre),
                                                   dim3(BlockSize),
                                                   0,
                                                   out_grid_desc_m_2,
                                                   arg.out_dev_,
                                                   identityVal);
            };

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

    static bool IsSupportedArgument(const Argument* pArg)
    {
        if constexpr(use_multiblock)
        {
            if(static_cast<float>(pArg->beta_) != 0.0f)
                return (false);
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
            // if(pArg->reduce_total_length / KThreadSliceSize < 2)
            //     return (false);
        };

        return (true);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(dynamic_cast<const Argument*>(p_arg));
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
                        const void* in_index_dev,
                        void* out_dev,
                        void* out_index_dev,
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
                                          static_cast<const IndexDataType*>(in_index_dev),
                                          static_cast<OutDataType*>(out_dev),
                                          static_cast<IndexDataType*>(out_index_dev),
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
        str << (OutMemoryDataOperation == InMemoryDataOperationEnum::Set? "DeviceReduceBlockWise<" : "DeviceReduceMultiBlock<") << BlockSize << ",";
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
