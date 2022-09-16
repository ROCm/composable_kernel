// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_2d_reduction_multiblock.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_2d_reduction_threadwise.hpp"

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
          bool OutputIndex,
          bool HaveIndexInputIfOutputIndex,
          index_t BlockSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceReduceThreadWise : public DeviceReduce<InElementwiseOperation, AccElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");

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

    static constexpr index_t M_BlockTileSize = BlockSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = 1 * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides)
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

        const auto inPad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto inPad_K =
            math::integer_least_multiple(reduceLength, K_BlockTileSize) - reduceLength;

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
                 OutDataType* out_dev,
                 IndexDataType* out_index_dev,
                 const InElementwiseOperation in_elementwise_op,
                 const AccElementwiseOperation acc_elementwise_op)
            : outLengths_{outLengths},
              outStrides_{outStrides},
              in_dev_{in_dev},
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

            numBlockTileIteration = (reduce_total_length + K_BlockTileSize - 1) / K_BlockTileSize;

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize;
        }

        std::vector<index_t> inLengths_;
        std::vector<index_t> inStrides_;
        std::vector<index_t> outLengths_;
        std::vector<index_t> outStrides_;

        AccDataType alpha_;
        AccDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;
        IndexDataType* out_index_dev_;

        InElementwiseOperation in_elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

        index_t invariant_lowest_length;
        index_t reduce_lowest_length;
        long_index_t invariant_total_length;
        long_index_t reduce_total_length;

        int numBlockTileIteration;
        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto in_grid_desc_m_k =
                DeviceReduceThreadWise::MakeSrc2dDescriptor(arg.inLengths_, arg.inStrides_);
            const auto out_grid_desc_m =
                DeviceReduceThreadWise::MakeDst1dDescriptor(arg.outLengths_, arg.outStrides_);
            using InGridDesc_M_K = decltype(in_grid_desc_m_k);
            using OutGridDesc_M  = decltype(out_grid_desc_m);

            float avg_time = 0;

            using GridwiseReduce =
                GridwiseReduction_mk_to_m_threadwise<InDataType,
                                                     OutDataType,
                                                     AccDataType,
                                                     IndexDataType,
                                                     InGridDesc_M_K,
                                                     OutGridDesc_M,
                                                     ReduceOperation,
                                                     InElementwiseOperation,
                                                     AccElementwiseOperation,
                                                     InMemoryDataOperationEnum::Set,
                                                     PropagateNan,
                                                     BlockSize,
                                                     MThreadSliceSize,
                                                     KThreadSliceSize,
                                                     InSrcVectorDim,
                                                     InSrcVectorSize,
                                                     OutDstVectorSize>;

            const auto kernel = kernel_reduce_threadwise<GridwiseReduce,
                                                         OutputIndex,
                                                         HaveIndexInput,
                                                         InDataType,
                                                         OutDataType,
                                                         AccDataType,
                                                         IndexDataType,
                                                         InGridDesc_M_K,
                                                         OutGridDesc_M,
                                                         InElementwiseOperation,
                                                         AccElementwiseOperation>;

            avg_time = launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(arg.gridSize),
                                              dim3(BlockSize),
                                              0,
                                              in_grid_desc_m_k,
                                              out_grid_desc_m,
                                              arg.in_elementwise_op_,
                                              arg.acc_elementwise_op_,
                                              arg.alpha_,
                                              arg.in_dev_,
                                              nullptr,
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

        // To improve
        if(pArg->invariant_lowest_length % OutDstVectorSize != 0)
            return (false);

        // cases with big reduce_total_length should be handled by Blockwise kernel
        if(pArg->reduce_total_length / KThreadSliceSize >= 32)
            return (false);

        return (true);
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
        (void)in_index_dev;

        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          reduceDims,
                                          alpha,
                                          beta,
                                          static_cast<const InDataType*>(in_dev),
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
        str << "DeviceReduceThreadWise<" << BlockSize << ",";
        str << "M_C" << BlockSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << 1 << "_S" << KThreadSliceSize << ",";
        str << "InSrcVectorDim_" << InSrcVectorDim << "_InSrcVectorSize_" << InSrcVectorSize << "_OutDstVectorSize_" << OutDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
