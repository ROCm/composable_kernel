#ifndef DEVICE_REDUCE_THREADWISE_HPP
#define DEVICE_REDUCE_THREADWISE_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_reduce.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_2d_reduction_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDims,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename OutElementwiseOperation,
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
struct DeviceReduceThreadWise : public DeviceReduce<InElementwiseOperation, OutElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert((BlockSize == MThreadClusterSize) && (KThreadClusterSize == 1),
                  "Threadwise can only be called with KThreadClusterSize be 1 !");

    using IndexDataType = int32_t;

    static constexpr bool BetaIsZero = NeedIndices;

    static constexpr index_t NumInvariantDims = Rank - NumReduceDims;
    using InvariantDims =
        typename conditional<NumInvariantDims == 0,
                             Sequence<>,
                             typename arithmetic_sequence_gen<0, NumInvariantDims, 1>::type>::type;
    using ReduceDims = typename arithmetic_sequence_gen<NumInvariantDims, Rank, 1>::type;

    static constexpr index_t srcDims    = Rank;
    static constexpr index_t dstDims    = (InvariantDims::Size() == 0) ? 1 : InvariantDims::Size();
    static constexpr bool reduceAllDims = (InvariantDims::Size() == 0);

    static constexpr int M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr int K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides)
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
                    make_tuple_from_array_and_index_seq(inLengths, ReduceDims{});
                const auto invariantDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, InvariantDims{});

                return transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(toReduceDimLengths)),
                    make_tuple(InvariantDims{}, ReduceDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto outerLen = in_grid_desc_m_k.GetLength(Number<0>{});
        const auto innerLen = in_grid_desc_m_k.GetLength(Number<1>{});

        const auto inPad_M = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;
        const auto inPad_K = math::integer_least_multiple(innerLen, K_BlockTileSize) - innerLen;

        auto in_grid_desc_m_k_padded =
            transform_tensor_descriptor(in_grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(outerLen, inPad_M),
                                                   make_right_pad_transform(innerLen, inPad_K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

    static auto MakeDst1dDescriptor(const std::vector<int>& outLengths,
                                    const std::vector<int>& outStrides)
    {
        const auto tupleDstLengths = make_tuple_from_array(outLengths, Number<dstDims>{});
        const auto tupleDstStrides = make_tuple_from_array(outStrides, Number<dstDims>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out_grid_desc_m = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto outerLen = out_grid_desc_m.GetLength(Number<0>{});

        const auto outPad = math::integer_least_multiple(outerLen, M_BlockTileSize) - outerLen;

        auto out_grid_desc_m_padded =
            transform_tensor_descriptor(out_grid_desc_m,
                                        make_tuple(make_right_pad_transform(outerLen, outPad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (out_grid_desc_m_padded);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<int>& inLengths,
                 const std::vector<int>& inStrides,
                 const std::vector<int>& outLengths,
                 const std::vector<int>& outStrides,
                 const std::vector<int>& toReduceDims,
                 float alpha,
                 float beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev,
                 IndexDataType* out_indices_dev,
                 AccDataType* workspace_dev,
                 const InElementwiseOperation& in_elementwise_op,
                 const OutElementwiseOperation& acc_elementwise_op)
            : in_dev_{in_dev}, out_dev_{out_dev}, out_indices_dev_{out_indices_dev}
        {
            (void)workspace_dev;

            outLengths_ = outLengths;
            outStrides_ = outStrides;

            std::tie(inLengths_, inStrides_) =
                shuffle_tensor_dimensions<Rank, NumReduceDims>(inLengths, inStrides, toReduceDims);

            in_elementwise_op_  = in_elementwise_op;
            acc_elementwise_op_ = acc_elementwise_op;

            alpha_ = static_cast<AccDataType>(alpha);
            beta_  = static_cast<OutDataType>(beta);

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, ReduceDims>(inLengths_);

            if constexpr(InvariantDims::Size() == 0)
                invariant_lowest_length = 1;
            else
                invariant_lowest_length = inLengths_[InvariantDims::At(InvariantDims::Size() - 1)];

            reduce_lowest_length = inLengths_[ReduceDims::At(ReduceDims::Size() - 1)];

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize;
        }

        std::vector<int> inLengths_;
        std::vector<int> inStrides_;
        std::vector<int> outLengths_;
        std::vector<int> outStrides_;

        AccDataType alpha_;
        OutDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;
        IndexDataType* out_indices_dev_;

        InElementwiseOperation in_elementwise_op_;
        OutElementwiseOperation acc_elementwise_op_;

        int invariant_lowest_length;
        int reduce_lowest_length;
        size_t invariant_total_length;
        size_t reduce_total_length;

        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto in_grid_desc_m_k =
                DeviceReduceThreadWise::MakeSrc2dDescriptor(arg.inLengths_, arg.inStrides_);
            const auto out_grid_desc_m =
                DeviceReduceThreadWise::MakeDst1dDescriptor(arg.outLengths_, arg.outStrides_);
            using InGridDesc_M_K = decltype(in_grid_desc_m_k);
            using OutGridDesc_M  = decltype(out_grid_desc_m);

            using GridwiseReduce = GridwiseReduction_mk_to_m_threadwise<InDataType,
                                                                        OutDataType,
                                                                        AccDataType,
                                                                        IndexDataType,
                                                                        InGridDesc_M_K,
                                                                        OutGridDesc_M,
                                                                        ReduceOperation,
                                                                        InElementwiseOperation,
                                                                        OutElementwiseOperation,
                                                                        PropagateNan,
                                                                        BetaIsZero,
                                                                        BlockSize,
                                                                        MThreadClusterSize,
                                                                        KThreadClusterSize,
                                                                        MThreadSliceSize,
                                                                        KThreadSliceSize,
                                                                        InSrcVectorDim,
                                                                        InSrcVectorSize,
                                                                        OutDstVectorSize>;

            float avg_time = 0;

            const auto kernel = kernel_reduce_threadwise<GridwiseReduce,
                                                         NeedIndices,
                                                         InDataType,
                                                         OutDataType,
                                                         AccDataType,
                                                         IndexDataType,
                                                         InGridDesc_M_K,
                                                         OutGridDesc_M,
                                                         InElementwiseOperation,
                                                         OutElementwiseOperation>;

            avg_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(arg.gridSize),
                                              dim3(BlockSize),
                                              0,
                                              in_grid_desc_m_k,
                                              out_grid_desc_m,
                                              arg.in_elementwise_op_,
                                              arg.acc_elementwise_op_,
                                              arg.alpha_,
                                              arg.in_dev_,
                                              arg.beta_,
                                              arg.out_dev_,
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

        if constexpr(InSrcVectorDim == 0)
        {
            if constexpr(InvariantDims::Size() == 0)
                return (false);

            if(pArg->inStrides_[InvariantDims::At(InvariantDims::Size() - 1)] != 1)
                return (false);

            if(pArg->invariant_lowest_length % InSrcVectorSize != 0)
                return (false);
        }
        else
        {
            if(pArg->inStrides_[ReduceDims::At(ReduceDims::Size() - 1)] != 1)
                return (false);

            if(pArg->reduce_lowest_length % InSrcVectorSize != 0)
                return (false);
        };

        // To improve
        if(pArg->invariant_lowest_length % OutDstVectorSize != 0)
            return (false);

        // TODO: remove this. Should return true, as long as this DeviceOP instance support this
        // case for bigger reduce_total_length size, we are supposed to use BlockWise method for
        // better performance
        if(pArg->reduce_total_length / KThreadSliceSize >= 32)
            return (false);

        return (true);
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<int>& inLengths,
                        const std::vector<int>& inStrides,
                        const std::vector<int>& outLengths,
                        const std::vector<int>& outStrides,
                        const std::vector<int>& toReduceDims,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        void* out_dev,
                        void* out_indices_dev,
                        void* workspace_dev,
                        const InElementwiseOperation& in_elementwise_op,
                        const OutElementwiseOperation& acc_elementwise_op) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          toReduceDims,
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
        str << "DeviceReducceThreadWise<" << BlockSize << ",";
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
