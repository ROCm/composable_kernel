#ifndef DEVICE_REDUCE_BLOCKWISE_SECOND_CALL_HPP
#define DEVICE_REDUCE_BLOCKWISE_SECOND_CALL_HPP

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
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceReduceBlockWiseSecondCall
    : public DeviceReduce<InElementwiseOperation, AccElementwiseOperation>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static constexpr bool BetaIsZero = NeedIndices;

    static_assert(
        std::is_same<InDataType, AccDataType>::value,
        "InDataType and AccDataType should be the same to use DEviceReduceBlockWiseSecondCall!");

    using OuterDims = decltype(get_outer_dims<Rank, InnerDims>());

    static constexpr index_t dstDims = (OuterDims::Size() == 0) ? 1 : OuterDims::Size();

    static constexpr int M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr int K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides)
    {
        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<2>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<2>{});

        const auto in_grid_desc_m_k =
            make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

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
                 float alpha,
                 float beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev,
                 int32_t* out_indices_dev,
                 AccDataType* workspace_dev,
                 const InElementwiseOperation& in_elementwise_op,
                 const AccElementwiseOperation& acc_elementwise_op)
            : in_dev_{in_dev}, out_dev_{out_dev}, out_indices_dev_{out_indices_dev}
        {
            inLengths_  = inLengths;
            inStrides_  = inStrides;
            outLengths_ = outLengths;
            outStrides_ = outStrides;

            in_elementwise_op_  = in_elementwise_op;
            acc_elementwise_op_ = acc_elementwise_op;

            alpha_ = static_cast<AccDataType>(alpha);
            beta_  = static_cast<OutDataType>(beta);

            outer_total_length = inLengths[0];
            inner_total_length = inLengths[1];

            outer_lowest_length = inLengths[0];
            inner_lowest_length = inLengths[1];

            gridSize =
                math::integer_least_multiple(outer_total_length, M_BlockTileSize) / M_BlockTileSize;

            size_t ws_buf2_bytes_offset = math::integer_least_multiple(
                outer_total_length * inner_total_length * sizeof(AccDataType), 64);

            if constexpr(NeedIndices)
                workspace_indices_dev_ = reinterpret_cast<index_t*>(
                    reinterpret_cast<char*>(workspace_dev) + ws_buf2_bytes_offset);
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
        int32_t* out_indices_dev_;
        int32_t* workspace_indices_dev_;

        InElementwiseOperation in_elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

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
            const auto in_grid_desc_m_k = DeviceReduceBlockWiseSecondCall::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_);
            const auto out_grid_desc_m = DeviceReduceBlockWiseSecondCall::MakeDst1dDescriptor(
                arg.outLengths_, arg.outStrides_);
            using InGridDesc_M_K = decltype(in_grid_desc_m_k);
            using OutGridDesc_M  = decltype(out_grid_desc_m);

            using GridwiseReduce = GridwiseReduction_mk_to_m_blockwise<InDataType,
                                                                       OutDataType,
                                                                       AccDataType,
                                                                       int32_t,
                                                                       InGridDesc_M_K,
                                                                       OutGridDesc_M,
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
                                                                       InSrcVectorDim,
                                                                       InSrcVectorSize,
                                                                       OutDstVectorSize>;

            float avg_time = 0;

            const auto kernel = kernel_reduce_blockwise_second_call<GridwiseReduce,
                                                                    NeedIndices,
                                                                    InDataType,
                                                                    OutDataType,
                                                                    AccDataType,
                                                                    int32_t,
                                                                    InGridDesc_M_K,
                                                                    OutGridDesc_M,
                                                                    InElementwiseOperation,
                                                                    AccElementwiseOperation>;

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
                                              arg.workspace_indices_dev_,
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
            return (false);

        if(pArg->inner_lowest_length % InSrcVectorSize != 0)
            return (false);

        // To improve
        if(pArg->outer_lowest_length % OutDstVectorSize != 0)
            return (false);

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
                                          static_cast<int32_t*>(out_indices_dev),
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
        str << "DeviceReduceBlockWiseSecondCall<" << BlockSize << ",";
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
