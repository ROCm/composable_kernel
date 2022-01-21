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

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          typename opReduce,
          typename preUnaryOpType,
          typename posUnaryOpType,
          bool propagate_nan,
          bool need_indices,
          int blockSize,
          int dim0_thread_cluster_size,
          int dim1_thread_cluster_size,
          int vectorDim,
          int dim0_thread_slice_size,
          int dim1_thread_slice_size>
struct DeviceReduceBlockWiseSecondCall : public DeviceReduce<preUnaryOpType, posUnaryOpType>
{
    static_assert(rank <= 6, "Bigger rank size is not supported!");
    static_assert(blockSize == dim0_thread_cluster_size * dim1_thread_cluster_size,
                  "Invalid thread cluster size assignments!");

    static_assert(std::is_same<inType, compType>::value,
                  "inType and compType should be the same to use DEviceReduceBlockWiseSecondCall!");

    using invariantDims = decltype(get_invariantDims<rank, toReduceDims>());

    static constexpr index_t dstDims = (invariantDims::Size() == 0) ? 1 : invariantDims::Size();

    static constexpr int dim0_tile_size = dim0_thread_cluster_size * dim0_thread_slice_size;
    static constexpr int dim1_tile_size = dim1_thread_cluster_size * dim1_thread_slice_size;

    static constexpr int vectorSize =
        (vectorDim == 0) ? math::gcd(dim0_thread_slice_size, max_vector_size_for_type<inType>())
                         : math::gcd(dim1_thread_slice_size, max_vector_size_for_type<inType>());

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides)
    {
        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<2>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<2>{});

        const auto src2dDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto invariantLen = src2dDesc.GetLength(Number<0>{});
        const auto toReduceLen  = src2dDesc.GetLength(Number<1>{});

        const auto srcPad0 =
            math::integer_least_multiple(invariantLen, dim0_tile_size) - invariantLen;
        const auto srcPad1 =
            math::integer_least_multiple(toReduceLen, dim1_tile_size) - toReduceLen;

        auto src2dDesc_2 =
            transform_tensor_descriptor(src2dDesc,
                                        make_tuple(make_right_pad_transform(invariantLen, srcPad0),
                                                   make_right_pad_transform(toReduceLen, srcPad1)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (src2dDesc_2);
    };

    static auto MakeDst1dDescriptor(const std::vector<int>& outLengths,
                                    const std::vector<int>& outStrides)
    {
        const auto tupleDstLengths = make_tuple_from_array(outLengths, Number<dstDims>{});
        const auto tupleDstStrides = make_tuple_from_array(outStrides, Number<dstDims>{});

        auto dstDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto dst1dDesc = transform_tensor_descriptor(
            dstDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto invariantLen = dst1dDesc.GetLength(Number<0>{});

        const auto dstPad =
            math::integer_least_multiple(invariantLen, dim0_tile_size) - invariantLen;

        auto dst1dDesc_2 =
            transform_tensor_descriptor(dst1dDesc,
                                        make_tuple(make_right_pad_transform(invariantLen, dstPad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (dst1dDesc_2);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<int>& inLengths,
                 const std::vector<int>& inStrides,
                 const std::vector<int>& outLengths,
                 const std::vector<int>& outStrides,
                 float alpha,
                 float beta,
                 const inType* in_dev,
                 outType* out_dev,
                 int* out_indices_dev,
                 compType* workspace_dev,
                 const preUnaryOpType& preUnaryOp,
                 const posUnaryOpType& posUnaryOp)
            : in_dev_{in_dev}, out_dev_{out_dev}, out_indices_dev_{out_indices_dev}
        {
            inLengths_  = inLengths;
            inStrides_  = inStrides;
            outLengths_ = outLengths;
            outStrides_ = outStrides;

            preUnaryOp_ = preUnaryOp;
            posUnaryOp_ = posUnaryOp;

            alpha_ = static_cast<inType>(alpha);
            beta_  = static_cast<outType>(beta);

            dim0_total_length = inLengths[0];
            dim1_total_length = inLengths[1];

            dim0_lowest_length = inLengths[0];
            dim1_lowest_length = inLengths[1];

            gridSize =
                math::integer_least_multiple(dim0_total_length, dim0_tile_size) / dim0_tile_size;

            size_t ws_buf2_bytes_offset =
                ((dim0_total_length * dim1_total_length * sizeof(compType) + 63) / 64) * 64;

            if constexpr(need_indices)
                workspace_indices_dev_ = reinterpret_cast<int*>(
                    reinterpret_cast<char*>(workspace_dev) + ws_buf2_bytes_offset);
            else
                workspace_indices_dev_ = nullptr;
        }

        std::vector<int> inLengths_;
        std::vector<int> inStrides_;
        std::vector<int> outLengths_;
        std::vector<int> outStrides_;

        inType alpha_;
        outType beta_;

        const inType* in_dev_;
        outType* out_dev_;
        int* out_indices_dev_;
        int* workspace_indices_dev_;

        preUnaryOpType preUnaryOp_;
        posUnaryOpType posUnaryOp_;

        int dim0_lowest_length;
        int dim1_lowest_length;
        size_t dim0_total_length;
        size_t dim1_total_length;

        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto src2dDesc = DeviceReduceBlockWiseSecondCall::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_);
            const auto dst1dDesc = DeviceReduceBlockWiseSecondCall::MakeDst1dDescriptor(
                arg.outLengths_, arg.outStrides_);
            using src2dDescType = decltype(src2dDesc);
            using dst1dDescType = decltype(dst1dDesc);

            using gridwise_reduce = GridwiseReduction_xy_to_x_blockwise<inType,
                                                                        outType,
                                                                        compType,
                                                                        src2dDescType,
                                                                        dst1dDescType,
                                                                        opReduce,
                                                                        preUnaryOpType,
                                                                        posUnaryOpType,
                                                                        propagate_nan,
                                                                        blockSize,
                                                                        dim0_thread_cluster_size,
                                                                        dim1_thread_cluster_size,
                                                                        dim0_thread_slice_size,
                                                                        dim1_thread_slice_size,
                                                                        vectorDim,
                                                                        vectorSize>;

            float avg_time = 0;

            const auto kernel = kernel_reduce_blockwise_second_call<gridwise_reduce,
                                                                    need_indices,
                                                                    inType,
                                                                    outType,
                                                                    src2dDescType,
                                                                    dst1dDescType,
                                                                    preUnaryOpType,
                                                                    posUnaryOpType>;

            avg_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(arg.gridSize),
                                              dim3(blockSize),
                                              0,
                                              src2dDesc,
                                              dst1dDesc,
                                              arg.preUnaryOp_,
                                              arg.posUnaryOp_,
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

        if constexpr(vectorDim == 0)
            return (false);

        if(pArg->dim0_lowest_length % dim0_thread_slice_size != 0)
            return (false);

        if(pArg->dim1_lowest_length % dim1_thread_slice_size != 0)
            return (false);

        // cases with very small dim1_total_length should be handled by the ThreadWise method
        if(pArg->dim1_total_length / dim1_thread_slice_size < 2)
            return (false);

        return (true);
    };

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<int>& inLengths,
                                                      const std::vector<int>& inStrides,
                                                      const std::vector<int>& outLengths,
                                                      const std::vector<int>& outStrides,
                                                      float alpha,
                                                      float beta,
                                                      const void* in_dev,
                                                      void* out_dev,
                                                      void* out_indices_dev,
                                                      void* workspace_dev,
                                                      const preUnaryOpType& preUnaryOp,
                                                      const posUnaryOpType& posUnaryOp) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          alpha,
                                          beta,
                                          static_cast<const inType*>(in_dev),
                                          static_cast<outType*>(out_dev),
                                          static_cast<int*>(out_indices_dev),
                                          static_cast<compType*>(workspace_dev),
                                          preUnaryOp,
                                          posUnaryOp);
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        str << "DeviceReduceBlockWiseSecondCall<" << blockSize << ",";
        str << "Dim0_C" << dim0_thread_cluster_size << "_S" << dim0_thread_slice_size << ",";
        str << "Dim1_C" << dim1_thread_cluster_size << "_S" << dim1_thread_slice_size << ">";

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
