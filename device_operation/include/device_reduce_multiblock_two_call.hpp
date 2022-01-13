#ifndef DEVICE_REDUCE_MULTIBLOCK_TWO_CALL_HPP
#define DEVICE_REDUCE_MULTIBLOCK_TWO_CALL_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_reduce.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_2d_reduction_multiblock_two_call.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt,
          int blockSize,
          int dim0_thread_cluster_size,
          int dim1_thread_cluster_size,
          int vectorDim,
          int dim0_thread_slice_size,
          int dim1_thread_slice_size>
struct DeviceReduceMultiBlockTwoCall : public DeviceReduce<inType,
                                                           compType,
                                                           outType,
                                                           rank,
                                                           toReduceDims,
                                                           reduceOp,
                                                           nanOpt,
                                                           indicesOpt>
{
    static_assert(rank <= 6, "Bigger rank size is not supported!");
    static_assert(blockSize == dim0_thread_cluster_size * dim1_thread_cluster_size,
                  "Invalid thread cluster size assignments!");
    static_assert(reduceOp != ReduceTensorOp_t::MUL && reduceOp != ReduceTensorOp_t::NORM1,
                  "MUL and NORM1 are not supported in Composable Kernel!");

    using invariantDims = decltype(get_invariantDims<rank, toReduceDims>());

    static constexpr index_t srcDims    = rank;
    static constexpr index_t dstDims    = (invariantDims::Size() == 0) ? 1 : invariantDims::Size();
    static constexpr bool reduceAllDims = (invariantDims::Size() == 0);

    static constexpr bool need_indices =
        (reduceOp == ReduceTensorOp_t::MIN || reduceOp == ReduceTensorOp_t::MAX ||
         reduceOp == ReduceTensorOp_t::AMAX) &&
        (indicesOpt != ReduceTensorIndices_t::NO_INDICES);

    static constexpr int dim0_tile_size = dim0_thread_cluster_size * dim0_thread_slice_size;
    static constexpr int dim1_tile_size = dim1_thread_cluster_size * dim1_thread_slice_size;

    static constexpr int vectorSize =
        (vectorDim == 0) ? math::gcd(dim0_thread_slice_size, max_vector_size_for_type<inType>())
                         : math::gcd(dim1_thread_slice_size, max_vector_size_for_type<inType>());

    size_t getWorkspaceSizeInBytes(const std::vector<int>& inLengths) override
    {
        size_t dim0_total_length;
        size_t dim1_total_length;

        std::tie(dim0_total_length, dim1_total_length) =
            get_2d_lengths<rank, toReduceDims>(inLengths);

        int iterations = 1;
        while(true)
        {
            int test_blkGroupSize = (dim1_total_length + (dim1_tile_size * iterations) - 1) /
                                    (dim1_tile_size * iterations);

            // we want the blkGroupSize be not more than 128
            if(test_blkGroupSize <= 128)
                break;

            iterations++;
        };

        int blkGroupSize =
            (dim1_total_length + (dim1_tile_size * iterations) - 1) / (dim1_tile_size * iterations);

        size_t workspace_size = dim0_total_length * blkGroupSize;

        size_t wsSizeInBytes =
            !need_indices ? workspace_size * sizeof(compType)
                          : workspace_size * (sizeof(compType) + sizeof(int)) + 64 + sizeof(int);

        return (wsSizeInBytes);
    };

    bool hasFurtherCall() override { return (true); };

    static auto MakeSrc2dDescriptor(const std::vector<int>& inLengths,
                                    const std::vector<int>& inStrides,
                                    size_t gridSize,
                                    int blkGroupSize)
    {
        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<srcDims>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<srcDims>{});

        const auto srcDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto src2dDesc = [&]() {
            if constexpr(reduceAllDims)
            {
                const auto one_dim_srcDesc = transform_tensor_descriptor(
                    srcDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, srcDims, 1>::type{}),
                    make_tuple(Sequence<0>{}));

                return transform_tensor_descriptor(one_dim_srcDesc,
                                                   make_tuple(make_unmerge_transform(make_tuple(
                                                       1, one_dim_srcDesc.GetLength(Number<0>{})))),
                                                   make_tuple(Sequence<0>{}),
                                                   make_tuple(Sequence<0, 1>{}));
            }
            else
            {
                const auto toReduceDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, toReduceDims{});
                const auto invariantDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, invariantDims{});

                return transform_tensor_descriptor(
                    srcDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(toReduceDimLengths)),
                    make_tuple(invariantDims{}, toReduceDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto invariantLen = src2dDesc.GetLength(Number<0>{});
        const auto toReduceLen  = src2dDesc.GetLength(Number<1>{});

        const int reduceSizePerBlock =
            (((toReduceLen + blkGroupSize - 1) / blkGroupSize + dim1_tile_size - 1) /
             dim1_tile_size) *
            dim1_tile_size;
        const auto srcPad1 = gridSize / blkGroupSize * dim0_tile_size - invariantLen;
        const auto srcPad2 = reduceSizePerBlock * blkGroupSize - toReduceLen;

        auto src2dDesc_2 =
            transform_tensor_descriptor(src2dDesc,
                                        make_tuple(make_pad_transform(invariantLen, 0, srcPad1),
                                                   make_pad_transform(toReduceLen, 0, srcPad2)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (src2dDesc_2);
    };

    static auto MakeWorkspace2dDescriptor(int invariantLen, int blkGroupSize)
    {
        auto ws2dDesc = make_naive_tensor_descriptor_packed(make_tuple(invariantLen, blkGroupSize));

        return (ws2dDesc);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<int> inLengths,
                 const std::vector<int> inStrides,
                 const std::vector<int> outLengths,
                 const std::vector<int> outStrides,
                 float alpha,
                 float beta,
                 const inType* in_dev,
                 outType* out_dev,
                 int* out_indices_dev,
                 compType* workspace_dev)
            : in_dev_{in_dev},
              out_dev_{out_dev},
              out_indices_dev_{out_indices_dev},
              workspace_dev_{workspace_dev}
        {
            inLengths_  = inLengths;
            inStrides_  = inStrides;
            outLengths_ = outLengths;
            outStrides_ = outStrides;
            alpha_      = static_cast<inType>(alpha);
            beta_       = static_cast<outType>(beta);

            std::tie(dim0_total_length, dim1_total_length) =
                get_2d_lengths<rank, toReduceDims>(inLengths);

            if constexpr(invariantDims::Size() == 0)
                dim0_lowest_length = 1;
            else
                dim0_lowest_length = inLengths[invariantDims::At(invariantDims::Size() - 1)];

            dim1_lowest_length = inLengths[toReduceDims::At(toReduceDims::Size() - 1)];

            int iterations = 1;
            while(true)
            {
                int test_blkGroupSize = (dim1_total_length + (dim1_tile_size * iterations) - 1) /
                                        (dim1_tile_size * iterations);

                // we want the blkGroupSize be not more than 128
                if(test_blkGroupSize <= 128)
                    break;

                iterations++;
            };

            blkGroupSize = (dim1_total_length + (dim1_tile_size * iterations) - 1) /
                           (dim1_tile_size * iterations);

            gridSize = (dim0_total_length + dim0_tile_size - 1) / dim0_tile_size * blkGroupSize;

            size_t ws_buf2_bytes_offset =
                ((dim0_total_length * blkGroupSize * sizeof(compType) + 63) / 64) * 64;

            if constexpr(need_indices)
                workspace_indices_dev_ = reinterpret_cast<int*>(
                    reinterpret_cast<char*>(workspace_dev_) + ws_buf2_bytes_offset);
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
        compType* workspace_dev_;
        int* workspace_indices_dev_;

        int dim0_lowest_length;
        int dim1_lowest_length;
        size_t dim0_total_length;
        size_t dim1_total_length;

        int blkGroupSize;
        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        template <int total_cluster_size, int dim0_cluster_size, int dim1_cluster_size>
        struct ClusterAssign
        {
            static constexpr int total_cluster_size_ = total_cluster_size;
            static constexpr int dim0_cluster_size_  = dim0_cluster_size;
            static constexpr int dim1_cluster_size_  = dim1_cluster_size;
        };

        using cluster_assign_instances = std::tuple<ClusterAssign<256, 128, 2>,
                                                    ClusterAssign<256, 64, 4>,
                                                    ClusterAssign<256, 32, 8>,
                                                    ClusterAssign<256, 16, 16>,
                                                    ClusterAssign<256, 8, 32>,
                                                    ClusterAssign<256, 4, 64>,
                                                    ClusterAssign<256, 2, 128>,
                                                    ClusterAssign<256, 1, 256>>;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto src2dDesc = DeviceReduceMultiBlockTwoCall::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.gridSize, arg.blkGroupSize);
            const auto ws2dDesc = DeviceReduceMultiBlockTwoCall::MakeWorkspace2dDescriptor(
                arg.dim0_total_length, arg.blkGroupSize);
            using src2dDescType = decltype(src2dDesc);
            using ws2dDescType  = decltype(ws2dDesc);

            using opReduce = typename reduce_binary_operator<compType, reduceOp>::opType;
            using preUnaryOpType =
                typename reduce_unary_operator<compType, reduceOp, true, false>::preUnaryOp;
            using posUnaryOpType =
                typename reduce_unary_operator<compType, reduceOp, true, false>::posUnaryOp;

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_multiblock_two_call<inType,
                                                              outType,
                                                              compType,
                                                              src2dDescType,
                                                              ws2dDescType,
                                                              opReduce,
                                                              preUnaryOpType,
                                                              posUnaryOpType,
                                                              nanOpt,
                                                              blockSize,
                                                              dim0_thread_cluster_size,
                                                              dim1_thread_cluster_size,
                                                              dim0_thread_slice_size,
                                                              dim1_thread_slice_size,
                                                              vectorDim,
                                                              vectorSize>;

            float avg_time = 0;

            const auto kernel = kernel_reduce_multiblock_two_call<gridwise_reduce,
                                                                  need_indices,
                                                                  inType,
                                                                  compType,
                                                                  src2dDescType,
                                                                  ws2dDescType,
                                                                  preUnaryOpType,
                                                                  posUnaryOpType>;

            avg_time =
                launch_and_time_kernel(kernel,
                                       nrepeat,
                                       dim3(arg.gridSize),
                                       dim3(blockSize),
                                       0,
                                       src2dDesc,
                                       ws2dDesc,
                                       preUnaryOpType{static_cast<int>(arg.dim1_total_length)},
                                       posUnaryOpType{static_cast<int>(arg.dim1_total_length)},
                                       arg.blkGroupSize,
                                       arg.alpha_,
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

        if constexpr(vectorDim == 0)
        {
            if constexpr(invariantDims::Size() == 0)
                return (false);

            if(pArg->inStrides_[invariantDims::At(invariantDims::Size() - 1)] != 1)
                return (false);
        }
        else
        {
            if(pArg->inStrides_[toReduceDims::At(toReduceDims::Size() - 1)] != 1)
                return (false);
        };

        if(pArg->dim0_lowest_length % dim0_thread_slice_size != 0)
            return (false);

        if(pArg->dim1_lowest_length % dim1_thread_slice_size != 0)
            return (false);

        // cases with small dim1_total_length should be handled by the BlockWise method
        if(pArg->dim1_total_length <= blockSize * dim1_thread_slice_size)
            return (false);

        return (true);
    };

    std::vector<int> getWorkspace2dLengths(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        return (std::vector<int>{static_cast<int>(pArg->dim0_total_length), pArg->blkGroupSize});
    };

    std::pair<size_t, size_t> getReduction2dLengths(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        return (std::make_pair(pArg->dim0_total_length, pArg->dim1_total_length));
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
                                                      void* workspace_dev) override
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
                                          static_cast<compType*>(workspace_dev));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        str << "DeviceReduceMultiBlockTwoCall<" << blockSize << ",";
        str << "Dim0_C" << dim0_thread_cluster_size << "_S" << dim0_thread_slice_size << ",";
        str << "Dim1_C" << dim1_thread_cluster_size << "_S" << dim1_thread_slice_size << ">";

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
