#ifndef DEVICE_REDUCE_MULTIBLOCK_ATOMIC_ADD_HPP
#define DEVICE_REDUCE_MULTIBLOCK_ATOMIC_ADD_HPP

#include <iostream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_reduce.hpp"
#include "device_reduce_common.hpp"
#include "gridwise_2d_reduction_multiblock_atomic_add.hpp"

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
          int dim0_max_vector_size,
          int dim1_max_vector_size,
          int dim0_thread_slice_size,
          int dim1_thread_slice_size>
struct DeviceReduceMultiBlockAtomicAdd : public DeviceReduce<inType,
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

    using invariantDims = decltype(get_invariantDims<rank, toReduceDims>());

    static constexpr index_t srcDims    = rank;
    static constexpr index_t dstDims    = (invariantDims::Size() == 0) ? 1 : invariantDims::Size();
    static constexpr bool reduceAllDims = (invariantDims::Size() == 0);

    static constexpr bool need_indices =
        (reduceOp == ReduceTensorOp_t::MIN || reduceOp == ReduceTensorOp_t::MAX ||
         reduceOp == ReduceTensorOp_t::AMAX) &&
        (indicesOpt != ReduceTensorIndices_t::NO_INDICES);

    static constexpr bool support_AtomicAdd =
        std::is_same<outType, float>::value || std::is_same<outType, double>::value;

    static_assert(!need_indices && support_AtomicAdd,
                  "MultiBlockAtomicAdd method can only be used with non-indiced operation and when "
                  "having float/double output type!");

    static constexpr int dim0_tile_size = dim0_thread_cluster_size * dim0_thread_slice_size;
    static constexpr int dim1_tile_size = dim1_thread_cluster_size * dim1_thread_slice_size;

    size_t getWorkspaceSize(const std::vector<int>& inLengths) override
    {
        (void)inLengths;
        return (0);
    };

    void showConfiguration(std::ostream& os, const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        os << std::endl;

        os << "MultiBlockAtomicAdd config: "
           << "BlkGroupSize_" << pArg->blkGroupSize << "_B" << blockSize;
        os << "_Dim0_C" << dim0_thread_cluster_size << "_V" << dim0_max_vector_size << "_S"
           << dim0_thread_slice_size;
        os << "_Dim1_C" << dim1_thread_cluster_size << "_V" << dim1_max_vector_size << "_S"
           << dim1_thread_slice_size;

        os << std::endl;
    };

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

    static auto MakeDst1dDescriptor(const std::vector<int>& outLengths,
                                    const std::vector<int>& outStrides,
                                    size_t gridSize,
                                    int blkGroupSize)
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

        const auto dstPad = gridSize / blkGroupSize * dim0_tile_size - invariantLen;

        auto dst1dDesc_2 =
            transform_tensor_descriptor(dst1dDesc,
                                        make_tuple(make_pad_transform(invariantLen, 0, dstPad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (dst1dDesc_2);
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
            : in_dev_{in_dev}, out_dev_{out_dev}
        {
            (void)out_indices_dev;
            (void)workspace_dev;

            inLengths_  = inLengths;
            inStrides_  = inStrides;
            outLengths_ = outLengths;
            outStrides_ = outStrides;

            alpha_ = static_cast<inType>(alpha);
            beta_  = static_cast<outType>(beta);

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

            gridSize_pre = (dim0_total_length + blockSize - 1) / blockSize;
        }

        std::vector<int> inLengths_;
        std::vector<int> inStrides_;
        std::vector<int> outLengths_;
        std::vector<int> outStrides_;

        inType alpha_;
        outType beta_;

        const inType* in_dev_;
        outType* out_dev_;

        int dim0_lowest_length;
        int dim1_lowest_length;
        size_t dim0_total_length;
        size_t dim1_total_length;

        int blkGroupSize;
        size_t gridSize;

        size_t gridSize_pre;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto src2dDesc = DeviceReduceMultiBlockAtomicAdd::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.gridSize, arg.blkGroupSize);
            const auto dst1dDesc = DeviceReduceMultiBlockAtomicAdd::MakeDst1dDescriptor(
                arg.outLengths_, arg.outStrides_, arg.gridSize, arg.blkGroupSize);
            using src2dDescType = decltype(src2dDesc);
            using dst1dDescType = decltype(dst1dDesc);

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_multiblock_atomic_add<inType,
                                                                outType,
                                                                compType,
                                                                src2dDescType,
                                                                dst1dDescType,
                                                                reduceOp,
                                                                nanOpt,
                                                                blockSize,
                                                                dim0_thread_cluster_size,
                                                                dim1_thread_cluster_size,
                                                                dim0_thread_slice_size,
                                                                dim1_thread_slice_size,
                                                                dim0_max_vector_size,
                                                                dim1_max_vector_size>;

            float avg_time_pre  = 0;
            float avg_time_main = 0;

            const auto kernel_pre  = kernel_buffer_set_value<blockSize, outType, dst1dDescType>;
            const auto kernel_main = kernel_reduce_multiblock_atocmi_add<gridwise_reduce,
                                                                         inType,
                                                                         outType,
                                                                         src2dDescType,
                                                                         dst1dDescType>;

            avg_time_pre = launch_and_time_kernel(kernel_pre,
                                                  nrepeat,
                                                  dim3(arg.gridSize_pre),
                                                  dim3(blockSize),
                                                  0,
                                                  dst1dDesc,
                                                  arg.out_dev_,
                                                  static_cast<outType>(0));

            avg_time_main = launch_and_time_kernel(kernel_main,
                                                   nrepeat,
                                                   dim3(arg.gridSize),
                                                   dim3(blockSize),
                                                   0,
                                                   src2dDesc,
                                                   dst1dDesc,
                                                   static_cast<int>(arg.dim1_total_length),
                                                   arg.blkGroupSize,
                                                   arg.alpha_,
                                                   arg.in_dev_,
                                                   arg.out_dev_);

            return (avg_time_pre + avg_time_main);
        };

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(static_cast<float>(pArg->beta_) != 0.0f)
            return (false);

        if(pArg->dim0_lowest_length % dim0_thread_slice_size != 0)
            return (false);

        if(pArg->dim1_lowest_length % dim1_thread_slice_size != 0)
            return (false);

        // cases with small dim1_total_length should be handled by the BlockWise method
        if(pArg->dim1_total_length <= blockSize * dim1_thread_slice_size)
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
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
