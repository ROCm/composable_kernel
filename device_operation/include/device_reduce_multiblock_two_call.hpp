#ifndef DEVICE_REDUCE_MULTIBLOCK_TWO_CALL_HPP
#define DEVICE_REDUCE_MULTIBLOCK_TWO_CALL_HPP

#include <iostream>
#include "device_reduce.hpp"
#include "gridwise_2d_reduction_multiblock_two_call.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename inType, typename compType, typename outType, int rank, typename reduceDims, 
	  ReduceTensorOp_t reduceOp, NanPropagation_t nanOpt, ReduceTensorIndices_t indicesOpt,
	  int blockSize, dim0_thread_cluster_size, dim1_thread_cluster_size, 
	  int dim0_max_vector_size, dim1_max_vector_size, int dim0_thread_slice_size, int dim1_thread_slice_size>
struct DeviceReduceMultiBlockTwoCall : public DeviceReduce<inType, compType, outType, rank, reduceDims, reduceOp, nanOpt, indicesOpt>		      
{
     using invariantDims = decltype(get_invariantDims<rank, toReduceDims>());

     static constexpr index_t srcDims = rank;
     static constexpr index_t dstDims = (invariantDims::Size() == 0) 1 : invariantDims::Size();
     static constexpr bool reduceAllDims = (invariantDims::Size() == 0);

     static constexpr bool need_indices = (reduceOp == 2 || reduceOp == 3 || reduceOp == 4) && (indicesOpt == 1);

     static constexpr dim0_tile_size = dim0_thread_cluster_size * dim0_thread_slice_size;
     static constexpr dim1_tile_size = dim1_thread_cluster_size * dim1_thread_slice_size;

     size_t getWorkspaceSize(std::vector<int> inLengths) override
     {
          size_t dim0_total_length;
          size_t dim1_total_length;

          std::tie(dim0_total_length, dim1_total_length) = get_2d_lengths<rank, toReduceDims>(inLengths);

          int iteration = 1;
          while (true) {
              int test_blkGroupSize = (dim1_total_length + (dim1_thread_cluster_size * dim1_thread_slice_size * iterations) - 1) /
                                      (dim1_thread_cluster_size * dim1_thread_slice_size * iterations);

              // we want the blkGroupSize be not more than 128    
              if(test_blkGroupSize <= 128)
                 break;
          };

          blkGroupSize = (dim1_total_length + (dim1_thread_cluster_size * dim1_thread_slice_size * iterations) - 1) /
                            (dim1_thread_cluster_size * dim1_thread_slice_size * iterations);

          size_t workspace_size = dim0_total_length * blkGroupSize;

          size_t wsSizeInBytes = !need_indices ? workspace_size * sizeof(compType)
                               : workspace_size * (sizeof(compType) + sizeof(int)) + 64 + sizeof(int);

          return(0);
     };

     bool hasFurtherCall()
     {
	  return(true); 
     }; 

    virtual std::vector<int> getWorkspace2dLengths(const BaseArgument *argPtr) = 0;


     static const auto MakeSrc2dDescriptor(std::vector<int> & inLengths, std::vector<int> & inStrides, size_t gridSize, int blkGroupSize)
     {
         const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<srcDims>{});
         const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<srcDims>{});

         const auto srcDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

         const auto src2dDesc = [&]() {
               if constexpr(reduceAllDims) {
                   const auto one_dim_srcDesc = transform_tensor_descriptor(srcDesc,
                                           make_tuple(make_merge_transform(tupleSrcLengths)),
                                           make_tuple(typename arithmetic_sequence_gen<0, srcDims, 1>::type{}),
                                           make_tuple(Sequence<0>{}));

                   return transform_tensor_descriptor(one_dim_srcDesc,
                                        make_tuple(make_unmerge_transform(make_tuple(1, one_dim_srcDesc.GetLength(Number<0>{})))),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0, 1>{}));
               }
               else {
                   const auto toReduceDimLengths = make_tuple_from_array_and_index_seq(inLengths, toReduceDims{});
                   const auto invariantDimLengths = make_tuple_from_array_and_index_seq(inLengths, invariantDims{});

                   return transform_tensor_descriptor(srcDesc,
                                        make_tuple(make_merge_transform(invariantDimLengths),
                                                   make_merge_transform(toReduceDimLengths)),
                                        make_tuple(invariantDims{}, toReduceDims{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
               }
         }();

         const auto invariantLen = src2dDesc.GetLength(Number<0>{});
         const auto toReduceLen  = src2dDesc.GetLength(Number<1>{});

         const index_t reduceSizePerBlock = (((toReduceLen + blkGroupSize - 1) / blkGroupSize + dim1_tile_len - 1) / dim1_tile_len) * dim1_tile_len;
         const auto srcPad1 = gridSize / blkGroupSize * dim0_tile_len - invariantLen;
         const auto srcPad2 = reduceSizePerBlock * blkGroupSize - toReduceLen;

         auto src2dDesc_2 = transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_pad_transform(invariantLen, 0, srcPad1),
                                                       make_pad_transform(toReduceLen, 0, srcPad2)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

         return(src2dDesc_2);
     };
     
     static const auto MakeDst1dDescriptor(std::vector<int> & outLengths, std::vector<int> & outStrides, size_t gridSize, int blkGroupSize)
     {
         const auto tupleDstLengths = make_tuple_from_array(outLengths, Number<dstDims>{});
         const auto tupleDstStrides = make_tuple_from_array(outStrides, Number<dstDims>{});

         auto dstDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

         auto dst1dDesc = transform_tensor_descriptor(dstDesc,
                                         make_tuple(make_merge_transform(tupleDstLengths)),
                                         make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                         make_tuple(Sequence<0>{}));

         const auto invariantLen = dst1dDesc.GetLength(Number<0>{});

         const auto dstPad = gridSize / blkGroupSize * dim0_tile_len - invariantLen;

         auto dst1dDesc_2 = transform_tensor_descriptor(dst1dDesc,
                                            make_tuple(make_pad_transform(invariantLen, 0, dstPad)),
                                            make_tuple(Sequence<0>{}),
                                            make_tuple(Sequence<0>{}));
         return(dst1dDesc_2);
     };

     static const auto MakeWorkspace2dDescriptor(int invariantLen2, int toReduceLen2, size_t gridSize2, int dim0_thread_cluster_size2, int dim1_thread_cluster_size2)
     {
	 auto ws2dDesc = make_naive_tensor_descriptor_packed(make_tuple(invariantLen2, toReduceLen2));

         const auto srcPad1 = gridSize2 * dim0_thread_cluster_size2 - invariantLen2;
         const auto srcPad2 = ((toReduceLen2 + dim1_thread_cluster_size2 - 1) / dim1_thread_cluster_size2) * dim1_thread_cluster_size2 - toReduceLen2;

         auto ws2dDesc_2 = transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_pad_transform(invariantLen2, 0, srcPad1),
                                                       make_pad_transform(toReduceLen2, 0, srcPad2)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

         return(src2dDesc_2);
     };

     static int get_lower_closest_pow2(int length)
     {
         int res; 

         assert(length >= 2); 

	 while (true) {
            if ( (length & (length-1)) == 0 ) {
		 res = length; 
		 break; 
	    }

	    length = length & (length-1); 
	 };  

	 if (res > blockSize) 
	     res = blockSize; 

	 return(res); 
     }; 

     struct Argument : public BaseArgument
     {
         Argument(std::vector<int> inLengths, std::vector<int> inStrides, std::vector<int> outLengths, std::vector<int> outStrides,
                             float alpha, float beta, const inType *in_dev, outType *out_dev, int *out_indices_dev, compType *workspace_dev)
            : alpha_{alpha}, beta_{beta}, in_dev_{in_dev}, out_dev_{out_dev}, out_indices_dev_{out_indices_dev}, workspace_dev_{workspace_dev}
         {
             inLengths_ = inLengths;
             inStrides_ = inStrides;
             outLengths_ = outLengths;
             outStrides_ = outStrides;

             std::tie(dim0_total_length, dim1_total_length) = get_2d_lengths<rank, toReduceDims>(inLengths);

             if constexpr(invariantDims::Size() == 0)
                dim0_lowest_length = 1;
             else
                dim0_lowest_length = inLengths[invariantDims::At[invariantDims::Size()-1]];

             dim1_lowest_length = inLengths[toReduceDims::At[toReduceDims::Size()-1]];

             int iteration = 1;
             while (true) {
                    int test_blkGroupSize =
                        (dim1_total_length + (dim1_thread_cluster_size * dim1_thread_slice_size * iterations) - 1) /
                        (dim1_thread_cluster_size * dim1_thread_slice_size * iterations);

                    // we want the blkGroupSize be not more than 128    
                    if(test_blkGroupSize <= 128)
                        break;
             };

             blkGroupSize = (dim1_total_length + (dim1_tile_len * iterations) - 1) / (dim1_tile_len * iterations);

             gridSize = (dim0_total_length + dim0_tile_len-1) / dim0_tile_len * blkGroupSize;

             dim1_thread_cluster_size2 = get_lower_closet_pow2(blkGroupSize); 
	     dim0_thread_cluster_size2 = blockSize / dim1_thread_cluster_size2; 

             gridSize2 = (dim0_total_length + dim0_thread_cluster_size2-1) / dim0_thread_cluster_size2; 

             size_t ws_buf2_bytes_offset = ((dim0_total_length * blkGroupSize * sizeof(compType) + 63) / 64) * 64;

             if constexpr(need_indices) 
 	       	 workspace_indices_dev_ = static_cast<int*>( static_cast<char *>(workspace_ev_) + ws_buf2_bytes_offset ); 
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
	 
         int dim0_thread_cluster_size2; 
	 int dim1_thread_cluster_size2; 
	 size_t gridSize2; 
     };
     
     struct Invoker : public BaseInvoker
     {
	 template <int total_cluster_size, int dim0_cluster_size, int dim1_cluster_size> 
         struct ClusterAssign
	 {
	     static constexpr int total_cluster_size_ = total_cluster_size; 
	     static constexpr int dim0_cluster_size_ = dim0_cluster_size; 
	     static constexpr int dim1_cluster_size_ = dim1_cluster_size;  
	 }; 

         using cluster_assign_instances = std::tuple<
		ClusterAssign<256, 128, 2>, 
		ClusterAssign<256, 64, 4>,
		ClusterAssign<256, 32, 8>,
		ClusterAssign<256, 16, 16>,
		ClusterAssign<256, 8, 32>,
		ClusterAssign<256, 4, 64>,
		ClusterAssign<256, 2, 128>,
		ClusterAssign<256, 1, 256>
		>;

         float Run(const Argument& arg, int nrepeat = 1)
         {
             const auto src2dDesc = DeviceReduceMultiBlockTwoCall::MakeSrc2dDescriptor(arg.inLengths_, arg.inStrides_, arg.gridSize);
             const auto dst1dDesc = DeviceReduceMultiBlockTwoCall::MakeDst1dDescriptor(arg.outLengths_, arg.outStrides_, arg.gridSize);
             const auto ws2dDesc = DeviceReduceMultiBlockTwoCall::MakeWorkspace2dDescriptor(arg.dim0_total_length, arg.blkGroupSize, arg.gridSize2, arg.dim0_thread_cluster_size2, arg.dim1_thread_cluster_size2); 
             using src2dDescType = decltype(src2dDesc);
             using dst1dDescType = decltype(dst1dDesc);
             using ws2dDescType = decltype(ws2dDesc); 

             using gridwise_reduce = GridwiseReduction_xy_to_x_multiblock_two_call<inType, outType, compType, src2dDescType, ws2dDescType,
                                                 reduceOp, nanOpt, indicesOpt, blockSize, dim0_thread_cluster_size, dim1_thread_cluster_size,
                                                 dim0_thread_slice_size, dim1_thread_slice_size, dim0_max_vector_size, dim1_max_vector_size, true, true>

             constexpr int RunId = need_indices ? 2 : 1;

             float avg_time = 0;

	     const auto kernel = kernel_reduce_multiblock_two_call<gridwise_reduce, RunId, inType, outType, compType, src2dDescType, dst1dDescType>;

             ave_time = launch_and_time_kernel(kernel, nrepeat,
                                                  dim3(arg.gridSize),
                                                  dim3(blockSize),
                                                  0,
                                                  src2dDesc,
                                                  ws2dDesc,
                                                  static_cast<int>(arg.dim1_total_length),
                                                  arg.blkGroupSize,
                                                  arg.alpha_,
                                                  arg.in_dev_,
                                                  arg.beta_,
                                                  arg.workspace_dev_,
                                                  arg.workspace_indices_dev_);

             float avg_time2 = 0; 

	     ck:static_for<0, std::tuple_size<cluster_assign_instances>::value, 1>{}( [&](auto i) {
                   using cluster_assign = decltype( std::get<i.value>(cluster_assign_instances{}) );

                   if (cluster_assign::total_cluster_size == blockSize && cluster_assign::dim0_cluster_size == arg.dim0_thread_cluster_size2 && cluster_assign::dim1_cluster_size == arg.dim1_thread_cluster_size2) {
                        using gridwise_reduce2 = GridwiseReduction_xy_to_x_threadwise<inType, outType, compType, ws2dDescType, dst1dDescType,
                                                 reduceOp, nanOpt, indicesOpt, blockSize, cluster_assign::dim0_cluster_size, cluster_assign::dim1_cluster_size,
                                                 1, 1, 1, 1, false, true>; 

		        constexpr int RunId2 = need_indices ? 3 : 1; 

                        const auto kernel2 = kernel_reduce_blockwise<gridwise_reduce2, RunId2, inType, outType, src2dDescType, dst1dDescType>; 

                        ave_time2 = launch_and_time_kernel(kernel2, nrepeat,
                                                  dim3(arg.gridSize2),
                                                  dim3(blockSize),
                                                  0,
                                                  ws2dDesc,
                                                  dst1dDesc,
                                                  static_cast<int>(arg.dim1_total_length),
                                                  arg.alpha_,
                                                  arg.workspace_dev_,
                                                  arg.beta_,
                                                  arg.out_dev_,
                                                  arg.workspace_indices_dev_,
                                                  arg.out_indices_dev_);
		   }; 
	     }); 

             return(ave_time+ave_time2);
         };

         float Run(const BaseArgument* p_arg, int nrepeat = 1) override
         {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
         };
     };

     bool IsSupportedArgument(const BaseArgument* p_arg) override
     {
         const Argument *pArg = dynamic_cast<const Argument*>(p_arg);

         if ( !support_AtomicAdd )
              return(false);

         if ( dim0_lowest_length % dim0_thread_slice_size != 0 )
              return(false);

         if ( dim1_lowest_length % dim1_thread_slice_size != 0 )
              return(false);

         // cases with small dim1_total_length should be handled by the BlockWise method
         if ( pArg->dim1_total_length <= blockSize*dim1_thread_slice_size )
              return(false);

         return(true);
     };     

     std::vector<int> getWorkspace2dLengths(const BaseArgument *p_arg) 
     {
         const Argument *pArg = dynamic_cast<const Argument*>(p_arg);

	 return( std::vector<int>{(int)pArg->dim0_total_length, pArg->blkGroupSize} ); 
     }; 

     std::unique_ptr<BaseArgument> MakeArgumentPointer(std::vector<int> inLengths, std::vector<int> inStrides, std::vector<int> outLengths, std::vector<int> outStrides,
                                                       float alpha, float beta, const void *in_dev, void *out_dev, void *out_indices_dev, void *workspace_dev)
     {
         return std::make_unique<Argument>(inLengths, inStrides, outLengths, outStrides, alpha, beta,
                         static_cast<const inType*>(in_dev), static_cast<outType*>(out_dev), static_cast<int*>(out_indices_dev), static_cast<inType*>(workspace_dev));
     };

     std::unique_ptr<BaseInvoker> MakeInvokerPointer()
     {
         return std::make_unique<Invoker>();
     };
};


} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
