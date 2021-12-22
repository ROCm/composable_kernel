#include <stdlib.h>
#include "config.hpp"
#include "device_reduce.hpp"
#include "device_reduce_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <int blockSize, int dim0_thread_cluster_length, dim1_thread_cluster_length>
struct ReductionConfiguration_1
{
    static_assert(blockSize == dim0_cluster_length * dim1_cluster_length, "Invalid Configuration!");

    static constexpr int blockSize_ = blockSize;
    static constexpr int dim0_thread_cluster_length_ = dim0_thread_cluster_length;
    static constexpr int dim1_thread_cluster_length_ = dim1_thread_cluster_length;
};

template<int dim0_max_vector_size, int dim1_max_vector_size, int dim0_thread_slice_length, int dim1_thread_slice_length>
struct ReductionConfiguration_2
{
    static constexpr int dim0_max_vector_size_ = dim0_max_vector_size;
    static constexpr int dim1_max_vector_size_ = dim1_max_vector_size;
    static constexpr int dim0_thread_slice_length_ = dim0_thread_slice_length;
    static constexpr int dim1_thread_slice_length_ = dim1_thread_slice_length;
};

using reduce_configuration_1_instances = std::tuple<
    ReductionConfiguration_1<256, 128, 2>,
    ReductionConfiguration_1<256, 64, 4>,
    ReductionConfiguration_1<256, 32, 8>,
    ReductionConfiguration_1<256, 16, 16>,
    ReductionConfiguration_1<256, 8, 32>,
    ReductionConfiguration_1<256, 4, 64>,
    ReductionConfiguration_1<256, 2, 128>,
    ReductionConfiguration_1<256, 1, 256>,
    >;

using reduce_configuration_2_instances = std::tuple<
    ReductionConfiguration_2<8, 1, 8, 1>,
    ReductionConfiguration_2<8, 1, 4, 1>,
    ReductionConfiguration_2<8, 1, 2, 1>,
    ReductionConfiguration_2<8, 1, 1, 1>,

    ReductionConfiguration_2<4, 1, 4, 1>,
    ReductionConfiguration_2<4, 1, 2, 1>,
    ReductionConfiguration_2<4, 1, 1, 1>,

    ReductionConfiguration_2<2, 1, 2, 1>,
    ReductionConfiguration_2<2, 1, 1, 1>,

    ReductionConfiguration_2<1, 1, 1, 1>,

    ReductionConfiguration_2<1, 2, 1, 2>,
    ReductionConfiguration_2<1, 2, 1, 1>,

    ReductionConfiguration_2<1, 4, 1, 4>,
    ReductionConfiguration_2<1, 4, 1, 2>,
    ReductionConfiguration_2<1, 4, 1, 1>,

    ReductionConfiguration_2<1, 8, 1, 8>,
    ReductionConfiguration_2<1, 8, 1, 4>,
    ReductionConfiguration_2<1, 8, 1, 2>,
    ReductionConfiguration_2<1, 8, 1, 1>,

    // special instances
    ReductionConfiguration_2<1, 1, 3, 1>,
    ReductionConfiguration_2<1, 1, 5, 1>,
    ReductionConfiguration_2<1, 1, 7, 1>,
    ReductionConfiguration_2<1, 1, 11, 1>,

    ReductionConfiguration_2<1, 1, 1, 3>,
    ReductionConfiguration_2<1, 1, 1, 5>,
    ReductionConfiguration_2<1, 1, 1, 7>,
    ReductionConfiguration_2<1, 1, 1, 11>,
    >;

template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
void add_device_reduce_instance_threadwise( std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, nanOpt, indicesOpt>>& device_op_instances)
{
     using cfg1 = ReductionConfiguration_1<256, 256, 1>; 

     ck:static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}( [&](auto j) {
          using cfg2 = decltype( std::get<j.value>( reduce_configuration_2_instances{}) );

          using ReduceOpInstance = DeviceReduceMultiThread<inType, compType, outType, rank, toReduceDims, reduceOp, nanOpt, indicesOpt,
                                        cfg1::blockSize, cfg1::dim0_thread_cluster_length_, cfg1::dim1_thread_cluster_length_,
                                        cfg2::dim0_max_vector_size, cfg2::dim1_max_vector_size, cfg2::dim0_thread_slice_length, cfg2::dim1_thread_slice_length>;

           device_op_instances.push_back( std::make_unique<ReduceOpInstance>(ReduceOpInstance{});
    });
}

template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
void add_device_reduce_instance_blockwise( std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, nanOpt, indicesOpt>>& device_op_instances)
{
    ck:static_for<0, std::tuple_size<reduce_configuration_1_instances>::value, 1>{}( [&](auto i) {
          using cfg1 = decltype( std::get<i.value>(reduce_configuration_1_instances{}) );

          ck:static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}( [&](auto j) {
              using cfg2 = decltype( std::get<j.value>( reduce_configuration_2_instances{}) );

              using ReduceOpInstance = DeviceReduceMultiThread<inType, compType, outType, rank, toReduceDims, reduceOp, nanOpt, indicesOpt, 
	                                    cfg1::blockSize, cfg1::dim0_thread_cluster_length_, cfg1::dim1_thread_cluster_length_, 
					    cfg2::dim0_max_vector_size, cfg2::dim1_max_vector_size, cfg2::dim0_thread_slice_length, cfg2::dim1_thread_slice_length>; 

              device_op_instances.push_back( std::make_unique<ReduceOpInstance>(ReduceOpInstance{});
          });
    }); 	  
}

template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
void add_device_reduce_instance_multiblock_atomic_add( std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, nanOpt, indicesOpt>>& device_op_instances)
{
    constexpr bool need_indices = ((reduceOp == 2 || reduceOp == 3 || reduceOp == 4) && indicesOpt == 1); 
    constexpr bool have_atomicAdd = (std::is_same<outType, float>::value || std::is_same<outType, double>::value); 

    if constexpr(need_indices || !have_atomicAdd)
	return; 

    ck:static_for<0, std::tuple_size<reduce_configuration_1_instances>::value, 1>{}( [&](auto i) {
          using cfg1 = decltype( std::get<i.value>(reduce_configuration_1_instances{}) );

          ck:static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}( [&](auto j) {
              using cfg2 = decltype( std::get<j.value>( reduce_configuration_2_instances{}) );

              using ReduceOpInstance = DeviceReduceMultiBlockAtomicAdd<inType, compType, outType, rank, toReduceDims, reduceOp, nanOpt, indicesOpt,
                                            cfg1::blockSize, cfg1::dim0_thread_cluster_length_, cfg1::dim1_thread_cluster_length_,
                                            cfg2::dim0_max_vector_size, cfg2::dim1_max_vector_size, cfg2::dim0_thread_slice_length, cfg2::dim1_thread_slice_length>;

              device_op_instances.push_back( std::make_unique<ReduceOpInstance>(ReduceOpInstance{});
          });
    });
}

template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
void add_device_reduce_instance_multiblock_two_call( std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, nanOpt, indicesOpt>>& device_op_instances)
{
    ck:static_for<0, std::tuple_size<reduce_configuration_1_instances>::value, 1>{}( [&](auto i) {
          using cfg1 = decltype( std::get<i.value>(reduce_configuration_1_instances{}) );

          ck:static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}( [&](auto j) {
              using cfg2 = decltype( std::get<j.value>( reduce_configuration_2_instances{}) );

              using ReduceOpInstance = DeviceReduceMultiBlockTwoCall<inType, compType, outType, rank, toReduceDims, reduceOp, nanOpt, indicesOpt,
                                            cfg1::blockSize, cfg1::dim0_thread_cluster_length_, cfg1::dim1_thread_cluster_length_,
                                            cfg2::dim0_max_vector_size, cfg2::dim1_max_vector_size, cfg2::dim0_thread_slice_length, cfg2::dim1_thread_slice_length>;

              device_op_instances.push_back( std::make_unique<ReduceOpInstance>(ReduceOpInstance{});
          });
    });
}

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
