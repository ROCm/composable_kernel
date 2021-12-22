#include "device_reduce_instance_common.hpp"
#include "device_reduce_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, ReduceTensorOp_t reduceOp, NanPropagation_t nanOpt, ReduceTensorIndices_t indicesOpt> 
void add_device_reduce_instance_multiblock_atomic_add( std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, reduceOp, nanOpt, indicesOpt>>& device_op_instances)
{
    static_assert(indicesOpt == ReduceTensorIndices_t::NO_INDICES, "AtomicAdd can only be used with reduction operations without indices!"); 

    constexpr bool op_acceptable = (reduceOp == ReduceTensorOp_t::ADD || reduceOp == ReduceTensorOp_t::MUL || reduceOp == ReduceTensorOp_t::AVG ||
	                            reduceOp == ReduceTensorOp_t::NORM1 || reduceOp == ReduceTensorOp_t::NORM2); 
                                                   
    constexpr bool out_type_acceptable = (std::is_same<outType, float>::value || std::is_same<outType, double>::value); 

    if constexpr(!op_acceptable || !out_type_acceptable)
	return; 
    else {
        static_for<0, std::tuple_size<reduce_configuration_1_instances>::value, 1>{}( [&](auto i) {
             using cfg1 = remove_cvref_t< decltype( std::get<i.value>(reduce_configuration_1_instances{}) ) >;

             static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}( [&](auto j) {
                  using cfg2 = remove_cvref_t< decltype( std::get<j.value>( reduce_configuration_2_instances{}) ) >;

                  using ReduceOpInstance = DeviceReduceMultiBlockAtomicAdd<inType, compType, outType, rank, toReduceDims, reduceOp, nanOpt, indicesOpt,
                                            cfg1::blockSize_, cfg1::dim0_thread_cluster_size_, cfg1::dim1_thread_cluster_size_,
                                            cfg2::dim0_max_vector_size_, cfg2::dim1_max_vector_size_, cfg2::dim0_thread_slice_size_, cfg2::dim1_thread_slice_size_>;

                  device_op_instances.push_back( std::make_unique<ReduceOpInstance>(ReduceOpInstance{}) );
             });
        });
    }
};

// half, float, float 
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 0, 0, 0, 4, 0, 1, 2);         // for ADD
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 0, 0, 0, 4, 0);
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 1, 0, 0, 4, 0, 1, 2);         // for MUL
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 1, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 5, 0, 0, 4, 0, 1, 2);         // for AVG
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 5, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 6, 0, 0, 4, 0, 1, 2);         // for NORM1
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 6, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 7, 0, 0, 4, 0, 1, 2);         // for NORM2
ADD_INST_BY_ID(multiblock_atomic_add, half_t, float, float, 7, 0, 0, 4, 0);               //

// float, float, float
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 0, 0, 0, 4, 0, 1, 2);         // for ADD
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 0, 0, 0, 4, 0);
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 1, 0, 0, 4, 0, 1, 2);         // for MUL
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 1, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 5, 0, 0, 4, 0, 1, 2);         // for AVG
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 5, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 6, 0, 0, 4, 0, 1, 2);         // for NORM1
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 6, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 7, 0, 0, 4, 0, 1, 2);         // for NORM2
ADD_INST_BY_ID(multiblock_atomic_add, float, float, float, 7, 0, 0, 4, 0);               //

// float, double, float
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 0, 0, 0, 4, 0, 1, 2);         // for ADD
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 0, 0, 0, 4, 0);
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 1, 0, 0, 4, 0, 1, 2);         // for MUL
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 1, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 5, 0, 0, 4, 0, 1, 2);         // for AVG
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 5, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 6, 0, 0, 4, 0, 1, 2);         // for NORM1
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 6, 0, 0, 4, 0);               //
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 7, 0, 0, 4, 0, 1, 2);         // for NORM2
ADD_INST_BY_ID(multiblock_atomic_add, float, double, float, 7, 0, 0, 4, 0);               //

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
