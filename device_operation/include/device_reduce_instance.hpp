#ifndef DEVICE_CONV_INSTANTCE_HPP
#define DEVICE_CONV_INSTANTCE_HPP

#include "device_conv.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reducev_instance {

template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
void add_device_reduce_instance_multithread(std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, nanOpt, indicesOpt>>& device_op_instances); 

template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
void add_device_reduce_instance_multiblock_atomic_add(std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, nanOpt, indicesOpt>>& device_op_instances); 
	
template <typename inType, typename compType, typename outType,
         int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
void add_device_reduce_instance_multiblock_two_call(std::vector<DeviceReducePtr<inType, compType, outType, rank, toReduceDims, nanOpt, indicesOpt>>& device_op_instances); 

} // namespace device_conv_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
