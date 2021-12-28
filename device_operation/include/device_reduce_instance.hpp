#ifndef DEVICE_REDUCE_INSTANTCE_HPP
#define DEVICE_REDUCE_INSTANTCE_HPP

#include "reduction_enums.hpp"
#include "device_reduce.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
extern void add_device_reduce_instance_threadwise(
    std::vector<DeviceReducePtr<inType,
                                compType,
                                outType,
                                rank,
                                toReduceDims,
                                reduceOp,
                                nanOpt,
                                indicesOpt>>& device_op_instances);

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
extern void
add_device_reduce_instance_blockwise(std::vector<DeviceReducePtr<inType,
                                                                 compType,
                                                                 outType,
                                                                 rank,
                                                                 toReduceDims,
                                                                 reduceOp,
                                                                 nanOpt,
                                                                 indicesOpt>>& device_op_instances);

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
extern void add_device_reduce_instance_blockwise_second_call(
    std::vector<DeviceReducePtr<inType,
                                compType,
                                outType,
                                rank,
                                toReduceDims,
                                reduceOp,
                                nanOpt,
                                indicesOpt>>& device_op_instances);

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
extern void add_device_reduce_instance_multiblock_atomic_add(
    std::vector<DeviceReducePtr<inType,
                                compType,
                                outType,
                                rank,
                                toReduceDims,
                                reduceOp,
                                nanOpt,
                                indicesOpt>>& device_op_instances);

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
extern void add_device_reduce_instance_multiblock_two_call(
    std::vector<DeviceReducePtr<inType,
                                compType,
                                outType,
                                rank,
                                toReduceDims,
                                reduceOp,
                                nanOpt,
                                indicesOpt>>& device_op_instances);

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

#include "device_reduce_instance_threadwise.hpp"
#include "device_reduce_instance_blockwise.hpp"
#include "device_reduce_instance_blockwise_second_call.hpp"
#include "device_reduce_instance_multiblock_atomic_add.hpp"
#include "device_reduce_instance_multiblock_two_call.hpp"

#endif
