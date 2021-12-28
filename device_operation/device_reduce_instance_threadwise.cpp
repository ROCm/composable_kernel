#include "device_reduce_instance_impl_common.hpp"
#include "device_reduce_threadwise.hpp"

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
void add_device_reduce_instance_threadwise(
    std::vector<DeviceReducePtr<inType,
                                compType,
                                outType,
                                rank,
                                toReduceDims,
                                reduceOp,
                                nanOpt,
                                indicesOpt>>& device_op_instances)
{
    using cfg1 = ReductionConfiguration_1<256, 256, 1>;

    static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}([&](auto j) {
        using cfg2 =
            remove_cvref_t<decltype(std::get<j.value>(reduce_configuration_2_instances{}))>;

        using ReduceOpInstance = DeviceReduceThreadWise<inType,
                                                        compType,
                                                        outType,
                                                        rank,
                                                        toReduceDims,
                                                        reduceOp,
                                                        nanOpt,
                                                        indicesOpt,
                                                        cfg1::blockSize_,
                                                        cfg1::dim0_thread_cluster_size_,
                                                        cfg1::dim1_thread_cluster_size_,
                                                        cfg2::dim0_max_vector_size_,
                                                        cfg2::dim1_max_vector_size_,
                                                        cfg2::dim0_thread_slice_size_,
                                                        cfg2::dim1_thread_slice_size_>;

        device_op_instances.push_back(std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
    });
};

// half, half, half
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_INST_BY_ID(threadwise, half_t, half_t, half_t, 4, 0, 1, 4, 0);       //

// half, float, half
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 0, 0, 0, 4, 0);
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 1, 0, 0, 4, 0, 1, 2); // for MUL
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 1, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 5, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 6, 0, 0, 4, 0, 1, 2); // for NORM1
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 6, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_INST_BY_ID(threadwise, half_t, float, half_t, 7, 0, 0, 4, 0);       //

// float, float, float
ADD_INST_BY_ID(threadwise, float, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_BY_ID(threadwise, float, float, float, 0, 0, 0, 4, 0);
ADD_INST_BY_ID(threadwise, float, float, float, 1, 0, 0, 4, 0, 1, 2); // for MUL
ADD_INST_BY_ID(threadwise, float, float, float, 1, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_BY_ID(threadwise, float, float, float, 5, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 6, 0, 0, 4, 0, 1, 2); // for NORM1
ADD_INST_BY_ID(threadwise, float, float, float, 6, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_INST_BY_ID(threadwise, float, float, float, 7, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_INST_BY_ID(threadwise, float, float, float, 2, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_INST_BY_ID(threadwise, float, float, float, 3, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_INST_BY_ID(threadwise, float, float, float, 4, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_INST_BY_ID(threadwise, float, float, float, 2, 0, 1, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_INST_BY_ID(threadwise, float, float, float, 3, 0, 1, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, float, float, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_INST_BY_ID(threadwise, float, float, float, 4, 0, 1, 4, 0);       //

// float, double, float
ADD_INST_BY_ID(threadwise, float, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_BY_ID(threadwise, float, double, float, 0, 0, 0, 4, 0);
ADD_INST_BY_ID(threadwise, float, double, float, 1, 0, 0, 4, 0, 1, 2); // for MUL
ADD_INST_BY_ID(threadwise, float, double, float, 1, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_BY_ID(threadwise, float, double, float, 5, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, double, float, 6, 0, 0, 4, 0, 1, 2); // for NORM1
ADD_INST_BY_ID(threadwise, float, double, float, 6, 0, 0, 4, 0);       //
ADD_INST_BY_ID(threadwise, float, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_INST_BY_ID(threadwise, float, double, float, 7, 0, 0, 4, 0);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
