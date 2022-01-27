#include "device_reduce_instance_impl_common.hpp"
#include "device_reduce_multiblock_atomic_add.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <typename AccDataType, ReduceTensorOp_t ReduceOperation>
using deviceReducePtrType =
    DeviceReducePtr<typename reduce_unary_operator<AccDataType, ReduceOperation, true, true>::
                        InElementwiseOperation,
                    typename reduce_unary_operator<AccDataType, ReduceOperation, true, true>::
                        AccElementwiseOperation>;

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          typename InnerDims,
          ReduceTensorOp_t ReduceOpId,
          NanPropagation_t NanOpt,
          ReduceTensorIndices_t IndicesOpt>
void add_device_reduce_instance_multiblock_atomic_add(
    std::vector<deviceReducePtrType<AccDataType, ReduceOpId>>& device_op_instances)
{
    using ReduceOperation = typename reduce_binary_operator<AccDataType, ReduceOpId>::opType;
    using InElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation;
    using AccElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::
            AccElementwiseOperation;

    constexpr bool Indexable =
        (ReduceOpId == ReduceTensorOp_t::MIN || ReduceOpId == ReduceTensorOp_t::MAX ||
         ReduceOpId == ReduceTensorOp_t::AMAX);
    constexpr bool NeedIndices = Indexable && (IndicesOpt != ReduceTensorIndices_t::NO_INDICES);

    constexpr bool PropagateNan = (NanOpt == NanPropagation_t::NOT_PROPAGATE_NAN) ? false : true;

    static_assert(IndicesOpt == ReduceTensorIndices_t::NO_INDICES,
                  "AtomicAdd can only be used with reduction operations without indices!");

    constexpr bool op_acceptable =
        (ReduceOpId == ReduceTensorOp_t::ADD || ReduceOpId == ReduceTensorOp_t::MUL ||
         ReduceOpId == ReduceTensorOp_t::AVG || ReduceOpId == ReduceTensorOp_t::NORM1);

    constexpr bool out_type_acceptable =
        (std::is_same<OutDataType, float>::value || std::is_same<OutDataType, double>::value);

    if constexpr(!op_acceptable || !out_type_acceptable)
        return;
    else
    {
        static_for<0, std::tuple_size<reduce_configuration_1_instances>::value, 1>{}([&](auto i) {
            using cfg1 =
                remove_cvref_t<decltype(std::get<i.value>(reduce_configuration_1_instances{}))>;

            static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}(
                [&](auto j) {
                    using cfg2 = remove_cvref_t<decltype(
                        std::get<j.value>(reduce_configuration_2_instances{}))>;

                    using ReduceOpInstance =
                        DeviceReduceMultiBlockAtomicAdd<InDataType,
                                                        AccDataType,
                                                        OutDataType,
                                                        Rank,
                                                        InnerDims,
                                                        ReduceOperation,
                                                        InElementwiseOperation,
                                                        AccElementwiseOperation,
                                                        PropagateNan,
                                                        NeedIndices,
                                                        cfg1::blockSize_,
                                                        cfg1::dim0_thread_cluster_size_,
                                                        cfg1::dim1_thread_cluster_size_,
                                                        cfg2::vectorDim_,
                                                        cfg2::dim0_thread_slice_size_,
                                                        cfg2::dim1_thread_slice_size_>;

                    device_op_instances.push_back(
                        std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
                });
        });
    }
};

#define ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_TYPE(                                           \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...)                          \
    template void add_device_reduce_instance_multiblock_atomic_add<inT,                   \
                                                                   compT,                 \
                                                                   outT,                  \
                                                                   Rank,                  \
                                                                   Sequence<__VA_ARGS__>, \
                                                                   ReduceOpId,            \
                                                                   NanOpt,                \
                                                                   IndicesOpt>(           \
        std::vector<deviceReducePtrType<compT, ReduceOpId>> & device_op_instances)

#define ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(                                              \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...)                           \
    ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_TYPE(inT,                                            \
                                           compT,                                          \
                                           outT,                                           \
                                           static_cast<ReduceTensorOp_t>(ReduceOpId),      \
                                           static_cast<NanPropagation_t>(NanOpt),          \
                                           static_cast<ReduceTensorIndices_t>(IndicesOpt), \
                                           Rank,                                           \
                                           __VA_ARGS__)

// half, float, float
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 0, 0, 0, 4, 0);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 5, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 5, 0, 0, 2, 1);       //

// float, float, float
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, float, float, 0, 0, 0, 4, 0);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, float, float, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, float, float, 5, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, float, float, 5, 0, 0, 2, 1);       //

// float, double, float
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, double, float, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(float, double, float, 5, 0, 0, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
