#include "device_reduce_instance_impl_common.hpp"
#include "device_reduce_multiblock_atomic_add.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <typename compType, ReduceTensorOp_t reduceOp>
using deviceReducePtrType =
    DeviceReducePtr<typename reduce_unary_operator<compType, reduceOp, true, true>::preUnaryOp,
                    typename reduce_unary_operator<compType, reduceOp, true, true>::posUnaryOp>;

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
void add_device_reduce_instance_multiblock_atomic_add(
    std::vector<deviceReducePtrType<compType, reduceOp>>& device_op_instances)
{
    using opReduce = typename reduce_binary_operator<compType, reduceOp>::opType;
    using preUnaryOpType =
        typename reduce_unary_operator<compType, reduceOp, true, true>::preUnaryOp;
    using posUnaryOpType =
        typename reduce_unary_operator<compType, reduceOp, true, true>::posUnaryOp;

    constexpr bool is_indexable =
        (reduceOp == ReduceTensorOp_t::MIN || reduceOp == ReduceTensorOp_t::MAX ||
         reduceOp == ReduceTensorOp_t::AMAX);
    constexpr bool need_indices = is_indexable && (indicesOpt != ReduceTensorIndices_t::NO_INDICES);

    static_assert(indicesOpt == ReduceTensorIndices_t::NO_INDICES,
                  "AtomicAdd can only be used with reduction operations without indices!");

    constexpr bool op_acceptable =
        (reduceOp == ReduceTensorOp_t::ADD || reduceOp == ReduceTensorOp_t::MUL ||
         reduceOp == ReduceTensorOp_t::AVG || reduceOp == ReduceTensorOp_t::NORM1);

    constexpr bool out_type_acceptable =
        (std::is_same<outType, float>::value || std::is_same<outType, double>::value);

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
                        DeviceReduceMultiBlockAtomicAdd<inType,
                                                        compType,
                                                        outType,
                                                        rank,
                                                        toReduceDims,
                                                        opReduce,
                                                        preUnaryOpType,
                                                        posUnaryOpType,
                                                        nanOpt,
                                                        need_indices,
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
    inT, compT, outT, reduceOp, nanOpt, indicesOpt, rank, ...)                            \
    template void add_device_reduce_instance_multiblock_atomic_add<inT,                   \
                                                                   compT,                 \
                                                                   outT,                  \
                                                                   rank,                  \
                                                                   Sequence<__VA_ARGS__>, \
                                                                   reduceOp,              \
                                                                   nanOpt,                \
                                                                   indicesOpt>(           \
        std::vector<deviceReducePtrType<compT, reduceOp>> & device_op_instances)

#define ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(                                              \
    inT, compT, outT, reduceOp, nanOpt, indicesOpt, rank, ...)                             \
    ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_TYPE(inT,                                            \
                                           compT,                                          \
                                           outT,                                           \
                                           static_cast<ReduceTensorOp_t>(reduceOp),        \
                                           static_cast<NanPropagation_t>(nanOpt),          \
                                           static_cast<ReduceTensorIndices_t>(indicesOpt), \
                                           rank,                                           \
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
