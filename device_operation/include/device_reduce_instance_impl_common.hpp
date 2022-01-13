#ifndef DEVICE_REDUCE_INSTANCE_COMMON_HPP
#define DEVICE_REDUCE_INSTANCE_COMMON_HPP

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <int blockSize, int dim0_thread_cluster_size, int dim1_thread_cluster_size>
struct ReductionConfiguration_1
{
    static_assert(blockSize == dim0_thread_cluster_size * dim1_thread_cluster_size,
                  "Invalid Configuration!");

    static constexpr int blockSize_                = blockSize;
    static constexpr int dim0_thread_cluster_size_ = dim0_thread_cluster_size;
    static constexpr int dim1_thread_cluster_size_ = dim1_thread_cluster_size;
};

template <int vectorDim, int dim0_thread_slice_size, int dim1_thread_slice_size>
struct ReductionConfiguration_2
{
    static constexpr int vectorDim_              = vectorDim;
    static constexpr int dim0_thread_slice_size_ = dim0_thread_slice_size;
    static constexpr int dim1_thread_slice_size_ = dim1_thread_slice_size;
};

using reduce_configuration_1_instances = std::tuple<ReductionConfiguration_1<256, 128, 2>,
                                                    ReductionConfiguration_1<256, 64, 4>,
                                                    ReductionConfiguration_1<256, 32, 8>,
                                                    ReductionConfiguration_1<256, 16, 16>,
                                                    ReductionConfiguration_1<256, 8, 32>,
                                                    ReductionConfiguration_1<256, 4, 64>,
                                                    ReductionConfiguration_1<256, 2, 128>,
                                                    ReductionConfiguration_1<256, 1, 256>>;

#define QUICK_REDUCE_TEST 1

#ifdef QUICK_REDUCE_TEST
using reduce_configuration_2_instances = std::tuple<ReductionConfiguration_2<0, 2, 1>,
                                                    ReductionConfiguration_2<1, 1, 2>,

                                                    ReductionConfiguration_2<0, 3, 1>,
                                                    ReductionConfiguration_2<1, 1, 3>>;
#else
using reduce_configuration_2_instances = std::tuple<ReductionConfiguration_2<0, 8, 1>,
                                                    ReductionConfiguration_2<0, 4, 1>,
                                                    ReductionConfiguration_2<0, 2, 1>,

                                                    ReductionConfiguration_2<1, 1, 8>,
                                                    ReductionConfiguration_2<1, 1, 4>,
                                                    ReductionConfiguration_2<1, 1, 2>,

                                                    // special instances
                                                    ReductionConfiguration_2<0, 3, 1>,
                                                    ReductionConfiguration_2<0, 5, 1>,
                                                    ReductionConfiguration_2<0, 7, 1>,
                                                    ReductionConfiguration_2<0, 11, 1>,

                                                    ReductionConfiguration_2<0, 1, 3>,
                                                    ReductionConfiguration_2<0, 1, 5>,
                                                    ReductionConfiguration_2<0, 1, 7>,
                                                    ReductionConfiguration_2<0, 1, 11>>;
#endif

#define ADD_INST_BY_TYPE(key, inT, compT, outT, reduceOp, nanOpt, indicesOpt, rank, ...) \
    template void add_device_reduce_instance_##key<inT,                                  \
                                                   compT,                                \
                                                   outT,                                 \
                                                   rank,                                 \
                                                   Sequence<__VA_ARGS__>,                \
                                                   reduceOp,                             \
                                                   nanOpt,                               \
                                                   indicesOpt>(                          \
        std::vector<DeviceReducePtr<inT,                                                 \
                                    compT,                                               \
                                    outT,                                                \
                                    rank,                                                \
                                    Sequence<__VA_ARGS__>,                               \
                                    reduceOp,                                            \
                                    nanOpt,                                              \
                                    indicesOpt>> &                                       \
        device_op_instances)

#define ADD_INST_BY_ID(key, inT, compT, outT, reduceOp, nanOpt, indicesOpt, rank, ...) \
    ADD_INST_BY_TYPE(key,                                                              \
                     inT,                                                              \
                     compT,                                                            \
                     outT,                                                             \
                     static_cast<ReduceTensorOp_t>(reduceOp),                          \
                     static_cast<NanPropagation_t>(nanOpt),                            \
                     static_cast<ReduceTensorIndices_t>(indicesOpt),                   \
                     rank,                                                             \
                     __VA_ARGS__)

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
