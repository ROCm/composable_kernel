#ifndef DEVICE_REDUCE_INSTANCE_COMMON_HPP
#define DEVICE_REDUCE_INSTANCE_COMMON_HPP

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <int BlockSize, int MThreadClusterSize, int KThreadClusterSize>
struct ReductionConfiguration_1
{
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize, "Invalid Configuration!");

    static constexpr int BlockSize_          = BlockSize;
    static constexpr int MThreadClusterSize_ = MThreadClusterSize;
    static constexpr int KThreadClusterSize_ = KThreadClusterSize;
};

template <int InVectorDim, int InVectorSize, int OutVectorSize, int MThreadSliceSize, int KThreadSliceSize>
struct ReductionConfiguration_2
{
    static constexpr int InVectorDim_       = InVectorDim;
    static constexpr int InVectorSize_      = InVectorSize;
    static constexpr int OutVectorSize_     = OutVectorSize; 
    static constexpr int MThreadSliceSize_ = MThreadSliceSize;
    static constexpr int KThreadSliceSize_ = KThreadSliceSize;
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
using reduce_configuration_2_instances = std::tuple<ReductionConfiguration_2<0, 2, 2, 2, 1>,
                                                    ReductionConfiguration_2<1, 2, 1, 1, 2>,
                                                    ReductionConfiguration_2<1, 2, 2, 1, 2>,
                                                    ReductionConfiguration_2<0, 1, 1, 3, 1>,
                                                    ReductionConfiguration_2<1, 1, 1, 1, 3>,
                                                    ReductionConfiguration_2<1, 1, 2, 1, 3>>;
#else
using reduce_configuration_2_instances = std::tuple<ReductionConfiguration_2<0, 4, 4, 8, 1>,
                                                    ReductionConfiguration_2<0, 4, 4, 4, 1>,
                                                    ReductionConfiguration_2<0, 2, 2, 2, 1>,

                                                    ReductionConfiguration_2<1, 4, 1, 1, 8>,
                                                    ReductionConfiguration_2<1, 4, 1, 1, 4>,
                                                    ReductionConfiguration_2<1, 2, 1, 1, 2>,

                                                    // special instances
                                                    ReductionConfiguration_2<0, 1, 1, 3, 1>,
                                                    ReductionConfiguration_2<0, 1, 1, 5, 1>,
                                                    ReductionConfiguration_2<0, 1, 1, 7, 1>,
                                                    ReductionConfiguration_2<0, 1, 1, 11, 1>,

                                                    ReductionConfiguration_2<0, 1, 1, 1, 3>,
                                                    ReductionConfiguration_2<0, 1, 1, 1, 5>,
                                                    ReductionConfiguration_2<0, 1, 1, 1, 7>,
                                                    ReductionConfiguration_2<0, 1, 1, 1, 11>>;
#endif

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
