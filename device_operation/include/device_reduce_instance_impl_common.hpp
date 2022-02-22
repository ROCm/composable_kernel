#ifndef DEVICE_REDUCE_INSTANCE_IMPL_COMMON_HPP
#define DEVICE_REDUCE_INSTANCE_IMPL_COMMON_HPP

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

template <int InSrcVectorDim,
          int InSrcVectorSize,
          int OutDstVectorSize,
          int MThreadSliceSize,
          int KThreadSliceSize>
struct ReductionConfiguration_2
{
    static constexpr int InSrcVectorDim_   = InSrcVectorDim;
    static constexpr int InSrcVectorSize_  = InSrcVectorSize;
    static constexpr int OutDstVectorSize_ = OutDstVectorSize;
    static constexpr int MThreadSliceSize_ = MThreadSliceSize;
    static constexpr int KThreadSliceSize_ = KThreadSliceSize;
};

using reduce_configuration_1_instances = std::tuple<
    // clang-format off
    // BlockSize | MThreadClusterSize | KThreadClusterSize
    ReductionConfiguration_1<256, 128, 2>,
    ReductionConfiguration_1<256, 64, 4>,
    ReductionConfiguration_1<256, 32, 8>,
    ReductionConfiguration_1<256, 16, 16>,
    ReductionConfiguration_1<256, 8, 32>,
    ReductionConfiguration_1<256, 4, 64>,
    ReductionConfiguration_1<256, 2, 128>,
    ReductionConfiguration_1<256, 1, 256>
    // clang-format on
    >;

#define QUICK_REDUCE_TEST 1

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
