#ifndef DEVICE_BNORM_FWD_INSTANCE_IMPL_COMMON_HPP
#define DEVICE_BNORM_FWD_INSTANCE_IMPL_COMMON_HPP

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_bnorm_fwd_instance {

template <int BlockSize, int MThreadClusterSize, int KThreadClusterSize>
struct BatchNormFwdConfiguration_1
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
struct BatchNormFwdConfiguration_2
{
    static constexpr int InSrcVectorDim_   = InSrcVectorDim;
    static constexpr int InSrcVectorSize_  = InSrcVectorSize;
    static constexpr int OutDstVectorSize_ = OutDstVectorSize;
    static constexpr int MThreadSliceSize_ = MThreadSliceSize;
    static constexpr int KThreadSliceSize_ = KThreadSliceSize;
};

using bnorm_fwd_configuration_1_instances = std::tuple<
    // clang-format off
    // BlockSize | MThreadClusterSize | KThreadClusterSize
    BatchNormFwdConfiguration_1<256, 128, 2>,
    BatchNormFwdConfiguration_1<256, 64, 4>,
    BatchNormFwdConfiguration_1<256, 32, 8>,
    BatchNormFwdConfiguration_1<256, 16, 16>,
    BatchNormFwdConfiguration_1<256, 8, 32>,
    BatchNormFwdConfiguration_1<256, 4, 64>,
    BatchNormFwdConfiguration_1<256, 2, 128>,
    BatchNormFwdConfiguration_1<256, 1, 256>
    // clang-format on
    >;

#define QUICK_BNORM_FWD_TEST 1

#ifdef QUICK_BNORM_FWD_TEST
using bnorm_fwd_configuration_2_instances = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    BatchNormFwdConfiguration_2<0, 2, 2, 2, 1>,
    BatchNormFwdConfiguration_2<0, 1, 1, 2, 1>,
    BatchNormFwdConfiguration_2<1, 2, 1, 1, 2>,
    BatchNormFwdConfiguration_2<0, 1, 1, 3, 1>,
    BatchNormFwdConfiguration_2<1, 1, 1, 1, 3>
    // clang-format on
    >;
#else
using bnorm_fwd_configuration_2_instances = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    BatchNormFwdConfiguration_2<0, 4, 4, 8, 1>,
    BatchNormFwdConfiguration_2<0, 4, 4, 4, 1>,
    BatchNormFwdConfiguration_2<0, 2, 2, 2, 1>,

    BatchNormFwdConfiguration_2<1, 4, 1, 1, 8>,
    BatchNormFwdConfiguration_2<1, 4, 1, 1, 4>,
    BatchNormFwdConfiguration_2<1, 2, 1, 1, 2>,

    // special instances
    BatchNormFwdConfiguration_2<0, 1, 1, 3, 1>,
    BatchNormFwdConfiguration_2<0, 1, 1, 5, 1>,
    BatchNormFwdConfiguration_2<0, 1, 1, 7, 1>,
    BatchNormFwdConfiguration_2<0, 1, 1, 11, 1>,

    BatchNormFwdConfiguration_2<1, 1, 1, 1, 3>,
    BatchNormFwdConfiguration_2<1, 1, 1, 1, 5>,
    BatchNormFwdConfiguration_2<1, 1, 1, 1, 7>,
    BatchNormFwdConfiguration_2<1, 1, 1, 1, 11>
    // clang-format on
    >;
#endif

} // namespace device_bnorm_fwd_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
