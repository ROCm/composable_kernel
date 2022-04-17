#ifndef DEVICE_BNORM_FWD_INSTANCE_WITH_REDUCE_BLOCKWISE_HPP
#define DEVICE_BNORM_FWD_INSTANCE_WITH_REDUCE_BLOCKWISE_HPP

#include "device_bnorm_fwd_instance_impl_common.hpp"
#include "device_bnorm_fwd_nhwc_c_with_reduce_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_bnorm_fwd_instance {

template <typename InOutDataType, typename AccDataType>
void add_device_bnorm_fwd_with_reduce_blockwise_instance(
    std::vector<DeviceBatchNormFwdPtr>& device_op_instances)
{

    static_for<0, std::tuple_size<bnorm_fwd_configuration_1_instances>::value, 1>{}([&](auto i) {
        using cfg1 =
            remove_cvref_t<decltype(std::get<i.value>(bnorm_fwd_configuration_1_instances{}))>;

        static_for<0, std::tuple_size<bnorm_fwd_configuration_2_instances>::value, 1>{}([&](auto
                                                                                                j) {
            using cfg2 =
                remove_cvref_t<decltype(std::get<j.value>(bnorm_fwd_configuration_2_instances{}))>;

            // for NHWC layout + Spatial Batch-Norm mode, will expand the codes when more layouts
            // and modes are added
            if constexpr(cfg2::InSrcVectorDim_ == 0)
            {
                using BatchNormFwdOpInstance =
                    DeviceBatchNormFwd_Input_N_H_W_C_Output_C_With_Reduce_Blockwise<
                        InOutDataType,
                        AccDataType,
                        cfg1::BlockSize_,
                        cfg1::MThreadClusterSize_,
                        cfg1::KThreadClusterSize_,
                        cfg2::MThreadSliceSize_,
                        cfg2::KThreadSliceSize_,
                        cfg2::InSrcVectorSize_,
                        cfg2::OutDstVectorSize_>;

                device_op_instances.push_back(
                    std::make_unique<BatchNormFwdOpInstance>(BatchNormFwdOpInstance{}));
            };
        });
    });
};

#define ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST(InOutDataType, AccDataType)                       \
    template void add_device_bnorm_fwd_with_reduce_blockwise_instance<InOutDataType, AccDataType>( \
        std::vector<DeviceBatchNormFwdPtr> & device_op_instances)

#define ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST_REF(InOutDataType, AccDataType)         \
    extern template void                                                                 \
        add_device_bnorm_fwd_with_reduce_blockwise_instance<InOutDataType, AccDataType>( \
            std::vector<DeviceBatchNormFwdPtr> & device_op_instances)

} // namespace device_bnorm_fwd_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
