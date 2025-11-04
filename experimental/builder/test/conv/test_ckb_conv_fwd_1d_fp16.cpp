#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 1D FP16 (channels-last) with DEFAULT specialization
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_1D_FP16_ChannelsFirst_scale)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 1,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout1D::NWGC_GKXC_NWGK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle FwdConvAlgorithm{
        .thread_block               = FwdThreadBlock_64x32x32,
        .gridwise_gemm              = FwdGemmParams_Xdl_2x1_per_wave,
        .block_transfer             = FwdBlockTransfer_4x16x1,
        .fwd_specialization         = ConvFwdSpecialization::DEFAULT,
        .gemm_specialization        = GemmSpecialization::MNKPadding,
        .num_gemm_k_prefetch_stages = 1,
        .num_groups_to_merge        = 2,
        .loop_scheduler             = LoopScheduler::DEFAULT};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>(
        {"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle", "64, 64, 32, 32", "Default"});
}

} // namespace
