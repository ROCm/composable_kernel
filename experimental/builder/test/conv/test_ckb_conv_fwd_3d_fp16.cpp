#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 3D FP16 NDHWGC (channels-last) with Pipeline V4 and FILTER_1X1_PAD0
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP16_NDHWGC_ChannelsLast)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 3,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock_128x128x32,
        .gridwise_gemm       = FwdGemmParams_Xdl_4x4_per_wave,
        .block_transfer      = FwdBlockTransfer_4x64_1,
        .fwd_specialization  = ConvFwdSpecialization::FILTER_1X1_PAD0,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .block_gemm          = BlockGemmDesc_v4_intrawave};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                      "256, 128, 128, 32",
                      "Filter1x1Pad0",
                      "BlkGemmPipelineScheduler: Intrawave",
                      "BlkGemmPipelineVersion: v4"});
}

} // namespace
