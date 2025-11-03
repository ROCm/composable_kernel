#include "utils/ckb_conv_test_common.hpp"

namespace{

using namespace ck_tile::builder::test_utils;

// 2D BF16 NHWGC (channels-last) with Pipeline V1 and DEFAULT
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_ChannelsLast)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
        .data_type             = DataType::BF16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock_256x256x32,
        .gridwise_gemm       = FwdGemmParams_Xdl_4x4_per_wave,
        .block_transfer      = FwdBlockTransfer_4x64_1,
        .fwd_specialization  = ConvFwdSpecialization::DEFAULT,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .block_gemm          = BlockGemmDesc_v1_intrawave};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>();
}

// 2D BF16 NHWGC (channels-last) with Pipeline V5 and FILTER_3x3
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_NHWGC_Filter3x3)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
        .data_type             = DataType::BF16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock_256x256x32,
        .gridwise_gemm       = FwdGemmParams_Xdl_4x4_per_wave,
        .block_transfer      = FwdBlockTransfer_4x64_1,
        .fwd_specialization  = ConvFwdSpecialization::FILTER_3x3,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .block_gemm          = BlockGemmDesc_v5_intrawave};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>();
}

} // namespace
