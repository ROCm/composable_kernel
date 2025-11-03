#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_common.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_FP32_NGCHW_GKCYX)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::NGCHW_GKCYX_NGKHW,
        .data_type             = DataType::FP32,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock_128x128x32,
        .gridwise_gemm       = FwdGemmParams_Xdl_4x4_per_wave,
        .block_transfer      = FwdBlockTransfer_4x64_1,
        .fwd_specialization  = ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .block_gemm          = BlockGemmDesc_v4_intrawave};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>();
}

} // namespace
