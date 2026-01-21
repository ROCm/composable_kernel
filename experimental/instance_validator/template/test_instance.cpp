// Test compilation of grouped conv forward instance with specific parameters
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using namespace ck::tensor_operation::device;
using namespace ck::tensor_operation::element_wise;

// Test instance with parameters:
// BlockSize={{BLOCK_SIZE}}, MPerBlock={{M_PER_BLOCK}}, NPerBlock={{N_PER_BLOCK}}, KPerBlock={{K_PER_BLOCK}}
// MPerXDL={{M_PER_XDL}}, NPerXDL={{N_PER_XDL}}, AK1={{AK1}}, BK1={{BK1}}
// MXdlPerWave={{M_XDL_PER_WAVE}}, NXdlPerWave={{N_XDL_PER_WAVE}}

using DeviceInstance = DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    2,                                           // NDimSpatial
    ck::tensor_layout::convolution::NHWGC,      // ALayout
    ck::tensor_layout::convolution::GKYXC,      // BLayout
    ck::Tuple<>,                                 // DsLayout
    ck::tensor_layout::convolution::NHWGK,      // ELayout
    ck::half_t,                                  // ADataType
    ck::half_t,                                  // BDataType
    float,                                       // AccDataType
    ck::half_t,                                  // CShuffleDataType
    ck::Tuple<>,                                 // DsDataType
    ck::half_t,                                  // EDataType
    PassThrough,                                 // AElementwiseOperation
    PassThrough,                                 // BElementwiseOperation
    PassThrough,                                 // CDEElementwiseOperation
    ConvolutionForwardSpecialization::Default,   // ConvForwardSpecialization
    GemmSpecialization::MNKPadding,             // GemmSpec
    1,                                           // NumGemmKPrefetchStage
    {{BLOCK_SIZE}},                              // BlockSize
    {{M_PER_BLOCK}},                             // MPerBlock
    {{N_PER_BLOCK}},                             // NPerBlock
    {{K_PER_BLOCK}},                             // KPerBlock
    {{AK1}},                                     // AK1
    {{BK1}},                                     // BK1
    {{M_PER_XDL}},                               // MPerXDL
    {{N_PER_XDL}},                               // NPerXDL
    {{M_XDL_PER_WAVE}},                          // MXdlPerWave
    {{N_XDL_PER_WAVE}},                          // NXdlPerWave
    ck::Sequence<{{A_BLOCK_TRANSFER_THREAD_CLUSTER}}>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
    ck::Sequence<{{A_BLOCK_TRANSFER_ARRANGE}}>,  // ABlockTransferThreadClusterArrangeOrder
    ck::Sequence<{{A_BLOCK_TRANSFER_SRC_ACCESS}}>, // ABlockTransferSrcAccessOrder
    {{A_BLOCK_TRANSFER_SRC_VECTOR_DIM}},         // ABlockTransferSrcVectorDim
    {{A_BLOCK_TRANSFER_SRC_SCALAR_PER_VECTOR}},  // ABlockTransferSrcScalarPerVector
    {{A_BLOCK_TRANSFER_DST_SCALAR_PER_VECTOR}},  // ABlockTransferDstScalarPerVector_AK1
    1,                                           // ABlockLdsExtraM
    ck::Sequence<{{B_BLOCK_TRANSFER_THREAD_CLUSTER}}>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
    ck::Sequence<{{B_BLOCK_TRANSFER_ARRANGE}}>,  // BBlockTransferThreadClusterArrangeOrder
    ck::Sequence<{{B_BLOCK_TRANSFER_SRC_ACCESS}}>, // BBlockTransferSrcAccessOrder
    {{B_BLOCK_TRANSFER_SRC_VECTOR_DIM}},         // BBlockTransferSrcVectorDim
    {{B_BLOCK_TRANSFER_SRC_SCALAR_PER_VECTOR}},  // BBlockTransferSrcScalarPerVector
    {{B_BLOCK_TRANSFER_DST_SCALAR_PER_VECTOR}},  // BBlockTransferDstScalarPerVector_BK1
    1,                                           // BBlockLdsExtraN
    {{C_SHUFFLE_M_XDL_PER_WAVE_PER_SHUFFLE}},    // CShuffleMXdlPerWavePerShuffle
    {{C_SHUFFLE_N_XDL_PER_WAVE_PER_SHUFFLE}},    // CShuffleNXdlPerWavePerShuffle
    ck::Sequence<{{CDE_BLOCK_TRANSFER_CLUSTER}}>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    {{CDE_BLOCK_TRANSFER_SCALAR_PER_VECTOR}}     // CDEBlockTransferScalarPerVector_NPerBlock
>;

int main()
{
    // Create an instance get the type string to ensure all compile-time checks are done. 
    auto instance = DeviceInstance{};
    const auto type_string = instance.GetTypeString();
    std::cout << type_string << std::endl;
    return 0;
}
