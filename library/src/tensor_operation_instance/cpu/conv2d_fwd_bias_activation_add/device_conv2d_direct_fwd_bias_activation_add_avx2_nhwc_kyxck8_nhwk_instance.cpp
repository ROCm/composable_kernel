#include <stdlib.h>
#include <utility>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/cpu/device/convolution_forward_specialization_cpu.hpp"
#include "ck/tensor_operation/cpu/device/device_convnd_direct_fwd_bias_activation_add_avx2_nhwc_kyxck8_nhwk.hpp"
#include "ck/tensor_operation/cpu/element/element_wise_operation_cpu.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {
namespace device_conv2d_fwd_bias_activation_add_avx2_instance {

using InType                           = float;
using WeiType                          = float;
using OutType                          = float;
using AccType                          = float;
using InLayout                         = ck::tensor_layout::gemm::RowMajor;    // NHWC
using WeiLayout                        = ck::tensor_layout::gemm::ColumnMajor; // KYXCK8
static constexpr bool NonTemporalStore = false;

using PT         = ck::tensor_operation::cpu::element_wise::PassThrough;
using AddReluAdd = ck::tensor_operation::cpu::element_wise::AddReluAdd;
using AddRelu    = ck::tensor_operation::cpu::element_wise::AddRelu;
using Add        = ck::tensor_operation::cpu::element_wise::Add;
using AddAddRelu = ck::tensor_operation::cpu::element_wise::AddAddRelu;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::cpu::device::ConvolutionForwardSpecialization_t::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::cpu::device::ConvolutionForwardSpecialization_t::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::cpu::device::ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0;

static constexpr auto DefaultGemmKLoop =
    ck::tensor_operation::cpu::device::ConvolutionForwardGemmKSpecialization_t::DefaultGemmKLoop;
static constexpr auto GemmKLoopOverC =
    ck::tensor_operation::cpu::device::ConvolutionForwardGemmKSpecialization_t::NHWC_GemmKLoopOverC;

static constexpr auto LoopOver_MNK = ck::tensor_operation::cpu::device::LoopOver_MNK;
static constexpr auto LoopOver_MKN = ck::tensor_operation::cpu::device::LoopOver_MKN;

void add_device_conv2d_direct_fwd_bias_activation_add_avx2_nhwc_kyxck8_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddReluAdd>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DeviceConvNDDirectFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float, float, float, float, float, PT, PT, AddReluAdd, ConvFwdDefault, 2, 6, 16, false, false, false, true, true, false>({0, 0, 0, DefaultGemmKLoop, LoopOver_MKN}),
            DeviceConvNDDirectFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float, float, float, float, float, PT, PT, AddReluAdd, ConvFwdDefault, 2, 6, 16, false, false, false, true, true, false>({0, 0, 0, DefaultGemmKLoop, LoopOver_MNK}),
            DeviceConvNDDirectFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float, float, float, float, float, PT, PT, AddReluAdd, ConvFwdDefault, 2, 4, 24, false, false, false, true, true, false>({0, 0, 0, DefaultGemmKLoop, LoopOver_MKN}),
            DeviceConvNDDirectFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float, float, float, float, float, PT, PT, AddReluAdd, ConvFwdDefault, 2, 4, 24, false, false, false, true, true, false>({0, 0, 0, DefaultGemmKLoop, LoopOver_MNK})
            // clang-format on
            ));
}

} // namespace device_conv2d_fwd_bias_activation_add_avx2_instance
} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck
