#include <stdlib.h>
#include "convolution_forward_specialization_cpu.hpp"
#include "config.hpp"
#include "device_convnd_fwd_avx2_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation_cpu.hpp"
#include "device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {
namespace device_conv2d_fwd_avx2_instance {

using InType                           = float;
using WeiType                          = float;
using OutType                          = float;
using AccType                          = float;
using InLayout                         = ck::tensor_layout::gemm::RowMajor;    // NHWC
using WeiLayout                        = ck::tensor_layout::gemm::ColumnMajor; // KYXC
static constexpr bool NonTemporalStore = false;

using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;
using ThreadwiseGemmAvx2_MxN_4x24_Dispatch =
    ck::cpu::ThreadwiseGemmAvx2_MxN_4x24_Dispatch<InType,
                                                  WeiType,
                                                  OutType,
                                                  InLayout,
                                                  WeiLayout,
                                                  NonTemporalStore>;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::cpu::device::ConvolutionForwardSpecialization_t::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::cpu::device::ConvolutionForwardSpecialization_t::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::cpu::device::ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0;

using device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_f32_instances = std::tuple<
    //#################################################################|InDataType|WeiDataType|OutDataType|AccDataType|InElementwiseOp|WeiElementwiseOp|OutElementwiseOp|ConvForwardSp|NumDimSpatial|MPerBlock|NPerBlock|KPerBlock|ThreadwiseGemm_Dispatch
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        float,
        float,
        float,
        float,
        PassThrough,
        PassThrough,
        PassThrough,
        ConvFwdDefault,
        2,
        256,
        128,
        64,
        ThreadwiseGemmAvx2_MxN_4x24_Dispatch>,
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        float,
        float,
        float,
        float,
        PassThrough,
        PassThrough,
        PassThrough,
        ConvFwdDefault,
        2,
        512,
        256,
        128,
        ThreadwiseGemmAvx2_MxN_4x24_Dispatch>,
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        float,
        float,
        float,
        float,
        PassThrough,
        PassThrough,
        PassThrough,
        ConvFwdDefault,
        2,
        1024,
        144,
        128,
        ThreadwiseGemmAvx2_MxN_4x24_Dispatch>>;

void add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances)
{
    ck::tensor_operation::device::add_device_operation_instances(
        instances, device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_f32_instances{});
}

} // namespace device_conv2d_fwd_avx2_instance
} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck
