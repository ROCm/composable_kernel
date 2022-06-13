#include <stdlib.h>
#include <utility>
#include "config.hpp"
#include "convolution_forward_specialization_cpu.hpp"
#include "device_convnd_fwd_avx2_nhwc_kyxck8_nhwk.hpp"
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
using WeiLayout                        = ck::tensor_layout::gemm::ColumnMajor; // KYXCK8
static constexpr bool NonTemporalStore = false;

using PT   = ck::tensor_operation::cpu::element_wise::PassThrough;
using Relu = ck::tensor_operation::cpu::element_wise::Relu;

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

// clang-format off
#define DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(a_elem_op, b_elem_op, c_elem_op, m_per_block, n_per_block, k_per_block, m_per_thread, n_per_thread, c_local_buf) \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, GemmKLoopOverC  , 2, m_per_thread, n_per_thread, true,  true,  c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MNK}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, GemmKLoopOverC  , 2, m_per_thread, n_per_thread, true,  true,  c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MNK}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, DefaultGemmKLoop, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MNK}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, GemmKLoopOverC  , 2, m_per_thread, n_per_thread, false, false, c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MNK}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, DefaultGemmKLoop, 2, m_per_thread, n_per_thread, true,  false, c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MNK}), \
    \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, GemmKLoopOverC  , 2, m_per_thread, n_per_thread, true,  true,  c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MKN}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, GemmKLoopOverC  , 2, m_per_thread, n_per_thread, true,  true,  c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MKN}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, DefaultGemmKLoop, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MKN}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, GemmKLoopOverC  , 2, m_per_thread, n_per_thread, false, false, c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MKN}), \
    DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K<float , float , float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, DefaultGemmKLoop, 2, m_per_thread, n_per_thread, true,  false, c_local_buf>({m_per_block, n_per_block, k_per_block, LoopOver_MKN})
// clang-format on

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk(
    std::vector<DeviceConvFwdPtr<PT, PT, PT>>& instances)
{
    ck::tensor_operation::device::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  256, 128,  64, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  256, 128, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  128, 256, 128, 6, 16, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  512, 240, 128, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  512, 256, 128, 6, 16, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  768, 320, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  896, 352, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT, 1024, 416, 128, 6, 16, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_local_c(
    std::vector<DeviceConvFwdPtr<PT, PT, PT>>& instances)
{
    ck::tensor_operation::device::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  256, 128,  64, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  256, 128, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  128, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  512, 240, 128, 4, 24, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  512, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  768, 320, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  896, 352, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT, 1024, 416, 128, 6, 16, true)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_mt(
    std::vector<DeviceConvFwdPtr<PT, PT, PT>>& instances)
{
    ck::tensor_operation::device::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   24,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   32,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   40,  24, 256, 4, 24, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   48,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   48,  48, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   56,  24, 256, 4, 24, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   72,  16, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   72,  16, 256, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   72,  32, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   72,  32, 256, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   96,  32, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,   96,  64, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  120,  32, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  120,  64, 128, 6, 16, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  256, 128, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  128, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  512, 240, 128, 4, 24, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  512, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  768, 320, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT,  896, 352, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, PT, 1024, 416, 128, 6, 16, true)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_relu(
    std::vector<DeviceConvFwdPtr<PT, PT, Relu>>& instances)
{
    ck::tensor_operation::device::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  256, 128,  64, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  256, 128, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  128, 256, 128, 6, 16, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  512, 240, 128, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  512, 256, 128, 6, 16, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  768, 320, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  896, 352, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu, 1024, 416, 128, 6, 16, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_local_c_relu(
    std::vector<DeviceConvFwdPtr<PT, PT, Relu>>& instances)
{
    ck::tensor_operation::device::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  256, 128,  64, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  256, 128, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  128, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  512, 240, 128, 4, 24, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  512, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  768, 320, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  896, 352, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu, 1024, 416, 128, 6, 16, true)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_mt_relu(
    std::vector<DeviceConvFwdPtr<PT, PT, Relu>>& instances)
{
    ck::tensor_operation::device::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   24,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   32,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   40,  24, 256, 4, 24, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   24,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   32,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   40, 24, 256, 4, 24, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   48,  24, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   48,  48, 256, 4, 24, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   56,  24, 256, 4, 24, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   72,  16, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   72,  16, 256, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   72,  32, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   72,  32, 256, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   96,  32, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,   96,  64, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  120,  32, 128, 6, 16, false),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  120,  64, 128, 6, 16, false),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  256, 128, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  128, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  512, 240, 128, 4, 24, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  512, 256, 128, 6, 16, true),

            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  768, 320, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu,  896, 352, 128, 6, 16, true),
            DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_F32(PT, PT, Relu, 1024, 416, 128, 6, 16, true)
            // clang-format on
            ));
}

} // namespace device_conv2d_fwd_avx2_instance
} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck
