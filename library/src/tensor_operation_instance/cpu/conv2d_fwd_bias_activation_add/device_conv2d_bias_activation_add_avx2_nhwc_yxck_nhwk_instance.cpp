#include <stdlib.h>
#include <utility>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/cpu/device/convolution_forward_specialization_cpu.hpp"
#include "ck/tensor_operation/cpu/device/device_convnd_fwd_bias_activation_add_avx2_nhwc_yxck_nhwk.hpp"
#include "ck/tensor_operation/cpu/element/element_wise_operation_cpu.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {
namespace device_conv2d_fwd_bias_activation_add_avx2_instance {

using InType  = float;
using WeiType = float;
using OutType = float;
using AccType = float;

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

// clang-format off
#define DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(a_elem_op, b_elem_op, c_elem_op, m_per_block, n_per_block, k_per_block, m_per_thread, n_per_thread, c_local_buf, fuse_bias, fuse_add, bias_along_m) \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, GemmKLoopOverC  , LoopOver_MNK}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, GemmKLoopOverC  , LoopOver_MNK}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, DefaultGemmKLoop, LoopOver_MNK}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, 2, m_per_thread, n_per_thread, false, false, c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, GemmKLoopOverC  , LoopOver_MNK}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, 2, m_per_thread, n_per_thread, true,  false, c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, DefaultGemmKLoop, LoopOver_MNK}), \
    \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, GemmKLoopOverC  , LoopOver_MKN}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, GemmKLoopOverC  , LoopOver_MKN}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, 2, m_per_thread, n_per_thread, true,  true,  c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, DefaultGemmKLoop, LoopOver_MKN}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwd1x1S1P0, 2, m_per_thread, n_per_thread, false, false, c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, GemmKLoopOverC  , LoopOver_MKN}), \
    DeviceConvNDFwdBiasActivationAddAvx2_Input_N_Hi_Wi_C_Weight_Y_X_C_K_Output_N_Ho_Wo_K<float , float , float, float, float, a_elem_op, b_elem_op, c_elem_op, ConvFwdDefault, 2, m_per_thread, n_per_thread, true,  false, c_local_buf, fuse_bias, fuse_add, bias_along_m>({m_per_block, n_per_block, k_per_block, DefaultGemmKLoop, LoopOver_MKN})
// clang-format on

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddReluAdd>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  256, 128,  64,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  256, 128, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  128, 256, 128,  6, 16, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  512, 240, 128,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  512, 256, 128,  6, 16, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  768, 320, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  896, 352, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd, 1024, 416, 128,  6, 16, false, true, true, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddReluAdd>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  256, 128,  64,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  256, 128, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  128, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  512, 240, 128,  4, 24, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  512, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  768, 320, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  896, 352, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd, 1024, 416, 128,  6, 16, true, true, true, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddReluAdd>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  24,  24, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  32,  24, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  40,  24, 256,  4, 24, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  48,  24, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  48,  48, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  56,  24, 256,  4, 24, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  72,  16, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  72,  16, 256,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  72,  32, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  72,  32, 256,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  96,  32, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  96,  64, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  120, 32, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  120, 64, 128,  6, 16, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  256, 128, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  128, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  512, 240, 128,  4, 24, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  512, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  768, 320, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd,  896, 352, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddReluAdd, 1024, 416, 128,  6, 16, true, true, true, false)
            // clang-format on
            ));
}

/****************************************************************************************************/

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddRelu>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  256, 128,  64,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  256, 128, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  128, 256, 128,  6, 16, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  512, 240, 128,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  512, 256, 128,  6, 16, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  768, 320, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  896, 352, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu, 1024, 416, 128,  6, 16, false, true, false, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddRelu>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  256, 128,  64,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  256, 128, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  128, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  512, 240, 128,  4, 24, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  512, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  768, 320, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  896, 352, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu, 1024, 416, 128,  6, 16, true, true, false, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddRelu>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  24,  24, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  32,  24, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  40,  24, 256,  4, 24, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  48,  24, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  48,  48, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  56,  24, 256,  4, 24, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  72,  16, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  72,  16, 256,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  72,  32, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  72,  32, 256,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  96,  32, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  96,  64, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  120, 32, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  120, 64, 128,  6, 16, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  256, 128, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  128, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  512, 240, 128,  4, 24, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  512, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  768, 320, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu,  896, 352, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddRelu, 1024, 416, 128,  6, 16, true, true, false, false)
            // clang-format on
            ));
}

/****************************************************************************************************/

void add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, Add>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  256, 128,  64,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  256, 128, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  128, 256, 128,  6, 16, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  512, 240, 128,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  512, 256, 128,  6, 16, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  768, 320, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  896, 352, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add, 1024, 416, 128,  6, 16, false, true, false, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, Add>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  256, 128,  64,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  256, 128, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  128, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  512, 240, 128,  4, 24, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  512, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  768, 320, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  896, 352, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add, 1024, 416, 128,  6, 16, true, true, false, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, Add>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  24,  24, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  32,  24, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  40,  24, 256,  4, 24, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  48,  24, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  48,  48, 256,  4, 24, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  56,  24, 256,  4, 24, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  72,  16, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  72,  16, 256,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  72,  32, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  72,  32, 256,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  96,  32, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  96,  64, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  120, 32, 128,  6, 16, false, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  120, 64, 128,  6, 16, false, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  256, 128, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  128, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  512, 240, 128,  4, 24, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  512, 256, 128,  6, 16, true, true, false, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  768, 320, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add,  896, 352, 128,  6, 16, true, true, false, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, Add, 1024, 416, 128,  6, 16, true, true, false, false)
            // clang-format on
            ));
}

/****************************************************************************************************/
void add_device_conv2d_fwd_bias_add_relu_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddAddRelu>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  256, 128,  64,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  256, 128, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  128, 256, 128,  6, 16, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  512, 240, 128,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  512, 256, 128,  6, 16, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  768, 320, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  896, 352, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu, 1024, 416, 128,  6, 16, false, true, true, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_add_relu_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddAddRelu>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  256, 128,  64,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  256, 128, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  128, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  512, 240, 128,  4, 24, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  512, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  768, 320, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  896, 352, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu, 1024, 416, 128,  6, 16, true, true, true, false)
            // clang-format on
            ));
}

void add_device_conv2d_fwd_bias_add_relu_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PT, PT, AddAddRelu>>& instances)
{
    ck::tensor_operation::device::instance::add_device_operation_instances(
        instances,
        std::make_tuple(
            // clang-format off
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  24,  24, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  32,  24, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  40,  24, 256,  4, 24, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  48,  24, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  48,  48, 256,  4, 24, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  56,  24, 256,  4, 24, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  72,  16, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  72,  16, 256,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  72,  32, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  72,  32, 256,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  96,  32, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  96,  64, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  120, 32, 128,  6, 16, false, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  120, 64, 128,  6, 16, false, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  256, 128, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  128, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  512, 240, 128,  4, 24, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  512, 256, 128,  6, 16, true, true, true, false),

            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  768, 320, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu,  896, 352, 128,  6, 16, true, true, true, false),
            DEVICE_CONV2D_FWD_BAA_AVX2_NHWC_YXCK_NHWK_F32(PT, PT, AddAddRelu, 1024, 416, 128,  6, 16, true, true, true, false)
            // clang-format on
            ));
}

} // namespace device_conv2d_fwd_bias_activation_add_avx2_instance
} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck
