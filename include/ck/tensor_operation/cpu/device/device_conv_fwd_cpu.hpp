#ifndef DEVICE_CONV_FWD_CPU_HPP
#define DEVICE_CONV_FWD_CPU_HPP

#include <iostream>
#include "device_base_cpu.hpp"
#include "convolution_forward_specialization_cpu.hpp"

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceConvFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
using DeviceConvFwdPtr = std::unique_ptr<
    DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>>;

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceConvFwdBiasActivationAdd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        const void* p_bias_grid,
                        const void* p_add_grid,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
using DeviceConvFwdBiasActivationAddPtr =
    std::unique_ptr<DeviceConvFwdBiasActivationAdd<InElementwiseOperation,
                                                   WeiElementwiseOperation,
                                                   OutElementwiseOperation>>;

struct DeviceConvFwdDynamicTunable
{
    ck::index_t m_per_block;
    ck::index_t n_per_block;
    ck::index_t k_per_block;

    // ck::index_t m_per_thread;
    // ck::index_t n_per_thread;

    // bool use_a_local_buffer;
    // bool use_b_local_buffer;
    // bool use_c_local_buffer;

    // ConvolutionForwardSpecialization_t  forward_spec;
    // ConvolutionForwardGemmKSpecialization_t gemm_k_spec;
    ConvolutionForwardBlockLoopOverSpecialization_t loop_over_spec;
};

} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck
#endif
