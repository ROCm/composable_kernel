#pragma once
#include "convolution_solution_descriptor.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

template<SolutionDescriptorV1 Solution>
struct ConvolutionBuilder {

    static constexpr auto GetSolution() {
        if constexpr(Solution::GemmImplementationType_ == GemmImplementationType::XDL) {
            if constexpr(Solution::ConvolutionDirection_ == ConvolutionDirection::Forward) {
                return ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2, ck::tensor_layout::convolution::NHWGC, ck::tensor_layout::convolution::GKYXC, ck::Tuple<>, ck::tensor_layout::convolution::NHWGK, float, float, float, float, ck::Tuple<>, float, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::device::ConvolutionForwardSpecialization::Default, ck::tensor_operation::device::GemmSpecialization::MNKPadding, 1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4, ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, 8, 8, 1, ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, 8, 8, 1, 1, 1, ck::Sequence<1, 32, 1, 8>, 8>{};
            } else if constexpr(Solution::ConvolutionDirection_ == ConvolutionDirection::BackwardData) {
                return std::tuple<>{};
            } else if constexpr(Solution::ConvolutionDirection_ == ConvolutionDirection::BackwardWeight) {
                return ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<2,  ck::tensor_layout::convolution::NHWGC, ck::tensor_layout::convolution::GKYXC, ck::tensor_layout::convolution::NHWGK,    float,     float,     float,     float, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough,                  ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,    64,    16,    16,     32,   8,   16,   16,    1,    1,  ck::Sequence<4, 8,  1>, ck::Sequence<2, 0, 1>,  ck::Sequence<1, 0, 2>,                   1,              1,              4,      false,  ck::Sequence<4, 8,  1>,  ck::Sequence<2, 0, 1>,  ck::Sequence<1, 0, 2>,                1,              1,              4,      false,           1,           1,   ck::Sequence<1, 8, 1, 8>,                  1>{};
            } else {
                return std::tuple<>{};
            }
        } else {
            return std::tuple<>{};
        }
    }


    std::string GetKernelName() const
    {
        return GetSolution().GetTypeString();
    }
};
