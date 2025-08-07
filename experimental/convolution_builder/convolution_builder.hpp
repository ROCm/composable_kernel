#pragma once
#include "convolution_problem_descriptor.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

template<typename Problem>
struct ConvolutionBuilder;

template<ProblemDescriptorV1 Problem>
struct ConvolutionBuilder<Problem> {

    enum class Kernel {
        GroupedConvFwdMultipleABD_Xdl_CShuffle,
        GroupedConvBwdWeightTwoStage_Xdl_CShuffle
    };

    static constexpr Kernel GetKernel() {
        if constexpr(Problem::GemmImplementationType_ == GemmImplementationType::XDL) {
            if constexpr(Problem::ConvolutionDirection_ == ConvolutionDirection::Forward) {
                return Kernel::GroupedConvFwdMultipleABD_Xdl_CShuffle;
            } else if constexpr(Problem::ConvolutionDirection_ == ConvolutionDirection::BackwardData) {
                static_assert("Instance not found!");
            } else if constexpr(Problem::ConvolutionDirection_ == ConvolutionDirection::BackwardWeight) {
                return Kernel::GroupedConvBwdWeightTwoStage_Xdl_CShuffle;
            } else {
                static_assert("Instance not found!");
            }
        } else {
            static_assert("Instance not found!");
        }
    }

    static constexpr auto GetInstance() {
        using FwdInstance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2, ck::tensor_layout::convolution::NHWGC, ck::tensor_layout::convolution::GKYXC, ck::Tuple<>, ck::tensor_layout::convolution::NHWGK, float, float, float, float, ck::Tuple<>, float, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::device::ConvolutionForwardSpecialization::Default, ck::tensor_operation::device::GemmSpecialization::MNKPadding, 1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4, ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, 8, 8, 1, ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, 8, 8, 1, 1, 1, ck::Sequence<1, 32, 1, 8>, 8>;
        using BwdWeightInstance =  ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<2,  ck::tensor_layout::convolution::NHWGC, ck::tensor_layout::convolution::GKYXC, ck::tensor_layout::convolution::NHWGK,    float,     float,     float,     float, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough,                  ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,    64,    16,    16,     32,   8,   16,   16,    1,    1,  ck::Sequence<4, 8,  1>, ck::Sequence<2, 0, 1>,  ck::Sequence<1, 0, 2>,                   1,              1,              4,      false,  ck::Sequence<4, 8,  1>,  ck::Sequence<2, 0, 1>,  ck::Sequence<1, 0, 2>,                1,              1,              4,      false,           1,           1,   ck::Sequence<1, 8, 1, 8>,                  1>;
        
        using SelectedInstance = std::conditional_t<GetKernel() == Kernel::GroupedConvFwdMultipleABD_Xdl_CShuffle, FwdInstance, BwdWeightInstance>;
        return SelectedInstance{};
    }

    std::string GetInstanceName() const
    {
        if constexpr(GetKernel() == Kernel::GroupedConvFwdMultipleABD_Xdl_CShuffle) {
            return "GroupedConvFwdMultipleABD_Xdl_CShuffle";
        } else if constexpr (GetKernel() == Kernel::GroupedConvBwdWeightTwoStage_Xdl_CShuffle) {
            return "GroupedConvBwdWeightTwoStage_Xdl_CShuffle";
        } else {
            return "Not found.";
        }
        // const auto instance =  GetInstance();
        // return instance.GetTypeString();
    }
};

