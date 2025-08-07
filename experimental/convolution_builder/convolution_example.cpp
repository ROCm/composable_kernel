#include <iostream>

#include <hip/hip_runtime.h>

#include "convolution_builder.hpp"

// Example of problem description for Forward Conv with default settings
struct GroupedConvFwdXdlImplicitGemm : public GroupedConvBaseXdlV1 {
    static constexpr ConvolutionDirection ConvolutionDirection_ = ConvolutionDirection::Forward;
};

// Example of problem description for Backward Weight Conv with default settings and Split K Two Stage
struct GroupedConvBwdWeightXdlImplicitGemmTwoStage : public GroupedConvBaseXdlV1  {
    static constexpr ConvolutionDirection ConvolutionDirection_ = ConvolutionDirection::BackwardWeight;
    static constexpr SplitKSupport SplitKSupport_ = SplitKSupport::SupportedTwoStage;
};

int main () {
    ConvolutionBuilder<GroupedConvFwdXdlImplicitGemm> builder_fwd;
    std::cout << builder_fwd.GetInstanceName() << std::endl;
    ConvolutionBuilder<GroupedConvBwdWeightXdlImplicitGemmTwoStage> builder_bwd_weight_two_stage;
    std::cout << builder_bwd_weight_two_stage.GetInstanceName() << std::endl;
    return 0;
}
