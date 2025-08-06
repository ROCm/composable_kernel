#include <iostream>

#include <hip/hip_runtime.h>

#include "convolution_builder.hpp"

// Example of solution description for Forward Conv with default settings
struct GroupedConvFwdXdlImplicitGemm : public GroupedConvBaseXdl {
    static constexpr ConvolutionDirection ConvolutionDirection_ = ConvolutionDirection::Forward;
};

// Example of solution description for Backward Weight Conv with default settings and Split K Two Stage
struct GroupedConvBwdWeightXdlImplicitGemm : public GroupedConvBaseXdl  {
    static constexpr ConvolutionDirection ConvolutionDirection_ = ConvolutionDirection::BackwardWeight;
    static constexpr SplitKSupport SplitKSupport_ = SplitKSupport::SupportedTwoStage;
};

int main () {
    ConvolutionBuilder<GroupedConvFwdXdlImplicitGemm> builder;
    std::cout << builder.GetKernelName() << std::endl;
    return 0;
}
