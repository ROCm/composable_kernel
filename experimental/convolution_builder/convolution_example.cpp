#include <iostream>

#include <hip/hip_runtime.h>

#include "convolution_builder.hpp"

// Example of kernel description for Forward Conv with default settings
struct GroupedConvFwdXdlImplicitGemm : public GroupedConvBaseXdlV1 {
    static constexpr ConvolutionDirection ConvolutionDirection_ = ConvolutionDirection::Forward;
    static constexpr ElementwiseOperation ElementwiseOperation_ = ElementwiseOperation::Bias;
};

// Example of kernel description for Backward Weight Conv with default settings and Split K Two Stage
struct GroupedConvBwdWeightXdlImplicitGemmTwoStage : public GroupedConvBaseXdlV1  {
    static constexpr ConvolutionDirection ConvolutionDirection_ = ConvolutionDirection::BackwardWeight;
    static constexpr SplitKSupport SplitKSupport_ = SplitKSupport::SupportedTwoStage;
};

struct Implementation16x16 : ImplementationDefaultV1 {
    static constexpr ck::index_t BlockSize_ = 64;
    static constexpr auto TileSizes_ = std::make_tuple(16, 16, 32);
    static constexpr ck::index_t K1_ = 8;
    static constexpr MFMAInstructionSize MFMAInstructionSize_ = MFMAInstructionSize::M16N16;
    static constexpr auto XdlPerWave_ = std::make_tuple(16, 16);
    static constexpr auto GlobalTransferVectorSize_ = std::make_tuple(1, 1, 1);
    static constexpr auto LDSStoreVectorSize_ = std::make_tuple(4, 4);
};

struct ProblemBF16NHWGC : public BF16ProblemBaseV1, public NHWGCProblemBaseV1 {};

int main () {
    ConvolutionBuilder<GroupedConvFwdXdlImplicitGemm, ProblemBF16NHWGC, Implementation16x16> builder_fwd;
    std::cout << builder_fwd.GetInstanceName() << std::endl;
    ConvolutionBuilder<GroupedConvBwdWeightXdlImplicitGemmTwoStage, ProblemBF16NHWGC, Implementation16x16> builder_bwd_weight_two_stage;
    std::cout << builder_bwd_weight_two_stage.GetInstanceName() << std::endl;
    return 0;
}
