#pragma once
#include "convolution_problem_descriptor.hpp"
#include "convolution_implementation_descriptor.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

template<typename Problem, typename Implementation>
struct ConvolutionBuilder;

template<ProblemDescriptorV1 Problem, ImplementationDescriptorV1 Implementation>
struct ConvolutionBuilder<Problem, Implementation> {
public:
    static constexpr auto GetInstance() {
        using DataType = typename Implementation::DataType;
        using AccDataType = std::conditional_t<std::is_same_v<DataType, int8_t>, int32_t, float>;
        using InLayout = std::tuple_element<0, decltype(GetLayout())>::type;
        using WeiLayout = std::tuple_element<1,decltype( GetLayout())>::type;
        using OutLayout = std::tuple_element<2,decltype( GetLayout())>::type;

        using GroupedConvFwdMultipleABD_Xdl_CShuffleInstance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< Implementation::NDimSpatial_, InLayout, WeiLayout, decltype(GetMultiDLayout()), OutLayout, DataType, DataType, DataType, AccDataType, typename Implementation::ElementwiseOpDataTypes, DataType, ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough,  decltype(GetOutElementwiseOp()), GetConvSpecialization(), ck::tensor_operation::device::GemmSpecialization::MNKPadding, 1, Implementation::BlockSize_, Implementation::TileSizes_::At(0),    Implementation::TileSizes_::At(1),     Implementation::TileSizes_::At(2), Implementation::K1_, Implementation::K1_, 16, 16, 1, 1, ck::Sequence<4, 8, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, 1, 4, 1, ck::Sequence<4, 8, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, 1, 4, 1, 1, 1, ck::Sequence<1, 32, 1, 8>, 1>;
        using DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffleInstance =  ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<Implementation::NDimSpatial_,  InLayout, WeiLayout, OutLayout,   DataType, DataType, DataType,    AccDataType , ck::tensor_operation::element_wise::PassThrough, ck::tensor_operation::element_wise::PassThrough, decltype(GetOutElementwiseOp()),  GetConvSpecialization(),    Implementation::BlockSize_,    Implementation::TileSizes_::At(0),    Implementation::TileSizes_::At(1),     Implementation::TileSizes_::At(2),   Implementation::K1_,   16,   16,    1,    1,  ck::Sequence<4, 8,  1>, ck::Sequence<2, 0, 1>,  ck::Sequence<1, 0, 2>,                   1,              1,              4,      false,  ck::Sequence<4, 8,  1>,  ck::Sequence<2, 0, 1>,  ck::Sequence<1, 0, 2>,                1,              1,              4,      false,           1,           1,   ck::Sequence<1, 8, 1, 8>,                  1>;
        
        using SelectedInstance = std::conditional_t<GetKernel() == Kernel::GroupedConvFwdMultipleABD_Xdl_CShuffle, GroupedConvFwdMultipleABD_Xdl_CShuffleInstance, DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffleInstance>;
        return SelectedInstance{};
    }

    std::string GetInstanceName() const
    {
        auto str = std::stringstream();

        std::map<Kernel, std::string> KernelToString{
            {Kernel::GroupedConvFwdMultipleABD_Xdl_CShuffle, "GroupedConvFwdMultipleABD_Xdl_CShuffle"},
            {Kernel::GroupedConvBwdWeightTwoStage_Xdl_CShuffle, "GroupedConvBwdWeightTwoStage_Xdl_CShuffle"}};

        std::map<GemmPipelineScheduler, std::string> GemmPipelineSchedulerToString{
            {GemmPipelineScheduler::Intrawave, "Intrawave"},
            {GemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<GemmPipelineVersion, std::string> GemmPipelineVersionToString{
            {GemmPipelineVersion::V1, "v1"},
            {GemmPipelineVersion::V2, "v2"},
            {GemmPipelineVersion::V3, "v3"},
            {GemmPipelineVersion::V4, "v4"},
            {GemmPipelineVersion::V5, "v5"}};

        std::map<MFMAInstructionSize, std::string> MFMAInstructionSizeToString{
            {MFMAInstructionSize::M16N16, "16x16"},
            {MFMAInstructionSize::M32N32, "32x32"}};

        std::map<ConvolutionSpecialization, std::string> ConvolutionSpecializationToString{
            {ConvolutionSpecialization::Default, "Default"},
            {ConvolutionSpecialization::Filter1x1Pad0, "Filter1x1Pad0"},
            {ConvolutionSpecialization::Filter1x1Stride1Pad0, "Filter1x1Stride1Pad0"},
            {ConvolutionSpecialization::Filter3x3, "Filter3x3"}};

        std::map<MergedGroups, std::string> MergedGroupsToString{
            {MergedGroups::X16, "16"},
            {MergedGroups::X8, "8"},
            {MergedGroups::X4, "4"},
            {MergedGroups::X2, "2"},
            {MergedGroups::NotSupported, "1"}};

        // clang-format off
        str << KernelToString[GetKernel()]
            << "<"
            << Implementation::BlockSize_ << ", "
            << std::get<0>(Implementation::TileSizes_) << ", "
            << std::get<1>(Implementation::TileSizes_) << ", "
            << std::get<2>(Implementation::TileSizes_) << ", "
            << ConvolutionSpecializationToString[Implementation::ConvolutionSpecialization_]  << ", "
            << Implementation::K1_ << ", "
            << MFMAInstructionSizeToString[Implementation::MFMAInstructionSize_] << ", "
            << std::get<0>(Implementation::XdlPerWave_) << ", "
            << std::get<1>(Implementation::XdlPerWave_) << ", "
            << std::get<0>(Implementation::GlobalTransferVectorSize_) << ", "
            << std::get<0>(Implementation::LDSStoreVectorSize_) << ", "
            << std::get<1>(Implementation::GlobalTransferVectorSize_) << ", "
            << std::get<1>(Implementation::LDSStoreVectorSize_) << ", "
            << std::get<2>(Implementation::GlobalTransferVectorSize_) << ", "
            << GemmPipelineSchedulerToString[Problem::GemmPipelineScheduler_] << ", "
            << GemmPipelineVersionToString[Problem::GemmPipelineVersion_] << ", "
            << MergedGroupsToString[Problem::MergedGroups_] << ">";
        // clang-format on

        return str.str();
    }

private:
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

    static constexpr auto GetLayout() {
        if constexpr(Implementation::NDimSpatial_ == 2) {
            if constexpr(Implementation::ConvolutionLayout_ == ConvolutionLayout::NHWGC_GKYXC_NHWGK) {
                return std::tuple<ck::tensor_layout::convolution::NHWGC, ck::tensor_layout::convolution::GKYXC, ck::tensor_layout::convolution::NHWGK>{};
            } else {
                static_assert("Layout not supported!");
            }
        } else {
            static_assert("Not supported spatial dim!");
        }
    }

    static constexpr auto GetMultiDLayout() {
        return ck::Tuple<>{};
    }

    static constexpr auto GetOutElementwiseOp() {
        return ck::tensor_operation::element_wise::PassThrough{};
    }
    
    static constexpr auto GetConvSpecialization() {
        if constexpr(Problem::ConvolutionDirection_ == ConvolutionDirection::Forward) {
            return ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
        } else if constexpr(Problem::ConvolutionDirection_ == ConvolutionDirection::BackwardWeight) {
            return ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;
        } else {
            static_assert("Specialization not found!");
        }
    }
};

