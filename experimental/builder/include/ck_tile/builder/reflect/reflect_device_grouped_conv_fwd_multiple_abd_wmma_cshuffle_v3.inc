// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// InstanceTraits specialization for DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
//
// This .inc file is #included at the bottom of the device op header
// (device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp) under
// #ifdef CK_EXPERIMENTAL_BUILDER, AFTER the struct is fully defined.
// This eliminates the need for forward declarations.
//
// CRITICAL MAINTENANCE NOTE:
// This file MUST be kept strictly in sync with the device implementation header:
//   ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp
// The template parameter order, names, and types MUST EXACTLY MATCH those in the device
// implementation.

#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"

namespace ck_tile::reflect {

// Specialization for DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
template <ck::index_t NDimSpatial,
          typename ALayout_,
          typename BLayout_,
          typename DsLayout_,
          typename ELayout_,
          typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CShuffleDataType_,
          typename DsDataType_,
          typename EDataType_,
          typename AElementwiseOperation_,
          typename BElementwiseOperation_,
          typename CDEElementwiseOperation_,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder_,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          ck::index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder_,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          ck::index_t BBlockLdsExtraN,
          ck::index_t CShuffleMRepeatPerShuffle,
          ck::index_t CShuffleNRepeatPerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          bool UseThreadTileTransfer,
          typename AComputeDataType_,
          typename BComputeDataType_,
          ck::index_t NumGroupsToMerge>
struct InstanceTraits<
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<
        NDimSpatial,
        ALayout_,
        BLayout_,
        DsLayout_,
        ELayout_,
        ADataType_,
        BDataType_,
        AccDataType_,
        CShuffleDataType_,
        DsDataType_,
        EDataType_,
        AElementwiseOperation_,
        BElementwiseOperation_,
        CDEElementwiseOperation_,
        ConvForwardSpecialization,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder_,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder_,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        UseThreadTileTransfer,
        AComputeDataType_,
        BComputeDataType_,
        NumGroupsToMerge>>
{
    /// @brief Tag type identifying this device kernel variant
    using device_kernel_tag = DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3_Tag;

    // Spatial dimension
    static constexpr int kSpatialDim = NDimSpatial;

    // Layout types
    using ALayout  = ALayout_;
    using BLayout  = BLayout_;
    using DsLayout = DsLayout_;
    using ELayout  = ELayout_;

    // Data types
    using ADataType        = ADataType_;
    using BDataType        = BDataType_;
    using AccDataType      = AccDataType_;
    using CShuffleDataType = CShuffleDataType_;
    using DsDataType       = DsDataType_;
    using EDataType        = EDataType_;

    // Element-wise operations
    using AElementwiseOperation   = AElementwiseOperation_;
    using BElementwiseOperation   = BElementwiseOperation_;
    using CDEElementwiseOperation = CDEElementwiseOperation_;

    // Specialization
    static constexpr ck::tensor_operation::device::ConvolutionForwardSpecialization
        kConvForwardSpecialization = ConvForwardSpecialization;
    static constexpr ck::tensor_operation::device::GemmSpecialization kGemmSpecialization =
        GemmSpec;

    // Block configuration
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kMPerBlock = MPerBlock;
    static constexpr int kNPerBlock = NPerBlock;
    static constexpr int kKPerBlock = KPerBlock;

    // Tuning parameters
    static constexpr int kAK1      = AK1;
    static constexpr int kBK1      = BK1;
    static constexpr int kMPerWmma = MPerWmma;
    static constexpr int kNPerWmma = NPerWmma;
    static constexpr int kMRepeat  = MRepeat;
    static constexpr int kNRepeat  = NRepeat;

    // A block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kAThreadClusterLengths =
        detail::SequenceToArray<ABlockTransferThreadClusterLengths_AK0_M_AK1>::value;
    static constexpr auto kAThreadClusterArrangeOrder =
        detail::SequenceToArray<ABlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kABlockTransferSrcAccessOrder =
        detail::SequenceToArray<ABlockTransferSrcAccessOrder_>::value;
    static constexpr int kABlockTransferSrcVectorDim         = ABlockTransferSrcVectorDim;
    static constexpr int kABlockTransferSrcScalarPerVector   = ABlockTransferSrcScalarPerVector;
    static constexpr int kABlockTransferDstScalarPerVectorK1 = ABlockTransferDstScalarPerVector_AK1;
    static constexpr int kABlockLdsExtraM                    = ABlockLdsExtraM;

    // B block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kBThreadClusterLengths =
        detail::SequenceToArray<BBlockTransferThreadClusterLengths_BK0_N_BK1>::value;
    static constexpr auto kBThreadClusterArrangeOrder =
        detail::SequenceToArray<BBlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kBBlockTransferSrcAccessOrder =
        detail::SequenceToArray<BBlockTransferSrcAccessOrder_>::value;
    static constexpr int kBBlockTransferSrcVectorDim         = BBlockTransferSrcVectorDim;
    static constexpr int kBBlockTransferSrcScalarPerVector   = BBlockTransferSrcScalarPerVector;
    static constexpr int kBBlockTransferDstScalarPerVectorK1 = BBlockTransferDstScalarPerVector_BK1;
    static constexpr int kBBlockLdsExtraN                    = BBlockLdsExtraN;

    // C shuffle parameters (converted to std::array)
    static constexpr int kCShuffleMRepeatPerShuffle = CShuffleMRepeatPerShuffle;
    static constexpr int kCShuffleNRepeatPerShuffle = CShuffleNRepeatPerShuffle;
    static constexpr auto kCDEThreadClusterLengths  = detail::SequenceToArray<
         CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>::value;
    static constexpr int kCDEBlockTransferScalarPerVector =
        CDEBlockTransferScalarPerVector_NPerBlock;

    // Pipeline configuration
    static constexpr ck::BlockGemmPipelineScheduler kPipelineScheduler = BlkGemmPipeSched;
    static constexpr ck::BlockGemmPipelineVersion kPipelineVersion     = BlkGemmPipelineVer;

    static constexpr bool kUseThreadTileTransfer = UseThreadTileTransfer;

    // Compute data types
    using AComputeDataType = AComputeDataType_;
    using BComputeDataType = BComputeDataType_;

    static constexpr int kNumGroupsToMerge = NumGroupsToMerge;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3";

        // Template parameters in exact order
        oss << "<" << kSpatialDim;                                  // 1. NDimSpatial
        oss << "," << detail::layout_name<ALayout>();               // 2. ALayout
        oss << "," << detail::layout_name<BLayout>();               // 3. BLayout
        oss << "," << detail::tuple_name<DsLayout>();               // 4. DsLayout
        oss << "," << detail::layout_name<ELayout>();               // 5. ELayout
        oss << "," << detail::type_or_type_tuple_name<ADataType>(); // 6. ADataType
        oss << "," << detail::type_or_type_tuple_name<BDataType>(); // 7. BDataType
        oss << "," << detail::type_name<AccDataType>();             // 8. AccDataType
        oss << "," << detail::type_name<CShuffleDataType>();        // 9. CShuffleDataType
        oss << "," << detail::tuple_name<DsDataType>();             // 10. DsDataType
        oss << "," << detail::type_name<EDataType>();               // 11. EDataType
        oss << ","
            << detail::elementwise_op_name<AElementwiseOperation>(); // 12. AElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<BElementwiseOperation>(); // 13. BElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<CDEElementwiseOperation>(); // 14.
                                                                       // CDEElementwiseOperation
        oss << ","
            << detail::conv_fwd_spec_name(
                   kConvForwardSpecialization);                    // 15. ConvForwardSpecialization
        oss << "," << detail::gemm_spec_name(kGemmSpecialization); // 16. GemmSpec
        oss << "," << kBlockSize;                                  // 17. BlockSize
        oss << "," << kMPerBlock;                                  // 18. MPerBlock
        oss << "," << kNPerBlock;                                  // 19. NPerBlock
        oss << "," << kKPerBlock;                                  // 20. KPerBlock
        oss << "," << kAK1;                                        // 21. AK1
        oss << "," << kBK1;                                        // 22. BK1
        oss << "," << kMPerWmma;                                   // 23. MPerWmma
        oss << "," << kNPerWmma;                                   // 24. NPerWmma
        oss << "," << kMRepeat;                                    // 25. MRepeat
        oss << "," << kNRepeat;                                    // 26. NRepeat
        oss << ","
            << detail::array_to_string(
                   kAThreadClusterLengths); // 27. ABlockTransferThreadClusterLengths
        oss << ","
            << detail::array_to_string(
                   kAThreadClusterArrangeOrder); // 28. ABlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::array_to_string(
                   kABlockTransferSrcAccessOrder);       // 29. ABlockTransferSrcAccessOrder
        oss << "," << kABlockTransferSrcVectorDim;       // 30. ABlockTransferSrcVectorDim
        oss << "," << kABlockTransferSrcScalarPerVector; // 31. ABlockTransferSrcScalarPerVector
        oss << ","
            << kABlockTransferDstScalarPerVectorK1; // 32. ABlockTransferDstScalarPerVector_AK1
        oss << "," << (kABlockLdsExtraM ? "true" : "false"); // 33. ABlockLdsExtraM
        oss << ","
            << detail::array_to_string(
                   kBThreadClusterLengths); // 34. BBlockTransferThreadClusterLengths
        oss << ","
            << detail::array_to_string(
                   kBThreadClusterArrangeOrder); // 35. BBlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::array_to_string(
                   kBBlockTransferSrcAccessOrder);       // 36. BBlockTransferSrcAccessOrder
        oss << "," << kBBlockTransferSrcVectorDim;       // 37. BBlockTransferSrcVectorDim
        oss << "," << kBBlockTransferSrcScalarPerVector; // 38. BBlockTransferSrcScalarPerVector
        oss << ","
            << kBBlockTransferDstScalarPerVectorK1; // 39. BBlockTransferDstScalarPerVector_BK1
        oss << "," << (kBBlockLdsExtraN ? "true" : "false"); // 40. BBlockLdsExtraN
        oss << "," << kCShuffleMRepeatPerShuffle;            // 41. CShuffleMRepeatPerShuffle
        oss << "," << kCShuffleNRepeatPerShuffle;            // 42. CShuffleNRepeatPerShuffle
        oss << ","
            << detail::array_to_string(
                   kCDEThreadClusterLengths); // 43. CDEBlockTransferClusterLengths
        oss << ","
            << kCDEBlockTransferScalarPerVector; // 44. CDEBlockTransferScalarPerVector_NPerBlock
        oss << "," << detail::pipeline_scheduler_name(kPipelineScheduler); // 45. BlkGemmPipeSched
        oss << "," << detail::pipeline_version_name(kPipelineVersion);     // 46. BlkGemmPipelineVer
        oss << "," << (kUseThreadTileTransfer ? "true" : "false"); // 47. UseThreadTileTransfer
        oss << "," << detail::type_name<AComputeDataType>();       // 48. AComputeDataType
        oss << "," << detail::type_name<BComputeDataType>();       // 49. BComputeDataType
        oss << "," << kNumGroupsToMerge;                           // 50. NumGroupsToMerge
        oss << ">";

        return oss.str();
    }
};

} // namespace ck_tile::reflect
