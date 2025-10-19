// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// InstanceTraits specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3

#pragma once

#include "instance_traits.hpp"

namespace ck_tile::reflect {

// Specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
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
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
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
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AComputeDataType_,
          typename BComputeDataType_>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
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
    MPerXDL,
    NPerXDL,
    MXdlPerWave,
    NXdlPerWave,
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
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    CDEBlockTransferScalarPerVector_NPerBlock,
    BlkGemmPipeSched,
    BlkGemmPipelineVer,
    AComputeDataType_,
    BComputeDataType_>>
{
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
    static constexpr int kAK1         = AK1;
    static constexpr int kBK1         = BK1;
    static constexpr int kMPerXDL     = MPerXDL;
    static constexpr int kNPerXDL     = NPerXDL;
    static constexpr int kMXdlPerWave = MXdlPerWave;
    static constexpr int kNXdlPerWave = NXdlPerWave;

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
    static constexpr int kCShuffleMXdlPerWavePerShuffle = CShuffleMXdlPerWavePerShuffle;
    static constexpr int kCShuffleNXdlPerWavePerShuffle = CShuffleNXdlPerWavePerShuffle;
    static constexpr auto kCThreadClusterLengths        = detail::SequenceToArray<
               CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>::value;
    static constexpr int kCBlockTransferScalarPerVector = CDEBlockTransferScalarPerVector_NPerBlock;

    // Pipeline configuration
    static constexpr ck::BlockGemmPipelineScheduler kPipelineScheduler = BlkGemmPipeSched;
    static constexpr ck::BlockGemmPipelineVersion kPipelineVersion     = BlkGemmPipelineVer;

    // Compute data types
    using AComputeDataType = AComputeDataType_;
    using BComputeDataType = BComputeDataType_;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3";

        // Template parameters in exact order matching InstanceTraits member order
        oss << "_" << kSpatialDim;                           // 1. NDimSpatial
        oss << "_" << detail::layout_name<ALayout>();        // 2. ALayout
        oss << "_" << detail::layout_name<BLayout>();        // 3. BLayout
        oss << "_" << detail::tuple_name<DsLayout>();        // 4. DsLayout
        oss << "_" << detail::layout_name<ELayout>();        // 5. ELayout
        oss << "_" << detail::type_name<ADataType>();        // 6. ADataType
        oss << "_" << detail::type_name<BDataType>();        // 7. BDataType
        oss << "_" << detail::type_name<AccDataType>();      // 8. AccDataType
        oss << "_" << detail::type_name<CShuffleDataType>(); // 9. CShuffleDataType
        oss << "_" << detail::tuple_name<DsDataType>();      // 10. DsDataType
        oss << "_" << detail::type_name<EDataType>();        // 11. EDataType
        oss << "_"
            << detail::elementwise_op_name<AElementwiseOperation>(); // 12. AElementwiseOperation
        oss << "_"
            << detail::elementwise_op_name<BElementwiseOperation>(); // 13. BElementwiseOperation
        oss << "_"
            << detail::elementwise_op_name<CDEElementwiseOperation>(); // 14.
                                                                       // CDEElementwiseOperation
        oss << "_"
            << detail::conv_fwd_spec_name(
                   kConvForwardSpecialization);                    // 15. ConvForwardSpecialization
        oss << "_" << detail::gemm_spec_name(kGemmSpecialization); // 16. GemmSpec
        oss << "_" << kBlockSize;                                  // 17. BlockSize
        oss << "_" << kMPerBlock;                                  // 18. MPerBlock
        oss << "_" << kNPerBlock;                                  // 19. NPerBlock
        oss << "_" << kKPerBlock;                                  // 20. KPerBlock
        oss << "_" << kAK1;                                        // 21. AK1
        oss << "_" << kBK1;                                        // 22. BK1
        oss << "_" << kMPerXDL;                                    // 23. MPerXDL
        oss << "_" << kNPerXDL;                                    // 24. NPerXDL
        oss << "_" << kMXdlPerWave;                                // 25. MXdlPerWave
        oss << "_" << kNXdlPerWave;                                // 26. NXdlPerWave
        oss << "_"
            << detail::array_to_string(
                   kAThreadClusterLengths); // 27. ABlockTransferThreadClusterLengths
        oss << "_"
            << detail::array_to_string(
                   kAThreadClusterArrangeOrder); // 28. ABlockTransferThreadClusterArrangeOrder
        oss << "_"
            << detail::array_to_string(
                   kABlockTransferSrcAccessOrder);       // 29. ABlockTransferSrcAccessOrder
        oss << "_" << kABlockTransferSrcVectorDim;       // 30. ABlockTransferSrcVectorDim
        oss << "_" << kABlockTransferSrcScalarPerVector; // 31. ABlockTransferSrcScalarPerVector
        oss << "_"
            << kABlockTransferDstScalarPerVectorK1; // 32. ABlockTransferDstScalarPerVector_AK1
        oss << "_" << kABlockLdsExtraM;             // 33. ABlockLdsExtraM
        oss << "_"
            << detail::array_to_string(
                   kBThreadClusterLengths); // 34. BBlockTransferThreadClusterLengths
        oss << "_"
            << detail::array_to_string(
                   kBThreadClusterArrangeOrder); // 35. BBlockTransferThreadClusterArrangeOrder
        oss << "_"
            << detail::array_to_string(
                   kBBlockTransferSrcAccessOrder);       // 36. BBlockTransferSrcAccessOrder
        oss << "_" << kBBlockTransferSrcVectorDim;       // 37. BBlockTransferSrcVectorDim
        oss << "_" << kBBlockTransferSrcScalarPerVector; // 38. BBlockTransferSrcScalarPerVector
        oss << "_"
            << kBBlockTransferDstScalarPerVectorK1;   // 39. BBlockTransferDstScalarPerVector_BK1
        oss << "_" << kBBlockLdsExtraN;               // 40. BBlockLdsExtraN
        oss << "_" << kCShuffleMXdlPerWavePerShuffle; // 41. CShuffleMXdlPerWavePerShuffle
        oss << "_" << kCShuffleNXdlPerWavePerShuffle; // 42. CShuffleNXdlPerWavePerShuffle
        oss << "_"
            << detail::array_to_string(
                   kCThreadClusterLengths); // 43. CDEBlockTransferClusterLengths
        oss << "_"
            << kCBlockTransferScalarPerVector; // 44. CDEBlockTransferScalarPerVector_NPerBlock
        oss << "_" << detail::pipeline_scheduler_name(kPipelineScheduler); // 45. BlkGemmPipeSched
        oss << "_" << detail::pipeline_version_name(kPipelineVersion);     // 46. BlkGemmPipelineVer
        oss << "_" << detail::type_name<AComputeDataType>();               // 47. AComputeDataType
        oss << "_" << detail::type_name<BComputeDataType>();               // 48. BComputeDataType

        return oss.str();
    }
};

} // namespace ck_tile::reflect
