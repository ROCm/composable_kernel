// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"

namespace ck::tensor_operation::device {

template <index_t NDimSpatial,
          typename OutLayout,   // output image
          typename WeiLayout,   // weight
          typename DsLayout,    // bias
          typename InLayout,    // input image
          typename OutDataType, // output image
          typename WeiDataType, // weight
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,       // bias
          typename InDataType,       // input image
          typename OutElementwiseOp, // output image
          typename WeiElementwiseOp, // weight
          typename InElementwiseOp,  // C, bias, and input image
          ck::tensor_operation::device::ConvolutionBackwardDataSpecialization
              ConvBackwardDataSpecialization,
          bool DoPadGemmM,
          bool DoPadGemmN,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerWMMA,
          ck::index_t NPerWMMA,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA,
          typename ComputeTypeB,
          ck::index_t max_transpose_transfer_src_scalar_per_vector,
          ck::index_t max_transpose_transfer_dst_scalar_per_vector>
struct DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3;

} // namespace ck::tensor_operation::device

namespace ck_tile::reflect {
/// @brief Tag type for DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3 device kernel
struct DeviceGroupedConvBwdData_multiple_d_Wmma_CShuffle_V3_Tag;

template <index_t NDimSpatial,
          typename OutLayout_,   // output image
          typename WeiLayout_,   // weight
          typename DsLayout_,    // bias
          typename InLayout_,    // input image
          typename OutDataType_, // output image
          typename WeiDataType_, // weight
          typename AccDataType_,
          typename CShuffleDataType_,
          typename DsDataType_,       // bias
          typename InDataType_,       // input image
          typename OutElementwiseOp_, // output image
          typename WeiElementwiseOp_, // weight
          typename InElementwiseOp_,  // C, bias, and input image
          ck::tensor_operation::device::ConvolutionBackwardDataSpecialization
              ConvBackwardDataSpecialization,
          bool DoPadGemmM,
          bool DoPadGemmN,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerWMMA,
          ck::index_t NPerWMMA,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1_,
          typename ABlockTransferThreadClusterArrangeOrder_,
          typename ABlockTransferSrcAccessOrder_,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1_,
          typename BBlockTransferThreadClusterArrangeOrder_,
          typename BBlockTransferSrcAccessOrder_,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
          typename CDEShuffleBlockTransferScalarPerVector_NPerBlock_,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          ck::index_t max_transpose_transfer_src_scalar_per_vector,
          ck::index_t max_transpose_transfer_dst_scalar_per_vector>
struct InstanceTraits<
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<
        NDimSpatial,
        OutLayout_,   // output image
        WeiLayout_,   // weight
        DsLayout_,    // bias
        InLayout_,    // input image
        OutDataType_, // output image
        WeiDataType_, // weight
        AccDataType_,
        CShuffleDataType_,
        DsDataType_,       // bias
        InDataType_,       // input image
        OutElementwiseOp_, // output image
        WeiElementwiseOp_, // weight
        InElementwiseOp_,  // C, bias, and input image
        ConvBackwardDataSpecialization,
        DoPadGemmM,
        DoPadGemmN,
        BlockSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        AK1,
        BK1,
        MPerWMMA,
        NPerWMMA,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1_,
        ABlockTransferThreadClusterArrangeOrder_,
        ABlockTransferSrcAccessOrder_,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1_,
        BBlockTransferThreadClusterArrangeOrder_,
        BBlockTransferSrcAccessOrder_,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock_,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA_,
        ComputeTypeB_,
        max_transpose_transfer_src_scalar_per_vector,
        max_transpose_transfer_dst_scalar_per_vector>>
{
    static constexpr auto kTensorOpName = "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3";

    /// @brief Tag type identifying this device kernel variant
    using device_kernel_tag = DeviceGroupedConvBwdData_multiple_d_Wmma_CShuffle_V3_Tag;

    static constexpr ck::index_t kSpatialDim = NDimSpatial;

    using InLayout  = InLayout_;
    using WeiLayout = WeiLayout_;
    using OutLayout = OutLayout_;
    using DsLayout  = DsLayout_;

    using InDataType  = InDataType_;
    using WeiDataType = WeiDataType_;
    using OutDataType = OutDataType_;
    using AccDataType = AccDataType_;
    using DsDataType  = DsDataType_;

    using InElementwiseOperation  = InElementwiseOp_;
    using WeiElementwiseOperation = WeiElementwiseOp_;
    using OutElementwiseOperation = OutElementwiseOp_;

    static constexpr auto kConvBwdDataSpecialization = ConvBackwardDataSpecialization;

    static constexpr ck::index_t kBlockSize                 = BlockSize;
    static constexpr ck::index_t kMPerBlock                 = MPerBlock;
    static constexpr ck::index_t kNPerBlock                 = NPerBlock;
    static constexpr ck::index_t kK0PerBlock                = K0PerBlock;
    static constexpr ck::index_t kAK1                       = AK1;
    static constexpr ck::index_t kBK1                       = BK1;
    static constexpr ck::index_t kMPerWmma                  = MPerWMMA;
    static constexpr ck::index_t kNPerWmma                  = NPerWMMA;
    static constexpr ck::index_t kMRepeat                   = MRepeat;
    static constexpr ck::index_t kNRepeat                   = NRepeat;
    static constexpr ck::index_t kCShuffleMRepeatPerShuffle = CShuffleMRepeatPerShuffle;
    static constexpr ck::index_t kCShuffleNRepeatPerShuffle = CShuffleNRepeatPerShuffle;
    static constexpr ck::index_t kMaxTransposeTransferSrcScalarPerVector =
        max_transpose_transfer_src_scalar_per_vector;
    static constexpr ck::index_t kMaxTransposeTransferDstScalarPerVector =
        max_transpose_transfer_dst_scalar_per_vector;
    static constexpr bool kDoPadGemmM = DoPadGemmM;
    static constexpr bool kDoPadGemmN = DoPadGemmN;
    using CDEShuffleBlockTransferScalarPerVector_NPerBlock =
        CDEShuffleBlockTransferScalarPerVector_NPerBlock_;

    static constexpr auto kCDEShuffleBlockTransferScalarPerVector_NPerBlock =
        detail::SequenceToArray<CDEShuffleBlockTransferScalarPerVector_NPerBlock>::value;

    static constexpr ck::BlockGemmPipelineScheduler kBlkGemmPipeSched = BlkGemmPipeSched;
    static constexpr ck::BlockGemmPipelineVersion kBlkGemmPipelineVer = BlkGemmPipelineVer;

    using ABlockTransferThreadClusterLengths_AK0_M_AK1 =
        ABlockTransferThreadClusterLengths_AK0_M_AK1_;
    using ABlockTransferThreadClusterArrangeOrder = ABlockTransferThreadClusterArrangeOrder_;
    using ABlockTransferSrcAccessOrder            = ABlockTransferSrcAccessOrder_;
    // A block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kAThreadClusterLengths =
        detail::SequenceToArray<ABlockTransferThreadClusterLengths_AK0_M_AK1>::value;
    static constexpr auto kAThreadClusterArrangeOrder =
        detail::SequenceToArray<ABlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kABlockTransferSrcAccessOrder =
        detail::SequenceToArray<ABlockTransferSrcAccessOrder_>::value;

    static constexpr ck::index_t kABlockTransferSrcVectorDim = ABlockTransferSrcVectorDim;
    static constexpr ck::index_t kABlockTransferSrcScalarPerVector =
        ABlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kABlockTransferDstScalarPerVectorK1 =
        ABlockTransferDstScalarPerVector_AK1;
    static constexpr bool kABlockLdsExtraM = ABlockLdsExtraM;

    using BBlockTransferThreadClusterLengths_BK0_N_BK1 =
        BBlockTransferThreadClusterLengths_BK0_N_BK1_;
    using BBlockTransferThreadClusterArrangeOrder = BBlockTransferThreadClusterArrangeOrder_;
    using BBlockTransferSrcAccessOrder            = BBlockTransferSrcAccessOrder_;
    // B block transfer thread cluster dimensions (converted to std::array)
    // B block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kBThreadClusterLengths =
        detail::SequenceToArray<BBlockTransferThreadClusterLengths_BK0_N_BK1>::value;
    static constexpr auto kBThreadClusterArrangeOrder =
        detail::SequenceToArray<BBlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kBBlockTransferSrcAccessOrder =
        detail::SequenceToArray<BBlockTransferSrcAccessOrder_>::value;
    static constexpr ck::index_t kBBlockTransferSrcVectorDim = BBlockTransferSrcVectorDim;
    static constexpr ck::index_t kBBlockTransferSrcScalarPerVector =
        BBlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kBBlockTransferDstScalarPerVectorK1 =
        BBlockTransferDstScalarPerVector_BK1;
    static constexpr bool kBBlockLdsExtraN = BBlockLdsExtraN;

    using CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock =
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_;

    static constexpr auto kCDEThreadClusterLengths = detail::SequenceToArray<
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>::value;

    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;
    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << kTensorOpName;

        // Template parameters in exact order
        oss << "<" << kSpatialDim;                      // 1. NDimSpatial
        oss << "," << detail::layout_name<OutLayout>(); // 2. OutLayout
        oss << "," << detail::layout_name<WeiLayout>(); // 3. WeiLayout
        oss << "," << detail::tuple_name<DsLayout>();   // 4. DsLayout
        oss << "," << detail::layout_name<InLayout>();  // 5. InLayout
        oss << "," << detail::type_name<OutDataType>(); // 6. OutDataType
        oss << "," << detail::type_name<WeiDataType>(); // 7. WeiDataType
        oss << "," << detail::type_name<AccDataType>(); // 8. AccDataType
        oss << "," << detail::tuple_name<DsDataType>(); // 9. DsDataType
        oss << "," << detail::type_name<InDataType>();  // 10. InDataType
        oss << ","
            << detail::elementwise_op_name<OutElementwiseOperation>(); // 11.
                                                                       // OutElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<WeiElementwiseOperation>(); // 12.
                                                                       // WeiElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<InElementwiseOperation>(); // 13. InElementwiseOperation
        oss << ","
            << detail::conv_bwd_data_spec_name(
                   kConvBwdDataSpecialization); // 14. ConvBackwardDataSpecialization
        oss << "," << kDoPadGemmM;
        oss << "," << kDoPadGemmN;
        oss << "," << kBlockSize;  // 15. BlockSize
        oss << "," << kMPerBlock;  // 16. MPerBlock
        oss << "," << kNPerBlock;  // 17. NPerBlock
        oss << "," << kK0PerBlock; // 18. K0PerBlock
        oss << "," << kAK1;        // 19. ABK1
        oss << "," << kBK1;        // 19. ABK1
        oss << "," << kMPerWmma;   // 20. MPerWmma
        oss << "," << kNPerWmma;   // 21. NPerWmma
        oss << "," << kMRepeat;    // 22. MRepeat
        oss << "," << kNRepeat;    // 23. NRepeat
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterLengths_AK0_M_AK1>(); // 24.
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterArrangeOrder>();      // 25.
        oss << "," << detail::sequence_name<ABlockTransferSrcAccessOrder>();                 // 26.
        oss << "," << kABlockTransferSrcVectorDim;                                           // 27.
        oss << "," << kABlockTransferSrcScalarPerVector;                                     // 28.
        oss << "," << kABlockTransferDstScalarPerVectorK1;                                   // 29.
        oss << "," << (kABlockLdsExtraM ? "true" : "false");                                 // 30.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterLengths_BK0_N_BK1>(); // 31.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterArrangeOrder>();      // 32.
        oss << "," << detail::sequence_name<BBlockTransferSrcAccessOrder>();                 // 33.
        oss << "," << kBBlockTransferSrcVectorDim;                                           // 34.
        oss << "," << kBBlockTransferSrcScalarPerVector;                                     // 35.
        oss << "," << kBBlockTransferDstScalarPerVectorK1;                                   // 36.
        oss << "," << (kBBlockLdsExtraN ? "true" : "false");                                 // 37.
        oss << "," << kCShuffleMRepeatPerShuffle;                                            // 38.
        oss << "," << kCShuffleNRepeatPerShuffle;                                            // 39.
        oss << ","
            << detail::sequence_name<
                   CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>(); // 40.
        oss << "," << kCDEShuffleBlockTransferScalarPerVector_NPerBlock[0]; // 41.
        oss << "," << detail::pipeline_scheduler_name(kBlkGemmPipeSched);   // 43.
        oss << "," << detail::pipeline_version_name(kBlkGemmPipelineVer);   // 44.
        oss << "," << detail::type_name<ComputeTypeA>();                    // 45.
        oss << "," << detail::type_name<ComputeTypeB>();                    // 46.
        oss << "," << kMaxTransposeTransferSrcScalarPerVector;              // 47.
        oss << "," << kMaxTransposeTransferDstScalarPerVector;              // 48.

        oss << ">";

        return oss.str();
    }
};

} // namespace ck_tile::reflect
