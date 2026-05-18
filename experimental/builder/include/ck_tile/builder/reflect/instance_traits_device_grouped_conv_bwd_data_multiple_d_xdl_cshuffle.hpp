// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"

namespace ck::tensor_operation::device {

template <ck::index_t NDimSpatial,
          typename OutLayout,
          typename WeiLayout,
          typename DsLayout,
          typename InLayout,
          typename OutDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutComputeType,
          typename DsDataType,
          typename InDataType,
          typename OutElementwiseOperation,
          typename WeiElementwiseOperation,
          typename InElementwiseOperation,
          ck::tensor_operation::device::ConvolutionBackwardDataSpecialization
              ConvBackwardDataSpecialization,
          bool do_pad_gemm_m,
          bool do_pad_gemm_n,
          ck::index_t num_gemm_k_prefetch_stages,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          ck::index_t ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          ck::index_t BBlockLdsAddExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          ck::LoopScheduler LoopSched,
          typename ComputeTypeA,
          typename ComputeTypeB,
          ck::index_t max_transpose_transfer_src_scalar_per_vector,
          ck::index_t max_transpose_transfer_dst_scalar_per_vector>
struct DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1;

} // namespace ck::tensor_operation::device

namespace ck_tile::reflect {
/// @brief Tag type for DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle device kernel
struct DeviceGroupedConvBwdData_multiple_d_Xdl_CShuffle_Tag;

template <ck::index_t NDimSpatial,
          typename OutLayout_,
          typename WeiLayout_,
          typename DsLayout_,
          typename InLayout_,
          typename OutDataType_,
          typename WeiDataType_,
          typename AccDataType_,
          typename OutComputeType_,
          typename DsDataType_,
          typename InDataType_,
          typename OutElementwiseOperation_,
          typename WeiElementwiseOperation_,
          typename InElementwiseOperation_,
          ck::tensor_operation::device::ConvolutionBackwardDataSpecialization
              ConvBackwardDataSpecialization,
          bool do_pad_gemm_m,
          bool do_pad_gemm_n,
          ck::index_t num_gemm_k_prefetch_stages,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1_,
          typename ABlockTransferThreadClusterArrangeOrder_,
          typename ABlockTransferSrcAccessOrder_,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1_,
          typename BBlockTransferThreadClusterArrangeOrder_,
          typename BBlockTransferSrcAccessOrder_,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsAddExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
          ck::index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          ck::LoopScheduler LoopSched,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          ck::index_t max_transpose_transfer_src_scalar_per_vector,
          ck::index_t max_transpose_transfer_dst_scalar_per_vector>
struct InstanceTraits<
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<
        NDimSpatial,
        OutLayout_,
        WeiLayout_,
        DsLayout_,
        InLayout_,
        OutDataType_,
        WeiDataType_,
        AccDataType_,
        OutComputeType_,
        DsDataType_,
        InDataType_,
        OutElementwiseOperation_,
        WeiElementwiseOperation_,
        InElementwiseOperation_,
        ConvBackwardDataSpecialization,
        do_pad_gemm_m,
        do_pad_gemm_n,
        num_gemm_k_prefetch_stages,
        BlockSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1_,
        ABlockTransferThreadClusterArrangeOrder_,
        ABlockTransferSrcAccessOrder_,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1_,
        BBlockTransferThreadClusterArrangeOrder_,
        BBlockTransferSrcAccessOrder_,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        LoopSched,
        ComputeTypeA_,
        ComputeTypeB_,
        max_transpose_transfer_src_scalar_per_vector,
        max_transpose_transfer_dst_scalar_per_vector>>
{
    static constexpr auto kTensorOpName = "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle";

    /// @brief Tag type identifying this device kernel variant
    using device_kernel_tag = DeviceGroupedConvBwdData_multiple_d_Xdl_CShuffle_Tag;

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

    using InElementwiseOperation  = InElementwiseOperation_;
    using WeiElementwiseOperation = WeiElementwiseOperation_;
    using OutElementwiseOperation = OutElementwiseOperation_;

    static constexpr auto kConvBwdDataSpecialization = ConvBackwardDataSpecialization;

    static constexpr ck::index_t kBlockSize                     = BlockSize;
    static constexpr ck::index_t kMPerBlock                     = MPerBlock;
    static constexpr ck::index_t kNPerBlock                     = NPerBlock;
    static constexpr ck::index_t kK0PerBlock                    = K0PerBlock;
    static constexpr ck::index_t kAK1                           = AK1;
    static constexpr ck::index_t kBK1                           = BK1;
    static constexpr ck::index_t kMPerXDL                       = MPerXDL;
    static constexpr ck::index_t kNPerXDL                       = NPerXDL;
    static constexpr ck::index_t kMXdlPerWave                   = MXdlPerWave;
    static constexpr ck::index_t kNXdlPerWave                   = NXdlPerWave;
    static constexpr ck::index_t kCShuffleMXdlPerWavePerShuffle = CShuffleMXdlPerWavePerShuffle;
    static constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = CShuffleNXdlPerWavePerShuffle;
    static constexpr ck::index_t kCBlockTransferScalarPerVector_NWaveNPerXdl =
        CBlockTransferScalarPerVector_NWaveNPerXdl;
    static constexpr ck::index_t kMaxTransposeTransferSrcScalarPerVector =
        max_transpose_transfer_src_scalar_per_vector;
    static constexpr ck::index_t kMaxTransposeTransferDstScalarPerVector =
        max_transpose_transfer_dst_scalar_per_vector;
    static constexpr bool kDoPadGemmM           = do_pad_gemm_m;
    static constexpr bool kDoPadGemmN           = do_pad_gemm_n;
    static constexpr int kNumGemmKPrefetchStage = num_gemm_k_prefetch_stages;

    using ABlockTransferThreadClusterLengths_K0_M_K1 = ABlockTransferThreadClusterLengths_K0_M_K1_;
    using ABlockTransferThreadClusterArrangeOrder    = ABlockTransferThreadClusterArrangeOrder_;
    using ABlockTransferSrcAccessOrder               = ABlockTransferSrcAccessOrder_;
    // A block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kAThreadClusterLengths =
        detail::SequenceToArray<ABlockTransferThreadClusterLengths_K0_M_K1>::value;
    static constexpr auto kAThreadClusterArrangeOrder =
        detail::SequenceToArray<ABlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kABlockTransferSrcAccessOrder =
        detail::SequenceToArray<ABlockTransferSrcAccessOrder_>::value;
    static constexpr ck::index_t kABlockTransferSrcVectorDim = ABlockTransferSrcVectorDim;
    static constexpr ck::index_t kABlockTransferSrcScalarPerVector =
        ABlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kABlockTransferDstScalarPerVectorK1 =
        ABlockTransferDstScalarPerVector_AK1;
    static constexpr bool kABlockLdsExtraM = ABlockLdsAddExtraM;

    using BBlockTransferThreadClusterLengths_K0_N_K1 = BBlockTransferThreadClusterLengths_K0_N_K1_;
    using BBlockTransferThreadClusterArrangeOrder    = BBlockTransferThreadClusterArrangeOrder_;
    using BBlockTransferSrcAccessOrder               = BBlockTransferSrcAccessOrder_;
    // B block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kBThreadClusterLengths =
        detail::SequenceToArray<BBlockTransferThreadClusterLengths_K0_N_K1>::value;
    static constexpr auto kBThreadClusterArrangeOrder =
        detail::SequenceToArray<BBlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kBBlockTransferSrcAccessOrder =
        detail::SequenceToArray<BBlockTransferSrcAccessOrder_>::value;
    static constexpr ck::index_t kBBlockTransferSrcVectorDim = BBlockTransferSrcVectorDim;
    static constexpr ck::index_t kBBlockTransferSrcScalarPerVector =
        BBlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kBBlockTransferDstScalarPerVectorK1 =
        BBlockTransferDstScalarPerVector_BK1;
    static constexpr bool kBBlockLdsExtraN = BBlockLdsAddExtraN;

    using CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_;

    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;

    static constexpr ck::LoopScheduler kLoopScheduler = LoopSched;

    static constexpr auto kCThreadClusterLengths = detail::SequenceToArray<
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>::value;

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
        oss << "," << kDoPadGemmM;              // 15. GEMM padding for M dimension
        oss << "," << kDoPadGemmN;              // 16. GEMM padding for N dimension
        oss << "," << kNumGemmKPrefetchStage;   // 17. Number of GEMM K prefetch stages
        oss << "," << kBlockSize;               // 18. BlockSize
        oss << "," << kMPerBlock;               // 19. MPerBlock
        oss << "," << kNPerBlock;               // 20. NPerBlock
        oss << "," << kK0PerBlock;              // 21. K0PerBlock
        oss << "," << kAK1;                     // 22. AK1
        oss << "," << kBK1;                     // 23. BK1
        oss << "," << kMPerXDL;                 // 24. MPerXDL
        oss << "," << kNPerXDL;                 // 25. NPerXDL
        oss << "," << kMXdlPerWave;             // 26. MXdlPerWave
        oss << "," << kNXdlPerWave;             // 27. NXdlPerWave
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterLengths_K0_M_K1>(); // 28.
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterArrangeOrder>();    // 29.
        oss << "," << detail::sequence_name<ABlockTransferSrcAccessOrder>();               // 30.
        oss << "," << kABlockTransferSrcVectorDim;                                         // 31.
        oss << "," << kABlockTransferSrcScalarPerVector;                                   // 32.
        oss << "," << kABlockTransferDstScalarPerVectorK1;                                 // 33.
        oss << "," << (kABlockLdsExtraM ? "true" : "false");                               // 34.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterLengths_K0_N_K1>(); // 35.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterArrangeOrder>();    // 36.
        oss << "," << detail::sequence_name<BBlockTransferSrcAccessOrder>();               // 37.
        oss << "," << kBBlockTransferSrcVectorDim;                                         // 38.
        oss << "," << kBBlockTransferSrcScalarPerVector;                                   // 39.
        oss << "," << kBBlockTransferDstScalarPerVectorK1;                                 // 40.
        oss << "," << (kBBlockLdsExtraN ? "true" : "false");                               // 41.
        oss << "," << kCShuffleMXdlPerWavePerShuffle;                                      // 42.
        oss << "," << kCShuffleNXdlPerWavePerShuffle;                                      // 43.
        oss << ","
            << detail::sequence_name<
                   CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>(); // 44.
        oss << "," << kCBlockTransferScalarPerVector_NWaveNPerXdl;                    // 45.
        oss << "," << kNumGemmKPrefetchStage;                                         // 46.
        oss << "," << detail::loop_scheduler_name(kLoopScheduler); // 47. LoopSched
        oss << "," << detail::type_name<ComputeTypeA>();           // 48.
        oss << "," << detail::type_name<ComputeTypeB>();           // 49.
        oss << "," << kMaxTransposeTransferSrcScalarPerVector;     // 50.
        oss << "," << kMaxTransposeTransferDstScalarPerVector;     // 51.

        oss << ">";

        return oss.str();
    }
};

} // namespace ck_tile::reflect
