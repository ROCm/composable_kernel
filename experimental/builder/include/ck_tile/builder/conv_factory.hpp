// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A factory for instantiating CK convolution kernels.
//
// This file translates a semantic description of a convolution operation
// (`ConvSignatureDescriptor` and `ConvAlgorithmDescriptor`) into specific,
// low-level template arguments required by the underlying CK device-level
// kernel implementations. This abstraction enables more complex build
// time logic and simplifies the kernel specification.
//
// Key Components:
//
// Template Metaprogram:
//  - ConvFactory: The main factory, with specializations for different
//                 convolution directions (currently only forward).
//
// Template Metaprogram Helpers:
//  - ConvTensorLayouts: Maps layout enums to CK layout types for different
//                       spatial dimensions (2D/3D) and directions.
//  - ConvTensorTypes:   Maps data type enums (FP16, BF16, FP32) to C++ types used by CK.
//  - ConvPassThroughOps: Hard-coded pass-through element-wise operations.
//  - ConvSpec:          Encapsulates convolution and GEMM specialization enums.
//
// `constexpr` Helper Functions:
//  - SetThreadBlockInfo:           Determines thread block dimensions and tile sizes.
//  - SetConvTuningInfo:            Sets XDL and AK1/BK1 tuning parameters.
//  - SetFwdConvBlockTransfer:      Configures A/B tensor block transfer parameters.
//  - SetCBlockTransfer:            Configures C tensor block transfer parameters.
//  - SetBlockGemmPipelineVersion:  Maps pipeline version enum to CK types.
//
// The primary entry point is the `ConvFactory` struct, which is currently
// specialized for forward convolutions and produces instances of
// DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3.

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
// WORKAROUND: Macro namespace collision in upstream CK device operation headers.
// device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp (line 41) and
// device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp (line 51) both define
// GridwiseGemmTemplateParameters macro without #undef, causing redefinition errors.
// Use pragma push/pop to isolate the Large_Tensor header's macro scope.
#pragma push_macro("GridwiseGemmTemplateParameters")
#ifdef GridwiseGemmTemplateParameters
#undef GridwiseGemmTemplateParameters
#endif
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp"
#pragma pop_macro("GridwiseGemmTemplateParameters")
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_limits.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/versions.hpp"

#include "ck_tile/builder/conv_signature_utils.hpp"
#include "ck_tile/builder/conv_layout_utils.hpp"
#include "ck_tile/builder/conv_data_type_utils.hpp"
#include "ck_tile/builder/conv_elementwise_op_utils.hpp"

namespace ck_tile::builder::factory_internal {

template<size_t SPATIAL_DIM>
consteval auto get_input_layout_value(ConvInputLayout layout) {
    if constexpr (SPATIAL_DIM == 1) return layout._1d;
    else if constexpr (SPATIAL_DIM == 2) return layout._2d;
    else if constexpr (SPATIAL_DIM == 3) return layout._3d;
    else static_assert(false, "Unsupported spatial dimension");
}

template<size_t SPATIAL_DIM>
consteval auto get_weight_layout_value(ConvWeightLayout layout) {
    if constexpr (SPATIAL_DIM == 1) return layout._1d;
    else if constexpr (SPATIAL_DIM == 2) return layout._2d;
    else if constexpr (SPATIAL_DIM == 3) return layout._3d;
    else static_assert(false, "Unsupported spatial dimension");
}

template<size_t SPATIAL_DIM>
consteval auto get_output_layout_value(ConvOutputLayout layout) {
    if constexpr (SPATIAL_DIM == 1) return layout._1d;
    else if constexpr (SPATIAL_DIM == 2) return layout._2d;
    else if constexpr (SPATIAL_DIM == 3) return layout._3d;
    else static_assert(false, "Unsupported spatial dimension");
}

// The algorithm specializations for the convolution and GEMM.
template <typename CONV_ENUM>
    requires(
        std::is_same_v<CONV_ENUM, ck::tensor_operation::device::ConvolutionForwardSpecialization>)
struct ConvSpec
{
    CONV_ENUM conv_spec;
    ck::tensor_operation::device::GemmSpecialization gemm_spec;
};

// Deduction guide for ConvSpec to simplify brace initialization.
template <typename CONV_ENUM, typename GEMM_ENUM>
ConvSpec(CONV_ENUM, GEMM_ENUM) -> ConvSpec<CONV_ENUM>;

struct BlockGemmSpec
{
    ck::BlockGemmPipelineVersion pipeline_version;
    ck::BlockGemmPipelineScheduler scheduler;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval BlockGemmSpec SetBlockGemm()
{
    constexpr auto& BG = ALGORITHM.block_gemm;

    ck::BlockGemmPipelineScheduler scheduler;
    ck::BlockGemmPipelineVersion version;

    switch(BG.scheduler)
    {
    case PipelineScheduler::INTRAWAVE: scheduler = ck::BlockGemmPipelineScheduler::Intrawave; break;
    case PipelineScheduler::INTERWAVE: scheduler = ck::BlockGemmPipelineScheduler::Interwave; break;
    case PipelineScheduler::DEFAULT: throw "Block GEMM scheduler must be Intrawave or Interwave.";
    default: throw "Unknown PipelineScheduler";
    }

    switch(BG.pipeline_version)
    {
    case PipelineVersion::V1: version = ck::BlockGemmPipelineVersion::v1; break;
    case PipelineVersion::V2: version = ck::BlockGemmPipelineVersion::v2; break;
    case PipelineVersion::V3: version = ck::BlockGemmPipelineVersion::v3; break;
    case PipelineVersion::V4: version = ck::BlockGemmPipelineVersion::v4; break;
    case PipelineVersion::V5: version = ck::BlockGemmPipelineVersion::v5; break;
    case PipelineVersion::WEIGHT_ONLY:
        throw "PipelineVersion::WEIGHT_ONLY is not supported for block GEMM.";
    default: throw "Unknown PipelineVersion";
    }

    return BlockGemmSpec{.pipeline_version = version, .scheduler = scheduler};
}

// Block info for a convolution.
struct MNK
{
    size_t m{};
    size_t n{};
    size_t k{};
};
struct ConvBlock
{
    size_t block_size = 0;
    MNK per_block     = {};
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ConvBlock SetThreadBlockInfo()
{
    constexpr auto& TB = ALGORITHM.thread_block;
    return ConvBlock{.block_size = TB.block_size,
                     .per_block  = {.m = TB.tile_size.m, .n = TB.tile_size.n, .k = TB.tile_size.k}};
}

// Block transfer parameters for A or B tensor.
struct BlockTransfer
{
    ck::Array<size_t, 3> thread_cluster_dims  = {0, 0, 0}; // k0, m, k1
    ck::Array<size_t, 3> thread_cluster_order = {0, 0, 0};
    ck::Array<size_t, 3> src_access_order     = {0, 0, 0};
    size_t src_vector_dim                     = 0;
    size_t src_scalar_per_vector              = 0;
    size_t lds_dst_scalar_per_vector          = 0;
    bool is_direct_load                       = false;
    bool lds_padding                          = false;
};

template <auto TRANSFER>
constexpr BlockTransfer SetFwdConvBlockTransfer()
{
    constexpr auto& TCL = TRANSFER.block_transfer;
    constexpr auto& TCO = TRANSFER.block_transfer_access_order;
    constexpr auto& SAO = TRANSFER.src_access_order;
    constexpr auto& LDS = TRANSFER.lds_transfer;

    BlockTransfer block_transfer{.thread_cluster_dims  = {TCL.k0, TCL.m_n, TCL.k1},
                                 .thread_cluster_order = {TCO.order[0], TCO.order[1], TCO.order[2]},
                                 .src_access_order     = {SAO.order[0], SAO.order[1], SAO.order[2]},
                                 .src_vector_dim       = LDS.src_vector_dim,
                                 .src_scalar_per_vector     = LDS.src_scalar_per_vector,
                                 .lds_dst_scalar_per_vector = LDS.lds_dst_scalar_per_vector,
                                 .is_direct_load            = LDS.is_direct_load,
                                 .lds_padding               = LDS.lds_padding};
    return block_transfer;
}

// Block transfer parameters for C tensor.
struct CBlockTransfer
{
    size_t m_per_wave_per_shuffle            = 0;
    size_t n_per_wave_per_shuffle            = 0;
    ck::Array<size_t, 4> thread_cluster_dims = {0, 0, 0, 0};
    size_t scalar_per_vector                 = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr CBlockTransfer SetCBlockTransfer()
{
    constexpr auto& TCL = ALGORITHM.transfer.c.thread_cluster_dims;
    constexpr auto& EPC = ALGORITHM.transfer.c.epilogue;
    CBlockTransfer block_transfer{.m_per_wave_per_shuffle = EPC.m_per_wave_per_shuffle,
                                  .n_per_wave_per_shuffle = EPC.n_per_wave_per_shuffle,
                                  .thread_cluster_dims =
                                      {
                                          TCL.m_block,
                                          TCL.m_wave_per_xdl,
                                          TCL.n_block,
                                          TCL.n_wave_per_xdl,
                                      },
                                  .scalar_per_vector = EPC.scalar_per_vector};
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::LoopScheduler SetLoopScheduler()
{
    constexpr auto loop_scheduler = ALGORITHM.loop_scheduler;
    using ck_loop_sched           = ck::LoopScheduler;
    switch(loop_scheduler)
    {
    case PipelineScheduler::DEFAULT: return ck_loop_sched::Default;
    case PipelineScheduler::INTERWAVE: return ck_loop_sched::Interwave;
    case PipelineScheduler::INTRAWAVE: throw "LoopScheduler must be either DEFAULT or INTERWAVE.";
    default: throw "Unknown PipelineScheduler";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::PipelineVersion SetGridwiseGemmPipelineVersion()
{
    constexpr auto pipeline_version = ALGORITHM.gridwise_gemm.pipeline_version;
    using ck_pipeline               = ck::PipelineVersion;
    switch(pipeline_version)
    {
    case PipelineVersion::V1: return ck_pipeline::v1;
    case PipelineVersion::V2: return ck_pipeline::v2;
    case PipelineVersion::V3: throw "PipelineVersion::V3 is used only for stream-K.";
    case PipelineVersion::V4: return ck_pipeline::v4;
    case PipelineVersion::V5: throw "PipelineVersion::V5 cannot be used for gridwise GEMM.";
    case PipelineVersion::WEIGHT_ONLY: return ck_pipeline::weight_only;
    default: throw "Unknown GridwiseGemmPipelineVersion";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::tensor_operation::device::GemmSpecialization SetGemmSpecialization()
{
    constexpr auto gemm_spec = ALGORITHM.gemm_specialization;
    using ck_gemm_spec       = ck::tensor_operation::device::GemmSpecialization;

    switch(gemm_spec)
    {
    case GemmSpecialization::Default: return ck_gemm_spec::Default;
    case GemmSpecialization::MPadding: return ck_gemm_spec::MPadding;
    case GemmSpecialization::NPadding: return ck_gemm_spec::NPadding;
    case GemmSpecialization::KPadding: return ck_gemm_spec::KPadding;
    case GemmSpecialization::MNPadding: return ck_gemm_spec::MNPadding;
    case GemmSpecialization::MKPadding: return ck_gemm_spec::MKPadding;
    case GemmSpecialization::NKPadding: return ck_gemm_spec::NKPadding;
    case GemmSpecialization::MNKPadding: return ck_gemm_spec::MNKPadding;
    case GemmSpecialization::OPadding: return ck_gemm_spec::OPadding;
    case GemmSpecialization::MOPadding: return ck_gemm_spec::MOPadding;
    case GemmSpecialization::NOPadding: return ck_gemm_spec::NOPadding;
    case GemmSpecialization::KOPadding: return ck_gemm_spec::KOPadding;
    case GemmSpecialization::MNOPadding: return ck_gemm_spec::MNOPadding;
    case GemmSpecialization::MKOPadding: return ck_gemm_spec::MKOPadding;
    case GemmSpecialization::NKOPadding: return ck_gemm_spec::NKOPadding;
    case GemmSpecialization::MNKOPadding: return ck_gemm_spec::MNKOPadding;
    default: throw "Unknown GemmSpecialization";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::BlockGemmPipelineVersion SetBlockGemmPipelineVersion()
{
    constexpr auto version = ALGORITHM.pipeline_version;
    using ck_pipeline      = ck::BlockGemmPipelineVersion;
    switch(version)
    {
    case PipelineVersion::V1: return ck_pipeline::v1;
    case PipelineVersion::V2: return ck_pipeline::v2;
    case PipelineVersion::V3: return ck_pipeline::v3;
    case PipelineVersion::V4: return ck_pipeline::v4;
    case PipelineVersion::V5: return ck_pipeline::v5;
    default: throw "Unknown block GEMM PipelineVersion";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::tensor_operation::device::ConvolutionForwardSpecialization SetFwdConvSpecialization()
{
    constexpr auto specialization = ALGORITHM.fwd_specialization;
    using ck_conv_spec            = ck::tensor_operation::device::ConvolutionForwardSpecialization;
    switch(specialization)
    {
    case ConvFwdSpecialization::DEFAULT: return ck_conv_spec::Default;
    case ConvFwdSpecialization::FILTER_1X1_PAD0: return ck_conv_spec::Filter1x1Pad0;
    case ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0: return ck_conv_spec::Filter1x1Stride1Pad0;
    case ConvFwdSpecialization::FILTER_3x3: return ck_conv_spec::Filter3x3;
    default: throw "Unknown ConvFwdSpecialization";
    }
}

} // namespace ck_tile::builder::factory_internal

namespace ck_tile::builder {

// Primary template for the convolution factory.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          auto VERSION>
struct ConvFactory
{
    // This will trigger if a specialization for the given convolution direction is not found.
    // We should always catch this in an earlier validation check.
    static_assert(false, "Unsupported device operation.");
};

// Factory specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<decltype(ALGORITHM)>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using AuxiliaryLayouts = decltype(factory_internal::GetAuxiliaryTensorLayouts<SIGNATURE,
                                                                                 SPATIAL_DIM,
                                                                                 ConvDirection::FORWARD>());
                                                                
    using Types         = factory_internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops           = decltype(factory_internal::GetElementwiseOps<SIGNATURE>());
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(ALGORITHM.transfer.a.lds_transfer.is_direct_load ==
                      ALGORITHM.transfer.b.lds_transfer.is_direct_load,
                  "A and B block transfers must both be direct load or not.");

    static constexpr bool IS_DIRECT_LOAD = ALGORITHM.transfer.a.lds_transfer.is_direct_load;
    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto BLOCK         = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM = ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();
    static constexpr auto BLOCK_GEMM = factory_internal::SetBlockGemm<ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        SPATIAL_DIM,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename AuxiliaryLayouts::DsLayout,
        typename Layouts::ELayout,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        typename Types::CShuffleDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.ak1,
        GRIDWISE_GEMM.bk1,
        GRIDWISE_GEMM.m_per_xdl,
        GRIDWISE_GEMM.n_per_xdl,
        GRIDWISE_GEMM.m_xdl_per_wave,
        GRIDWISE_GEMM.n_xdl_per_wave,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
        A_BLOCK_TRANSFER.src_vector_dim,
        A_BLOCK_TRANSFER.src_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(A_BLOCK_TRANSFER.lds_padding),
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
        B_BLOCK_TRANSFER.src_vector_dim,
        B_BLOCK_TRANSFER.src_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(B_BLOCK_TRANSFER.lds_padding),
        C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        BLOCK_GEMM.scheduler,
        BLOCK_GEMM.pipeline_version,
        typename Types::AComputeType,
        typename Types::BComputeType,
        IS_DIRECT_LOAD>;
};

// Factory specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<std::remove_const_t<decltype(ALGORITHM)>>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using AuxiliaryLayouts = decltype(factory_internal::GetAuxiliaryTensorLayouts<SIGNATURE,
                                                                                 SPATIAL_DIM,
                                                                                 ConvDirection::FORWARD>());
                                                                
    using Types         = factory_internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops           = decltype(factory_internal::GetElementwiseOps<SIGNATURE>());
    using AlgorithmType = decltype(ALGORITHM);

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = factory_internal::SetLoopScheduler<ALGORITHM>();
    static constexpr auto BLOCK          = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        SPATIAL_DIM,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename AuxiliaryLayouts::DsLayout,
        typename Layouts::ELayout,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        typename Types::CShuffleDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        ALGORITHM.num_gemm_k_prefetch_stages,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.ak1,
        GRIDWISE_GEMM.bk1,
        GRIDWISE_GEMM.m_per_xdl,
        GRIDWISE_GEMM.n_per_xdl,
        GRIDWISE_GEMM.m_xdl_per_wave,
        GRIDWISE_GEMM.n_xdl_per_wave,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
        A_BLOCK_TRANSFER.src_vector_dim,
        A_BLOCK_TRANSFER.src_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(A_BLOCK_TRANSFER.lds_padding),
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
        B_BLOCK_TRANSFER.src_vector_dim,
        B_BLOCK_TRANSFER.src_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(B_BLOCK_TRANSFER.lds_padding),
        C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        typename Types::AComputeType,
        typename Types::BComputeType,
        LOOP_SCHEDULER,
        ALGORITHM.num_groups_to_merge>;
};

// Factory specialization for DeviceGroupedConvFwdMultipleD_Wmma_CShuffle instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle<std::remove_const_t<decltype(ALGORITHM)>>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using AuxiliaryLayouts = decltype(factory_internal::GetAuxiliaryTensorLayouts<SIGNATURE,
                                                                                 SPATIAL_DIM,
                                                                                 ConvDirection::FORWARD>());
                                                                
    using Types         = factory_internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops           = decltype(factory_internal::GetElementwiseOps<SIGNATURE>());
    using AlgorithmType = decltype(ALGORITHM);

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = factory_internal::SetLoopScheduler<ALGORITHM>();
    static constexpr auto BLOCK          = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = ALGORITHM.gridwise_gemm;
    static constexpr auto GRIDWISE_GEMM_PIPELINE_VERSION =
        factory_internal::SetGridwiseGemmPipelineVersion<ALGORITHM>();
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<
        SPATIAL_DIM,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename AuxiliaryLayouts::DsLayout,
        typename Layouts::ELayout,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        typename Types::CShuffleDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        ALGORITHM.num_gemm_k_prefetch_stages,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.k1,
        GRIDWISE_GEMM.m_per_wmma,
        GRIDWISE_GEMM.n_per_wmma,
        GRIDWISE_GEMM.m_wmma_per_wave,
        GRIDWISE_GEMM.n_wmma_per_wave,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
        A_BLOCK_TRANSFER.src_vector_dim,
        A_BLOCK_TRANSFER.src_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(A_BLOCK_TRANSFER.lds_padding),
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
        B_BLOCK_TRANSFER.src_vector_dim,
        B_BLOCK_TRANSFER.src_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(B_BLOCK_TRANSFER.lds_padding),
        C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        LOOP_SCHEDULER,
        GRIDWISE_GEMM_PIPELINE_VERSION>;
};

// Factory specialization for DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK instance
// of a grouped forward convolution kernel using Direct Load (DL) approach.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> && DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
                                                      std::remove_const_t<decltype(ALGORITHM)>>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using AuxiliaryLayouts = decltype(factory_internal::GetAuxiliaryTensorLayouts<SIGNATURE,
                                                                                 SPATIAL_DIM,
                                                                                 ConvDirection::FORWARD>());
                                                                
    using Types         = factory_internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops           = decltype(factory_internal::GetElementwiseOps<SIGNATURE>());
    using AlgorithmType = decltype(ALGORITHM);

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();

    static constexpr auto BLOCK = factory_internal::SetThreadBlockInfo<ALGORITHM>();

    // DL-specific parameters from algorithm descriptor
    static constexpr auto DL_THREAD_CFG      = ALGORITHM.thread_config;
    static constexpr ck::index_t K0PerBlock  = DL_THREAD_CFG.k0_per_block;
    static constexpr ck::index_t K1          = DL_THREAD_CFG.k1;
    static constexpr ck::index_t M1PerThread = DL_THREAD_CFG.m1_per_thread;
    static constexpr ck::index_t N1PerThread = DL_THREAD_CFG.n1_per_thread;
    static constexpr ck::index_t KPerThread  = DL_THREAD_CFG.k_per_thread;

    // Thread cluster from descriptor
    static constexpr auto DL_CLUSTER = ALGORITHM.thread_cluster;
    using M1N1ThreadClusterM1Xs      = to_sequence_v<DL_CLUSTER.m1_xs>;
    using M1N1ThreadClusterN1Xs      = to_sequence_v<DL_CLUSTER.n1_xs>;

    // A Block Transfer from descriptor - K0_M0_M1_K1 tensor format
    static constexpr auto DL_A_TRANSFER = ALGORITHM.transfer.a.block_transfer;
    using ABlockTransferThreadSliceLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.thread_slice_lengths>;
    using ABlockTransferThreadClusterLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.thread_cluster_lengths>;
    using ABlockTransferThreadClusterArrangeOrder =
        to_sequence_v<DL_A_TRANSFER.thread_cluster_arrange_order>;
    using ABlockTransferSrcAccessOrder = to_sequence_v<DL_A_TRANSFER.src_access_order>;
    using ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.src_vector_tensor_lengths>;
    using ABlockTransferSrcVectorTensorContiguousDimOrder =
        to_sequence_v<DL_A_TRANSFER.src_vector_tensor_contiguous_dim_order>;
    using ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.dst_vector_tensor_lengths>;

    // B Block Transfer from descriptor - K0_N0_N1_K1 tensor format
    static constexpr auto DL_B_TRANSFER = ALGORITHM.transfer.b.block_transfer;
    using BBlockTransferThreadSliceLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.thread_slice_lengths>;
    using BBlockTransferThreadClusterLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.thread_cluster_lengths>;
    using BBlockTransferThreadClusterArrangeOrder =
        to_sequence_v<DL_B_TRANSFER.thread_cluster_arrange_order>;
    using BBlockTransferSrcAccessOrder = to_sequence_v<DL_B_TRANSFER.src_access_order>;
    using BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.src_vector_tensor_lengths>;
    using BBlockTransferSrcVectorTensorContiguousDimOrder =
        to_sequence_v<DL_B_TRANSFER.src_vector_tensor_contiguous_dim_order>;
    using BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.dst_vector_tensor_lengths>;

    // C Thread Transfer from descriptor
    static constexpr auto DL_C_TRANSFER    = ALGORITHM.transfer.c.epilogue;
    using CThreadTransferSrcDstAccessOrder = to_sequence_v<DL_C_TRANSFER.src_dst_access_order>;
    static constexpr ck::index_t CThreadTransferSrcDstVectorDim = DL_C_TRANSFER.src_dst_vector_dim;
    static constexpr ck::index_t CThreadTransferDstScalarPerVector =
        DL_C_TRANSFER.dst_scalar_per_vector;

    // The DL forward convolution kernel class instance
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
        SPATIAL_DIM,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Types::AccDataType,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename AuxiliaryLayouts::DsLayout,
        typename Layouts::ELayout,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        FWD_CONV_SPECIALIZATION,
        GEMM_SPECIALIZATION,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        K0PerBlock,
        K1,
        M1PerThread,
        N1PerThread,
        KPerThread,
        M1N1ThreadClusterM1Xs,
        M1N1ThreadClusterN1Xs,
        ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
        ABlockTransferSrcVectorTensorContiguousDimOrder,
        ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
        BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
        BBlockTransferSrcVectorTensorContiguousDimOrder,
        BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
        CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector>;
};

// Factory specialization for DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor instance
// of a grouped forward convolution kernel with large tensor support (N-splitting).
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
                 std::remove_const_t<decltype(ALGORITHM)>>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using AuxiliaryLayouts = decltype(factory_internal::GetAuxiliaryTensorLayouts<SIGNATURE,
                                                                                 SPATIAL_DIM,
                                                                                 ConvDirection::FORWARD>());
                                                                
    using Types         = factory_internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops           = decltype(factory_internal::GetElementwiseOps<SIGNATURE>());
    using AlgorithmType = decltype(ALGORITHM);

    static constexpr auto BASE_ALGORITHM = ALGORITHM.base_algorithm;

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<BASE_ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<BASE_ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = factory_internal::SetLoopScheduler<BASE_ALGORITHM>();
    static constexpr auto BLOCK          = factory_internal::SetThreadBlockInfo<BASE_ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = BASE_ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<BASE_ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBlockTransfer<BASE_ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, BASE_ALGORITHM>();

    // Check limits for the algorithm parameters.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance with large tensor support.
    using Instance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
            SPATIAL_DIM,
            typename Layouts::ALayout,
            typename Layouts::BLayout,
            typename AuxiliaryLayouts::DsLayout,
            typename Layouts::ELayout,
            typename Types::ADataType,
            typename Types::BDataType,
            typename Types::AccDataType,
            typename Types::CShuffleDataType,
            typename Types::DsDataTypes,
            typename Types::EDataType,
            typename Ops::AElementwiseOp,
            typename Ops::BElementwiseOp,
            typename Ops::CDEElementwiseOp,
            SPECIALIZATION.conv_spec,
            SPECIALIZATION.gemm_spec,
            BASE_ALGORITHM.num_gemm_k_prefetch_stages,
            BLOCK.block_size,
            BLOCK.per_block.m,
            BLOCK.per_block.n,
            BLOCK.per_block.k,
            GRIDWISE_GEMM.ak1,
            GRIDWISE_GEMM.bk1,
            GRIDWISE_GEMM.m_per_xdl,
            GRIDWISE_GEMM.n_per_xdl,
            GRIDWISE_GEMM.m_xdl_per_wave,
            GRIDWISE_GEMM.n_xdl_per_wave,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
            A_BLOCK_TRANSFER.src_vector_dim,
            A_BLOCK_TRANSFER.src_scalar_per_vector,
            A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
            static_cast<ck::index_t>(A_BLOCK_TRANSFER.lds_padding),
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
            B_BLOCK_TRANSFER.src_vector_dim,
            B_BLOCK_TRANSFER.src_scalar_per_vector,
            B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
            static_cast<ck::index_t>(B_BLOCK_TRANSFER.lds_padding),
            C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
            to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
            C_BLOCK_TRANSFER.scalar_per_vector,
            typename Types::AComputeType,
            typename Types::BComputeType,
            LOOP_SCHEDULER>;
};

} // namespace ck_tile::builder
