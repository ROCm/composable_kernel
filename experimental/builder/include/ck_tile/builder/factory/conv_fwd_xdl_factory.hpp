// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_limits.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tuning_params.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_block_transfer.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_thread_block.hpp"

namespace ck_tile::builder::factory {

// Factory for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE>
struct ConvFwdXdlFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts                       = internal::ConvTensorLayouts<SIGNATURE>;
    using Types                         = internal::ConvTensorDataTypes<SIGNATURE>;
    using Ops                           = internal::ConvElementwiseOps<SIGNATURE>;
    using AlgorithmType                 = decltype(ALGORITHM);

    static constexpr auto FWD_CONV_SPECIALIZATION = internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION     = internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                       .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = internal::SetLoopScheduler<ALGORITHM>();
    static constexpr auto BLOCK          = internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = ALGORITHM.gridwise_gemm;
    static constexpr auto XDL_PARAMS     = GRIDWISE_GEMM.xdl_params;
    static constexpr auto A_BLOCK_TRANSFER =
        internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER = internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();

    // Check limits for the algorithm parameters.
    static_assert(ValidABlockTransfer<A_BLOCK_TRANSFER,
                                      Types::input_types.first,
                                      sizeof(typename Types::InDataType),
                                      BLOCK.block_size,
                                      BLOCK.per_block>);
    static_assert(A_BLOCK_TRANSFER.src_vector_dim == 2 ||
                  (ALGORITHM.num_conv_groups_to_merge > 1 && A_BLOCK_TRANSFER.src_vector_dim == 1));
    static_assert(ValidBBlockTransfer<B_BLOCK_TRANSFER,
                                      Types::weight_types.first,
                                      sizeof(typename Types::WeiDataType),
                                      BLOCK.block_size,
                                      BLOCK.per_block>);
    static_assert(B_BLOCK_TRANSFER.src_vector_dim == 2);
    static_assert(ValidCBlockTransfer<C_BLOCK_TRANSFER,
                                      Types::output_types.first,
                                      BLOCK.block_size,
                                      BLOCK.per_block>);

    // Layout validations (same as DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle)
    using enum TensorLayout;
    static_assert(IsValidLayout<SIGNATURE.input.config.layout,
                                G_NW_C_strided,
                                G_NHW_C_strided,
                                G_NDHW_C_strided,
                                GNWC,
                                GNHWC,
                                GNDHWC,
                                NWGC,
                                NHWGC,
                                NDHWGC,
                                NGCW,
                                NGCHW,
                                NGCDHW>);

    static_assert(IsValidLayout<SIGNATURE.weight.config.layout,
                                G_K_X_C_strided,
                                G_K_YX_C_strided,
                                G_K_ZYX_C_strided,
                                GKXC,
                                GKYXC,
                                GKZYXC,
                                KXGC,
                                KYXGC,
                                KZYXGC,
                                GKCX,
                                GKCYX,
                                GKCZYX>);

    static_assert(IsValidLayout<SIGNATURE.output.config.layout,
                                G_NW_K_strided,
                                G_NHW_K_strided,
                                G_NDHW_K_strided,
                                GNWK,
                                GNHWK,
                                GNDHWK,
                                NWGK,
                                NHWGK,
                                NDHWGK,
                                NGKW,
                                NGKHW,
                                NGKDHW>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        SPATIAL_DIM,
        typename Layouts::InLayout,
        typename Layouts::WeiLayout,
        typename Layouts::DsLayout,
        typename Layouts::OutLayout,
        typename Types::InDataType,
        typename Types::WeiDataType,
        typename Types::AccDataType,
        typename Types::OutComputeType,
        typename Types::DsDataType,
        typename Types::OutDataType,
        typename Ops::InElementwiseOp,
        typename Ops::WeiElementwiseOp,
        typename Ops::OutElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        ALGORITHM.num_gemm_k_prefetch_stages,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.ak1,
        GRIDWISE_GEMM.bk1,
        XDL_PARAMS.m_per_xdl,
        XDL_PARAMS.n_per_xdl,
        XDL_PARAMS.m_xdl_per_wave,
        XDL_PARAMS.n_xdl_per_wave,
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
        C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        typename Types::InComputeType,
        typename Types::WeiComputeType,
        LOOP_SCHEDULER,
        ALGORITHM.num_conv_groups_to_merge>;
};

} // namespace ck_tile::builder::factory
