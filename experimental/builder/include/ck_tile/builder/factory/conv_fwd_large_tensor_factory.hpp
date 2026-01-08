// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp"
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

// Factory for DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE>
struct ConvFwdLargeTensorFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts = internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;
    using Types   = internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops     = internal::ElementwiseOps<SIGNATURE>;
    using AlgorithmType = decltype(ALGORITHM);

    static constexpr auto BASE_ALGORITHM = ALGORITHM.base_algorithm;

    static constexpr auto FWD_CONV_SPECIALIZATION =
        internal::SetFwdConvSpecialization<BASE_ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION = internal::SetGemmSpecialization<BASE_ALGORITHM>();
    static constexpr internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                       .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = internal::SetLoopScheduler<BASE_ALGORITHM>();
    static constexpr auto BLOCK          = internal::SetThreadBlockInfo<BASE_ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = BASE_ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        internal::SetFwdConvBlockTransfer<BASE_ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        internal::SetFwdConvBlockTransfer<BASE_ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER =
        internal::SetCBlockTransfer<SIGNATURE, BASE_ALGORITHM>();

    // Check limits for the data transfer parameters.
    static_assert(ValidABlockTransfer<A_BLOCK_TRANSFER,
                                      typename Types::ADataType,
                                      BLOCK.block_size,
                                      BLOCK.per_block>);
    static_assert(ValidBBlockTransfer<B_BLOCK_TRANSFER,
                                      typename Types::BDataType,
                                      BLOCK.block_size,
                                      BLOCK.per_block>);
    static_assert(ValidCBlockTransfer<C_BLOCK_TRANSFER,
                                      typename Types::EDataType,
                                      BLOCK.block_size,
                                      BLOCK.per_block>);

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
                                NDHWGC> &&
                  A_BLOCK_TRANSFER.src_vector_dim == 2);

    static_assert(IsValidLayout<SIGNATURE.weight.config.layout,
                                G_K_X_C_strided,
                                G_K_YX_C_strided,
                                G_K_ZYX_C_strided,
                                GKXC,
                                GKYXC,
                                GKZYXC,
                                KXGC,
                                KYXGC,
                                KZYXGC> &&
                  B_BLOCK_TRANSFER.src_vector_dim == 2);

    static_assert(IsValidLayout<SIGNATURE.output.config.layout,
                                G_NW_K_strided,
                                G_NHW_K_strided,
                                G_NDHW_K_strided,
                                GNWK,
                                GNHWK,
                                GNDHWK,
                                NWGK,
                                NHWGK,
                                NDHWGK>);

    // The forward convolution kernel class instance with large tensor support.
    using Instance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
            SPATIAL_DIM,
            typename Layouts::ALayout,
            typename Layouts::BLayout,
            typename Layouts::DsLayout,
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
            C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
            to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
            C_BLOCK_TRANSFER.scalar_per_vector,
            typename Types::AComputeType,
            typename Types::BComputeType,
            LOOP_SCHEDULER>;
};

} // namespace ck_tile::builder::factory
