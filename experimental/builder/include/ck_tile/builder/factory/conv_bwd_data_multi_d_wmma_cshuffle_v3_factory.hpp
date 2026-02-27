// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp"
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

// Factory for DeviceGroupedConvBwdDataMultipleD_wmma_CShuffle_v3 instance
// of a grouped bwd Data convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsBackwardData<SIGNATURE>
struct ConvBwdDataMultiDWmmaV3Factory
{
    static constexpr int SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts                    = internal::ConvTensorLayouts<SIGNATURE>;
    using Types                      = internal::ConvTensorDataTypes<SIGNATURE>;
    using Ops                        = internal::ConvElementwiseOps<SIGNATURE>;
    using AlgorithmType              = decltype(ALGORITHM);

    static constexpr auto BWD_CONV_SPECIALIZATION =
        internal::SetBwdDataConvSpecialization<ALGORITHM>();

    static constexpr auto LOOP_SCHEDULER = internal::SetLoopScheduler<ALGORITHM>();
    static constexpr auto BLOCK          = internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        internal::SetBwdConvBlockTransfer<ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        internal::SetBwdConvBlockTransfer<ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER = internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();
    static constexpr auto BLOCK_GEMM       = internal::SetBlockGemm<ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits3D<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits3D<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits3D<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits3D<B_BLOCK_TRANSFER.src_access_order>);

    // The backward convolution kernel class instance.
    using Instance =
        ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<
            SPATIAL_DIM,
            typename Layouts::OutLayout,
            typename Layouts::WeiLayout,
            typename Layouts::DsLayout,
            typename Layouts::InLayout,
            typename Types::OutDataType,
            typename Types::WeiDataType,
            typename Types::AccDataType,
            typename Types::OutComputeType,
            typename Types::DsDataType,
            typename Types::InDataType,
            typename Ops::OutElementwiseOp,
            typename Ops::WeiElementwiseOp,
            typename Ops::InElementwiseOp,
            BWD_CONV_SPECIALIZATION,
            ALGORITHM.DoPadGemmM,
            ALGORITHM.DoPadGemmN,
            BLOCK.block_size,
            BLOCK.per_block.m,
            BLOCK.per_block.n,
            BLOCK.per_block.k,
            GRIDWISE_GEMM.ak1,
            GRIDWISE_GEMM.bk1,
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
            A_BLOCK_TRANSFER.lds_padding,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
            B_BLOCK_TRANSFER.src_vector_dim,
            B_BLOCK_TRANSFER.src_scalar_per_vector,
            B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
            B_BLOCK_TRANSFER.lds_padding,
            C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
            to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
            ck::Sequence<C_BLOCK_TRANSFER.scalar_per_vector,
                         C_BLOCK_TRANSFER.scalar_per_vector,
                         C_BLOCK_TRANSFER.scalar_per_vector>,
            BLOCK_GEMM.scheduler,
            BLOCK_GEMM.pipeline_version,
            typename Types::OutComputeType,
            typename Types::InComputeType,
            ALGORITHM.max_transpose_transfer_src_scalar_per_vector,
            ALGORITHM.max_transpose_transfer_dst_scalar_per_vector>;
};

} // namespace ck_tile::builder::factory
