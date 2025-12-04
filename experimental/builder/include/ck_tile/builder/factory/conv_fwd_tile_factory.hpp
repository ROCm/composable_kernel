// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_limits.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/conv_signature_utils.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_tensor_layout.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_tensor_type.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_elementwise_op.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_tuning_params.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_block_transfer.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_thread_block.hpp"

namespace ck_tile::builder::factory {

// Factory for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE>
struct ConvFwdTileFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts                       = decltype(internal::GetTileTensorLayout<SIGNATURE.layout,
                                                                                 SPATIAL_DIM,
                                                                                 ConvDirection::FORWARD>());
    using Types                         = internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops           = internal::ElementwiseOps<get_elementwise_operation<SIGNATURE>()>;
    using AlgorithmType = decltype(ALGORITHM);

    static constexpr auto FWD_CONV_SPECIALIZATION =
        internal::SetTileFwdConvSpecialization<ALGORITHM>();
    static constexpr auto BLOCK             = internal::SetTileThreadBlockInfo<ALGORITHM>();
    static constexpr auto BLOCK_GEMM        = internal::SetTileBlockGemm<ALGORITHM>();
    static constexpr auto SCALAR_PER_VECTOR = internal::SetTileBlockTransfer<ALGORITHM.transfer>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(TileInputOutputVectorTransferLimits<SCALAR_PER_VECTOR>);

    using GroupedConvTraitsType = ck_tile::GroupedConvTraits<SPATIAL_DIM,
                                                             FWD_CONV_SPECIALIZATION,
                                                             typename Layouts::ALayout,
                                                             typename Layouts::BLayout,
                                                             typename Layouts::DsLayout,
                                                             typename Layouts::ELayout,
                                                             SCALAR_PER_VECTOR.a,
                                                             SCALAR_PER_VECTOR.b,
                                                             SCALAR_PER_VECTOR.c,
                                                             ALGORITHM.num_groups_to_merge,
                                                             ALGORITHM.split_image,
                                                             ALGORITHM.explicit_gemm>;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<BLOCK.per_block.m, BLOCK.per_block.n, BLOCK.per_block.k>,
        ck_tile::sequence<BLOCK_GEMM.warps.m, BLOCK_GEMM.warps.n, BLOCK_GEMM.warps.k>,
        ck_tile::sequence<BLOCK_GEMM.warp_tile.m, BLOCK_GEMM.warp_tile.n, BLOCK_GEMM.warp_tile.k>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
        GroupedConvTraitsType::FixedGemmParams::kPadM,
        GroupedConvTraitsType::FixedGemmParams::kPadN,
        GroupedConvTraitsType::FixedGemmParams::kPadK,
        BLOCK_GEMM.double_smem_buffer,
        typename GroupedConvTraitsType::AsLayoutFwd,
        typename GroupedConvTraitsType::BsLayoutFwd,
        typename GroupedConvTraitsType::CLayoutFwd,
        GroupedConvTraitsType::FixedGemmParams::TransposeC,
        GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
        GroupedConvTraitsType::FixedGemmParams::Persistent,
        BLOCK_GEMM.num_wave_groups>;

    template <bool has_hot_loop, ck_tile::TailNumber tail_number>
    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        GemmShape,
        GemmUniversalTraits,
        BLOCK_GEMM.loop_scheduler,
        has_hot_loop,
        tail_number,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Types::EDataType,
        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
        GroupedConvTraitsType::VectorSizeA,
        GroupedConvTraitsType::VectorSizeB>;

    template <bool has_hot_loop, ck_tile::TailNumber tail_number>
    using GemmPipeline =
        typename internal::TilePipelineType<BLOCK_GEMM.pipeline_version>::template GemmPipeline<
            UniversalGemmProblem<has_hot_loop, tail_number>>;

    template <ck_tile::memory_operation_enum memory_operation>
    using ConvEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<typename Types::ADataType,
                                         typename Types::BDataType,
                                         typename Types::DsDataTypes,
                                         typename Types::AccDataType,
                                         typename Types::EDataType,
                                         typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                                         typename GroupedConvTraitsType::FixedGemmParams::ELayout,
                                         typename Ops::CDEElementwiseOp,
                                         BLOCK.per_block.m,
                                         BLOCK.per_block.n,
                                         BLOCK_GEMM.warps.m,
                                         BLOCK_GEMM.warps.n,
                                         BLOCK_GEMM.warp_tile.m,
                                         BLOCK_GEMM.warp_tile.n,
                                         BLOCK_GEMM.warp_tile.k,
                                         GroupedConvTraitsType::FixedGemmParams::TransposeC,
                                         memory_operation,
                                         BLOCK_GEMM.num_wave_groups,
                                         GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                                         SCALAR_PER_VECTOR.c>>;

    template <bool has_hot_loop,
              ck_tile::TailNumber tail_number,
              ck_tile::memory_operation_enum memory_operation>
    using Instance =
        ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                 TilePartitioner,
                                                 GemmPipeline<has_hot_loop, tail_number>,
                                                 ConvEpilogue<memory_operation>>;
};

} // namespace ck_tile::builder::factory
