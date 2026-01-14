// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::test {

namespace ckb = ck_tile::builder;

// Convenience struct for a tuple of m, n, and k values.
template <typename T>
struct MNK
{
    T m{};
    T n{};
    T k{};
};

// Specify thread block dimensions for a GEMM.
struct ThreadBlock
{
    // Thread block size.
    size_t block_size;
    // Size of the submatrix problem in a thread block.
    MNK<size_t> tile_size;
};
static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

struct WarpGemmParams
{
    MatrixInstructionType matrix_instruction;
    size_t gemm_m_per_instruction      = 0;
    size_t gemm_n_per_instruction      = 0;
    size_t gemm_m_iters_per_wave       = 0;
    size_t gemm_n_iters_per_wave       = 0;
};
static_assert(ckb::WarpGemmDescriptor<WarpGemmParams>);

struct GemmPipeline
{
    size_t num_gemm_k_prefetch_stages{1};
    size_t num_conv_groups_to_merge{1};
    PipelineVersion pipeline_version;
    PipelineScheduler scheduler{PipelineScheduler::DEFAULT};
};
static_assert(ckb::GemmPipelineDescriptor<GemmPipeline>);

// Describe input tensor thread cluster lengths.
template <size_t ThreadClusterRank = 3>
struct InputThreadCluster
{
    size_t k0;
    size_t m_n;
    size_t k1;
    size_t k_batch_size;
};

// Specialization for ThreadClusterRank == 3
template <>
struct InputThreadCluster<3>
{
    size_t k0;
    size_t m_n;
    size_t k1;
};
static_assert(ckb::InputTileThreadClusterDescriptor<InputThreadCluster<3>, 3>);
static_assert(ckb::InputTileThreadClusterDescriptor<InputThreadCluster<4>, 4>);

// Describe C block transfer thread cluster lengths.
struct OutputThreadCluster
{
    size_t gemm_m_block_size;
    size_t gemm_m_per_block;
    size_t gemm_n_block_size;
    size_t gemm_n_per_block;
};
static_assert(OutputTileThreadClusterDescriptor<OutputThreadCluster>);

struct LdsTransfer
{
    size_t global_memory_vector_load_size;
    size_t src_vector_dim;
    size_t src_scalar_per_vector;
    size_t lds_dst_scalar_per_vector;
    bool is_direct_load;
    bool lds_padding;
};
static_assert(LdsTransferDescriptor<LdsTransfer>);

struct Epilogue
{
    size_t m_xdl_per_wave_per_shuffle;
    size_t n_per_wave_per_shuffle;
    size_t scalar_per_vector;
};
static_assert(EpilogueDescriptor<Epilogue>);

template <size_t ThreadClusterRank = 3>
struct AccessOrder
{
    std::array<size_t, ThreadClusterRank> order;
};
static_assert(AccessOrderDescriptor<AccessOrder<>>);
static_assert(AccessOrderDescriptor<AccessOrder<4>>);

template <size_t ThreadClusterRank = 3>
struct InputTileTransfer
{
    InputThreadCluster<ThreadClusterRank> thread_cluster;
    LdsTransfer lds_transfer;
    AccessOrder<ThreadClusterRank> thread_cluster_access_order;
    AccessOrder<ThreadClusterRank> src_access_order;
};

struct OutputTileTransfer
{
    OutputThreadCluster thread_cluster;
    Epilogue epilogue;
};

template <size_t ThreadClusterRank = 3>
struct InputOutputTileTransfer
{
    InputTileTransfer<ThreadClusterRank> a;
    InputTileTransfer<ThreadClusterRank> b;
    OutputTileTransfer c;
};

// DL-specific descriptors
struct DlThreadConfig
{
    size_t k0_per_block;
    size_t k1;
    size_t m1_per_thread;
    size_t n1_per_thread;
    size_t k_per_thread;
};
static_assert(ckb::DlThreadConfigDescriptor<DlThreadConfig>);

struct DlThreadCluster
{
    std::array<size_t, 2> m1_xs;
    std::array<size_t, 2> n1_xs;
};
static_assert(ckb::DlThreadClusterDescriptor<DlThreadCluster>);

template <size_t D = 4>
struct DlBlockTransfer
{
    std::array<size_t, D> thread_slice_lengths;
    std::array<size_t, D> thread_cluster_lengths;
    std::array<size_t, D> thread_cluster_arrange_order;
    std::array<size_t, D> src_access_order;
    std::array<size_t, D> src_vector_tensor_lengths;
    std::array<size_t, D> src_vector_tensor_contiguous_dim_order;
    std::array<size_t, D> dst_vector_tensor_lengths;
};
static_assert(ckb::DlBlockTransferDescriptor4D<DlBlockTransfer<4>>);
static_assert(ckb::DlBlockTransferDescriptor5D<DlBlockTransfer<5>>);

struct DlEpilogue
{
    std::array<size_t, 6> src_dst_access_order;
    size_t src_dst_vector_dim;
    size_t dst_scalar_per_vector;
};
static_assert(ckb::DlEpilogueDescriptor<DlEpilogue>);

// Factory types

struct ThreadBlock_
{
    ThreadBlock thread_block;
};

struct WarpGemm_
{
    WarpGemmParams warp_gemm;
};

template <size_t ThreadClusterRank = 3>
struct InputOutputTileTransfer_
{
    InputOutputTileTransfer<ThreadClusterRank> transfer;
};

struct ConvSpecializationFwd_
{
    ConvSpecialization fwd_specialization;
    GemmSpecialization gemm_specialization;
};

struct ConvSpecializationBwdWeight_
{
    ConvSpecialization bwd_weight_specialization;
};

struct TransposeParams_
{
    size_t max_transpose_transfer_src_scalar_per_vector{1};
    size_t max_transpose_transfer_dst_scalar_per_vector{1};
};

struct GemmPipeline_
{
    GemmPipeline gemm_pipeline;
};

struct DlThreadConfig_
{
    DlThreadConfig thread_config;
};

struct DlThreadCluster_
{
    DlThreadCluster thread_cluster;
};

template <size_t Dim = 4>
struct DlTransfer
{
    DlBlockTransfer<Dim> a;
    DlBlockTransfer<Dim> b;
    DlEpilogue c;
};

template <size_t Dim = 4>
struct DlTransfer_
{
    DlTransfer<Dim> transfer;
};

template <ConvAlgorithmSpecialization Specialization = ConvAlgorithmSpecialization::NONE>
struct AlgorithmSpecialization_
{
    static constexpr ConvAlgorithmSpecialization specialization = Specialization;
};

// Specify thread block dimensions for a GEMM (CK Tile).
struct TileThreadBlock
{
    // Size of the submatrix problem in a thread block.
    MNK<size_t> tile_size;
};
static_assert(ckb::TileThreadBlockDescriptor<TileThreadBlock>);

struct TileTransfer
{
    size_t a_scalar_per_vector;
    size_t b_scalar_per_vector;
    size_t c_scalar_per_vector;
};
static_assert(ckb::TileTransferDescriptor<TileTransfer>);

struct TileBlockGemm
{
    // Number of warps per each dimension.
    MNK<int> warps;
    // Number of data processed per each dimension for each XDL/WMMA instruction.
    MNK<int> warp_tile;
    // Double LDS buffer.
    bool double_smem_buffer;
    // Waves grouping (Ping-Pong scheduler).
    int num_wave_groups;
    PipelineVersion pipeline_version;
    PipelineScheduler scheduler;
};
static_assert(ckb::TileBlockGemmDescriptor<TileBlockGemm>);

struct TileOptimizations
{
    // Number of convolution groups processed per one workgroup
    int num_groups_to_merge;
    // Split image for large tensors
    bool split_image;
    // Explicit gemm for 1x1, stride=0, pad=0 cases
    bool explicit_gemm;
};
static_assert(ckb::TileOptimizationsDescriptor<TileOptimizations>);

struct TileConvSpecialization_
{
    TileConvSpecialization specialization;
};

struct TileThreadBlock_
{
    TileThreadBlock thread_block;
};

struct TileTransfer_
{
    TileTransfer transfer;
};

struct TileBlockGemm_
{
    TileBlockGemm block_gemm;
};

struct TileOptimizations_
{
    TileOptimizations optimizations;
};

// Factory

template <typename... Components>
struct ConvAlgorithmTemplate : Components...
{

    template <typename TB>
    constexpr auto with_thread_block(const TB& tb) const
    {
        static_assert(std::is_base_of_v<ThreadBlock_, ConvAlgorithmTemplate>);
        auto result         = *this;
        result.thread_block = tb;
        return result;
    }

    template <typename GemmConfig>
    constexpr auto with_gemm_config(const GemmConfig& gemm) const
    {
        auto result = *this;
        static_assert(std::is_base_of_v<WarpGemm_, ConvAlgorithmTemplate>);
        result.warp_gemm = gemm;
        return result;
    }

    template <typename T>
    constexpr auto with_transfer(const T& t) const
    {
        static_assert(std::is_base_of_v<InputOutputTileTransfer_<3>, ConvAlgorithmTemplate> ||
                      std::is_base_of_v<InputOutputTileTransfer_<4>, ConvAlgorithmTemplate>);
        auto result     = *this;
        result.transfer = t;
        return result;
    }

    constexpr auto with_fwd_specializations(ConvSpecialization fwd_spec,
                                            GemmSpecialization gemm_spec) const
    {
        static_assert(std::is_base_of_v<ConvSpecializationFwd_, ConvAlgorithmTemplate>);
        auto result                = *this;
        result.fwd_specialization  = fwd_spec;
        result.gemm_specialization = gemm_spec;
        return result;
    }

    constexpr auto with_bwd_specialization(ConvSpecialization bwd_spec) const
    {
        static_assert(std::is_base_of_v<ConvSpecializationBwdWeight_, ConvAlgorithmTemplate>);
        auto result                      = *this;
        result.bwd_weight_specialization = bwd_spec;
        return result;
    }

    constexpr auto with_transpose_params(size_t max_src_scalar_per_vector,
                                         size_t max_dst_scalar_per_vector) const
    {
        static_assert(std::is_base_of_v<TransposeParams_, ConvAlgorithmTemplate>);
        auto result                                         = *this;
        result.max_transpose_transfer_src_scalar_per_vector = max_src_scalar_per_vector;
        result.max_transpose_transfer_dst_scalar_per_vector = max_dst_scalar_per_vector;
        return result;
    }

    constexpr auto with_num_conv_groups_to_merge(size_t num_groups_to_merge) const
    {
        static_assert(std::is_base_of_v<GemmPipeline_, ConvAlgorithmTemplate>);
        auto result                     = *this;
        result.gemm_pipeline.num_conv_groups_to_merge = num_groups_to_merge;
        return result;
    }

    constexpr auto with_num_gemm_k_prefetch_stages(size_t num_prefetch_stages) const
    {
        static_assert(std::is_base_of_v<GemmPipeline_, ConvAlgorithmTemplate>);
        auto result                     = *this;
        result.gemm_pipeline.num_gemm_k_prefetch_stages = num_prefetch_stages;
        return result;
    }

    template <typename PL>
    constexpr auto with_gemm_pipeline(const PL& pl) const
    {
        static_assert(std::is_base_of_v<GemmPipeline_, ConvAlgorithmTemplate>);
        auto result                = *this;
        result.gemm_pipeline = pl;
        return result;
    }

    constexpr auto with_gemm_pipeline(const PipelineVersion plv) const
    {
        static_assert(std::is_base_of_v<GemmPipeline_, ConvAlgorithmTemplate>);
        auto result             = *this;
        result.gemm_pipeline.pipeline_version = plv;
        return result;
    }

    constexpr auto with_gemm_pipeline(const PipelineScheduler sch) const
    {
        static_assert(std::is_base_of_v<GemmPipeline_, ConvAlgorithmTemplate>);
        auto result             = *this;
        result.gemm_pipeline.scheduler = sch;
        return result;
    }

    constexpr auto with_gemm_pipeline(const PipelineVersion plv, const PipelineScheduler sch) const
    {
        static_assert(std::is_base_of_v<GemmPipeline_, ConvAlgorithmTemplate>);
        auto result             = *this;
        result.gemm_pipeline.pipeline_version = plv;
        result.gemm_pipeline.scheduler        = sch;
        return result;
    }

    template <typename TC>
    constexpr auto with_dl_thread_config(const TC& tc) const
    {
        static_assert(std::is_base_of_v<DlThreadConfig_, ConvAlgorithmTemplate>);
        auto result          = *this;
        result.thread_config = tc;
        return result;
    }

    template <typename TCl>
    constexpr auto with_dl_thread_cluster(const TCl& tcl) const
    {
        static_assert(std::is_base_of_v<DlThreadCluster_, ConvAlgorithmTemplate>);
        auto result           = *this;
        result.thread_cluster = tcl;
        return result;
    }

    template <typename T>
    constexpr auto with_dl_transfer(const T& t) const
    {
        static_assert(std::is_base_of_v<DlTransfer_<4>, ConvAlgorithmTemplate> ||
                      std::is_base_of_v<DlTransfer_<5>, ConvAlgorithmTemplate>);
        auto result     = *this;
        result.transfer = t;
        return result;
    }

    template <typename S>
    constexpr auto with_tile_specializations(const S& s) const
    {
        static_assert(std::is_base_of_v<TileConvSpecialization_, ConvAlgorithmTemplate>);
        auto result           = *this;
        result.specialization = s;
        return result;
    }

    template <typename TB>
    constexpr auto with_tile_thread_block(const TB& tb) const
    {
        static_assert(std::is_base_of_v<TileThreadBlock_, ConvAlgorithmTemplate>);
        auto result         = *this;
        result.thread_block = tb;
        return result;
    }

    template <typename BG>
    constexpr auto with_tile_block_gemm(const BG& bg) const
    {
        static_assert(std::is_base_of_v<TileBlockGemm_, ConvAlgorithmTemplate>);
        auto result       = *this;
        result.block_gemm = bg;
        return result;
    }

    template <typename T>
    constexpr auto with_tile_transfer(const T& t) const
    {
        static_assert(std::is_base_of_v<TileTransfer_, ConvAlgorithmTemplate>);
        auto result     = *this;
        result.transfer = t;
        return result;
    }

    template <typename O>
    constexpr auto with_tile_optimizations(const O& o) const
    {
        static_assert(std::is_base_of_v<TileOptimizations_, ConvAlgorithmTemplate>);
        auto result          = *this;
        result.optimizations = o;
        return result;
    }
};

// Fwd algorithm types

using enum ckb::ConvAlgorithmSpecialization;

// Covers both XDL and WMMA variants for generic fwd convolution
using ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationFwd_,
                          GemmPipeline_,
                          AlgorithmSpecialization_<>>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationFwd_,
                          GemmPipeline_,
                          AlgorithmSpecialization_<PIPELINE_V3>>;

using ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK =
    ConvAlgorithmTemplate<ThreadBlock_,
                          ConvSpecializationFwd_,
                          DlThreadConfig_,
                          DlThreadCluster_,
                          DlTransfer_<>>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationFwd_,
                          GemmPipeline_,
                          AlgorithmSpecialization_<LARGE_TENSOR | MULTIPLE_D>>;

// CK Tile algorithm
using ConvAlgorithm_Tile_GroupedConvolutionKernel = ConvAlgorithmTemplate<TileThreadBlock_,
                                                                          TileBlockGemm_,
                                                                          TileTransfer_,
                                                                          TileConvSpecialization_,
                                                                          TileOptimizations_>;

// Reference algorithm descriptor - for GPU reference validation
using  ConvAlgorithm_Reference = ConvAlgorithmTemplate<AlgorithmSpecialization_<REFERENCE>>;

// Bwd weight algorithm types
using ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<4>,
                          ConvSpecializationBwdWeight_,
                          TransposeParams_,
                          AlgorithmSpecialization_<>>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationBwdWeight_,
                          GemmPipeline_,
                          AlgorithmSpecialization_<>>;

// Covers both XDL and WMMA variants
using ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationBwdWeight_,
                          GemmPipeline_,
                          TransposeParams_,
                          AlgorithmSpecialization_<TWO_STAGE | PIPELINE_V3>>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationBwdWeight_,
                          GemmPipeline_,
                          AlgorithmSpecialization_<PIPELINE_V3>>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationBwdWeight_,
                          GemmPipeline_,
                          TransposeParams_,
                          AlgorithmSpecialization_<PIPELINE_V3>>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Dl =
    ConvAlgorithmTemplate<ThreadBlock_,
                          DlThreadConfig_,
                          DlThreadCluster_,
                          DlTransfer_<5>,
                          ConvSpecializationBwdWeight_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<4>,
                          ConvSpecializationBwdWeight_,
                          AlgorithmSpecialization_<MULTIPLE_D>>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationBwdWeight_,
                          GemmPipeline_,
                          TransposeParams_,
                          AlgorithmSpecialization_<TWO_STAGE | PIPELINE_V3>>;

using ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WarpGemm_,
                          InputOutputTileTransfer_<>,
                          ConvSpecializationBwdWeight_,
                          GemmPipeline_,
                          AlgorithmSpecialization_<MULTIPLE_D | PIPELINE_V3>>;

} // namespace ck_tile::builder::test
