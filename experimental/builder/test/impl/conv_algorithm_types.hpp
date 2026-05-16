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

struct XdlParams
{
    size_t m_per_xdl      = 0;
    size_t n_per_xdl      = 0;
    size_t m_xdl_per_wave = 0;
    size_t n_xdl_per_wave = 0;
};
static_assert(ckb::GridwiseXdlGemmDescriptor<XdlParams>);

// Describe gridwise XDL GEMM parameters.
struct GridwiseFwdXdlGemm
{
    // NOTE: ak1 and bk1 are difficult to verify in the kernel instantiation!!!
    size_t ak1 = 0;
    size_t bk1 = 0;
    XdlParams xdl_params;
};
static_assert(ckb::GridwiseFwdXdlGemmDescriptor<GridwiseFwdXdlGemm>);

struct GridwiseBwdXdlGemm
{
    size_t k1 = 0;
    XdlParams xdl_params;
};
static_assert(ckb::GridwiseBwdXdlGemmDescriptor<GridwiseBwdXdlGemm>);

struct GridwiseBwdDataXdlGemm
{
    size_t ak1 = 0;
    size_t bk1 = 0;
    XdlParams xdl_params;
};

// Describe gridwise WMMA GEMM parameters.
struct GridwiseWmmaGemm
{
    size_t k1              = 0;
    size_t m_per_wmma      = 0;
    size_t n_per_wmma      = 0;
    size_t m_wmma_per_wave = 0;
    size_t n_wmma_per_wave = 0;
};
static_assert(ckb::GridwiseWmmaGemmDescriptor<GridwiseWmmaGemm>);
struct GridwiseWmmaGemmABK1
{
    size_t ak1             = 0;
    size_t bk1             = 0;
    size_t m_per_wmma      = 0;
    size_t n_per_wmma      = 0;
    size_t m_wmma_per_wave = 0;
    size_t n_wmma_per_wave = 0;
};
static_assert(ckb::GridwiseWmmaGemmDescriptor<GridwiseWmmaGemmABK1>);

struct BlockGemmPipeline
{
    PipelineVersion pipeline_version;
    PipelineScheduler scheduler;
};
static_assert(ckb::BlockGemmPipelineDescriptor<BlockGemmPipeline>);

// Describe Aand B block transfer thread cluster lengths.
template <size_t ThreadSliceLength = 3>
struct BlockTransfer
{
    size_t k0;
    size_t m_n;
    size_t k1;
    size_t k_batch_size;
};

// Specialization for ThreadSliceLength == 3
template <>
struct BlockTransfer<3>
{
    size_t k0;
    size_t m_n;
    size_t k1;
};
static_assert(ckb::BlockTransferDescriptor<BlockTransfer<3>, 3>);
static_assert(ckb::BlockTransferDescriptor<BlockTransfer<4>, 4>);

// Describe C block transfer thread cluster lengths.
struct ThreadCluster
{
    size_t m_block;
    size_t m_wave_per_xdl;
    size_t n_block;
    size_t n_wave_per_xdl;
};
static_assert(ThreadClusterDescriptor<ThreadCluster>);

struct LdsTransfer
{
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
    size_t n_xdl_per_wave_per_shuffle;
    size_t scalar_per_vector;
};
static_assert(EpilogueDescriptor<Epilogue>);

template <size_t ThreadSliceLength = 3>
struct AccessOrder
{
    std::array<size_t, ThreadSliceLength> order;
};
static_assert(ThreadClusterOrderDescriptor<AccessOrder<>>);
static_assert(ThreadClusterOrderDescriptor<AccessOrder<4>>);

template <size_t ThreadSliceLength = 3>
struct InputTransfer
{
    BlockTransfer<ThreadSliceLength> block_transfer;
    LdsTransfer lds_transfer;
    AccessOrder<ThreadSliceLength> thread_cluster_arrange_order;
    AccessOrder<ThreadSliceLength> src_access_order;
};

struct OutputTransfer
{
    ThreadCluster thread_cluster_dims;
    Epilogue epilogue;
};

template <size_t ThreadSliceLength = 3>
struct Transfer
{
    InputTransfer<ThreadSliceLength> a;
    InputTransfer<ThreadSliceLength> b;
    OutputTransfer c;
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

struct FwdXdlGemm_
{
    GridwiseFwdXdlGemm gridwise_gemm;
};

struct BwdXdlGemm_
{
    GridwiseBwdXdlGemm gridwise_gemm;
};

struct BwdDataXdlGemm_
{
    GridwiseBwdDataXdlGemm gridwise_gemm;
};

struct WmmaGemm_
{
    GridwiseWmmaGemm gridwise_gemm;
};

struct WmmaGemmABK1_
{
    GridwiseWmmaGemmABK1 gridwise_gemm;
};

template <size_t ThreadSliceLength = 3>
struct Transfer_
{
    Transfer<ThreadSliceLength> transfer;
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

struct ConvSpecializationBwdData_
{
    ConvSpecialization bwd_data_specialization;
};

struct Prefetch_
{
    size_t num_gemm_k_prefetch_stages;
    PipelineScheduler loop_scheduler;
};

struct GemmPad_
{
    size_t DoPadGemmM;
    size_t DoPadGemmN;
};

struct TransposeParams_
{
    size_t max_transpose_transfer_src_scalar_per_vector{1};
    size_t max_transpose_transfer_dst_scalar_per_vector{1};
};

struct GemmBatchOptions_
{
    size_t num_conv_groups_to_merge{1};
};

struct BlockGemm_
{
    BlockGemmPipeline block_gemm_pipeline;
};

struct GridGemm_
{
    PipelineVersion pipeline_version;
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

struct TwoStageSpecialization_
{
    static constexpr ConvAlgorithmSpecialization specialization =
        ConvAlgorithmSpecialization::TWO_STAGE;
};

struct MultipleDSpecialization_
{
    static constexpr ConvAlgorithmSpecialization specialization =
        ConvAlgorithmSpecialization::MULTIPLE_D;
};

struct LargeTensorSpecialization_
{
    static constexpr ConvAlgorithmSpecialization specialization =
        ConvAlgorithmSpecialization::LARGE_TENSOR;
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
    // Two-stage kernels
    bool two_stage;
    // StreamK work distribution
    ckb::StreamKConfig streamk = ckb::StreamKConfig::disabled();
};
static_assert(ckb::TileOptimizationsDescriptor<TileOptimizations>);

// Depthwise-specific tile parameters (all as compile-time integers).
struct DepthwiseConvParams
{
    int block_size;
    int tile_h;
    int tile_w;
    int filter_h;
    int filter_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
    int nbatch;
    int subtile_h;
    int subtile_w;
    int in_vec;
    int out_vec;
};
static_assert(ckb::DepthwiseConvParamsDescriptor<DepthwiseConvParams>);

struct TileStreamKConfig
{
    // StreamK reduction strategy (Linear or Tree).
    StreamKReductionStrategy reduction_strategy;
    // Use persistent DP (true) or non-persistent DP (false).
    bool persistent;
};
static_assert(ckb::StreamKDescriptor<TileStreamKConfig>);

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

struct TileDepthwiseConvParams_
{
    DepthwiseConvParams depthwise_params;
};

struct TileStreamK_
{
    TileStreamKConfig streamk;
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
        if constexpr(std::is_base_of_v<FwdXdlGemm_, ConvAlgorithmTemplate>)
        {
            result.gridwise_gemm = gemm;
        }
        else if constexpr(std::is_base_of_v<BwdXdlGemm_, ConvAlgorithmTemplate>)
        {
            result.gridwise_gemm = gemm;
        }
        else if constexpr(std::is_base_of_v<BwdDataXdlGemm_, ConvAlgorithmTemplate>)
        {
            result.gridwise_gemm = gemm;
        }
        else if constexpr(std::is_base_of_v<WmmaGemm_, ConvAlgorithmTemplate>)
        {
            result.gridwise_gemm = gemm;
        }
        else if constexpr(std::is_base_of_v<WmmaGemmABK1_, ConvAlgorithmTemplate>)
        {
            result.gridwise_gemm = gemm;
        }
        else
        {
            static_assert(false, "Unrecognized GemmConfig type");
        }
        return result;
    }

    template <typename T>
    constexpr auto with_transfer(const T& t) const
    {
        static_assert(std::is_base_of_v<Transfer_<3>, ConvAlgorithmTemplate> ||
                      std::is_base_of_v<Transfer_<4>, ConvAlgorithmTemplate>);
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

    constexpr auto with_bwd_data_specialization(ConvSpecialization bwd_spec) const
    {
        static_assert(std::is_base_of_v<ConvSpecializationBwdData_, ConvAlgorithmTemplate>);
        auto result                    = *this;
        result.bwd_data_specialization = bwd_spec;
        return result;
    }

    constexpr auto with_prefetch_config(size_t k_prefetch_stages, PipelineScheduler scheduler) const
    {
        static_assert(std::is_base_of_v<Prefetch_, ConvAlgorithmTemplate>);
        auto result                       = *this;
        result.num_gemm_k_prefetch_stages = k_prefetch_stages;
        result.loop_scheduler             = scheduler;
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

    constexpr auto with_gemm_pad_params(size_t doPadGemmN_, size_t doPadGemmM_) const
    {
        static_assert(std::is_base_of_v<GemmPad_, ConvAlgorithmTemplate>);
        auto result       = *this;
        result.DoPadGemmN = doPadGemmN_;
        result.DoPadGemmM = doPadGemmM_;
        return result;
    }

    constexpr auto with_num_conv_groups_to_merge(size_t num_groups_to_merge) const
    {
        static_assert(std::is_base_of_v<GemmBatchOptions_, ConvAlgorithmTemplate>);
        auto result                     = *this;
        result.num_conv_groups_to_merge = num_groups_to_merge;
        return result;
    }

    template <typename BG>
    constexpr auto with_block_gemm(const BG& bg) const
    {
        static_assert(std::is_base_of_v<BlockGemm_, ConvAlgorithmTemplate>);
        auto result                = *this;
        result.block_gemm_pipeline = bg;
        return result;
    }

    constexpr auto with_gridwise_gemm_pipeline(const PipelineVersion plv) const
    {
        static_assert(std::is_base_of_v<GridGemm_, ConvAlgorithmTemplate>);
        auto result             = *this;
        result.pipeline_version = plv;
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

    template <typename SK>
    constexpr auto with_streamk(const SK& sk) const
    {
        static_assert(std::is_base_of_v<TileStreamK_, ConvAlgorithmTemplate>);
        auto result    = *this;
        result.streamk = sk;
        return result;
    }

    template <typename DW>
    constexpr auto with_depthwise_params(const DW& dw) const
    {
        static_assert(std::is_base_of_v<TileDepthwiseConvParams_, ConvAlgorithmTemplate>);
        auto result             = *this;
        result.depthwise_params = dw;
        return result;
    }
};

// Fwd algorithm types

using ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          FwdXdlGemm_,
                          Transfer_<>,
                          ConvSpecializationFwd_,
                          Prefetch_,
                          GemmBatchOptions_>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          FwdXdlGemm_,
                          Transfer_<>,
                          ConvSpecializationFwd_,
                          BlockGemm_,
                          GemmBatchOptions_>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemmABK1_,
                          Transfer_<>,
                          ConvSpecializationFwd_,
                          BlockGemm_,
                          GemmBatchOptions_>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemm_,
                          Transfer_<>,
                          ConvSpecializationFwd_,
                          GridGemm_,
                          Prefetch_,
                          GemmBatchOptions_>;

using ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK =
    ConvAlgorithmTemplate<ThreadBlock_,
                          ConvSpecializationFwd_,
                          DlThreadConfig_,
                          DlThreadCluster_,
                          DlTransfer_<>>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor =
    ConvAlgorithmTemplate<ThreadBlock_,
                          FwdXdlGemm_,
                          Transfer_<>,
                          ConvSpecializationFwd_,
                          Prefetch_,
                          GemmBatchOptions_,
                          LargeTensorSpecialization_>;

// CK Tile algorithm
using ConvAlgorithm_Tile_GroupedConvolutionKernel = ConvAlgorithmTemplate<TileThreadBlock_,
                                                                          TileBlockGemm_,
                                                                          TileTransfer_,
                                                                          TileConvSpecialization_,
                                                                          TileOptimizations_>;

// CK Tile algorithm with StreamK work distribution
using ConvAlgorithm_Tile_GroupedConvolutionKernel_StreamK =
    ConvAlgorithmTemplate<TileThreadBlock_,
                          TileBlockGemm_,
                          TileTransfer_,
                          TileConvSpecialization_,
                          TileOptimizations_,
                          TileStreamK_>;

// CK Tile depthwise convolution algorithm (no GEMM — direct spatial pipeline)
using ConvAlgorithm_Tile_DepthwiseConvolutionKernel =
    ConvAlgorithmTemplate<TileDepthwiseConvParams_>;

// Reference algorithm descriptor - for GPU reference validation
// This is a simple algorithm that requires no complex configuration,
// just a specialization marker to identify it as a reference implementation.
struct ConvAlgorithm_Reference
{
    static constexpr auto specialization = ckb::ConvAlgorithmSpecialization::REFERENCE;
    // GPU reference uses simple algorithm, no tile configuration needed
};

// Bwd weight algorithm types
using ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          BwdXdlGemm_,
                          Transfer_<4>,
                          ConvSpecializationBwdWeight_,
                          TransposeParams_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          BwdXdlGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdWeight_,
                          BlockGemm_,
                          TransposeParams_,
                          GemmBatchOptions_,
                          TwoStageSpecialization_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          BwdXdlGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdWeight_,
                          BlockGemm_,
                          GemmBatchOptions_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Dl =
    ConvAlgorithmTemplate<ThreadBlock_,
                          DlThreadConfig_,
                          DlThreadCluster_,
                          DlTransfer_<5>,
                          ConvSpecializationBwdWeight_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          BwdXdlGemm_,
                          Transfer_<4>,
                          ConvSpecializationBwdWeight_,
                          MultipleDSpecialization_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdWeight_,
                          BlockGemm_,
                          TransposeParams_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdWeight_,
                          BlockGemm_,
                          TransposeParams_,
                          GemmBatchOptions_,
                          TwoStageSpecialization_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdWeight_,
                          GridGemm_,
                          Prefetch_>;

using ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdWeight_,
                          BlockGemm_,
                          MultipleDSpecialization_>;

// Bwd Data algorithm types
using ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          BwdDataXdlGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdData_,
                          MultipleDSpecialization_,
                          Prefetch_,
                          TransposeParams_,
                          GemmPad_>;

using ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemm_,
                          Transfer_<>,
                          ConvSpecializationBwdData_,
                          GridGemm_,
                          MultipleDSpecialization_,
                          Prefetch_>;

using ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_,
                          WmmaGemmABK1_,
                          Transfer_<>,
                          ConvSpecializationBwdData_,
                          BlockGemm_,
                          MultipleDSpecialization_,
                          Prefetch_,
                          TransposeParams_,
                          GemmPad_>;

} // namespace ck_tile::builder::test
