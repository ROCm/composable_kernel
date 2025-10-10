#pragma once

#include "conv_algorithm_concepts.hpp"

namespace ck_tile::builder {
// Convenience struct for a tuple of m, n, and k values.
template <typename T>
struct MNK
{
    T m{};
    T n{};
    T k{};
};

// Specifiy thread block dimensions for a GEMM.
struct ThreadBlock
{
    // Thread block size.
    int block_size;
    // Size of the submatrix problem in a thread block.
    MNK<int> submatrix;
};
static_assert(ThreadBlockDescriptor<ThreadBlock>);

// Describe some convolution tuning parameters.
struct ConvTuningParams
{
    // NOTE: ak1 and bk1 are difficult to verify in the kernel instantiation!!!
    int ak1            = 0;
    int bk1            = 0;
    int m_per_xdl      = 0;
    int n_per_xdl      = 0;
    int m_xdl_per_wave = 0;
    int n_xdl_per_wave = 0;
};
static_assert(ConvTuningDescriptor<ConvTuningParams>);

// Describe A block transfer thread cluster lengths.
struct BlockATransferLengths
{
    int k0;
    int m;
    int k1;
};
static_assert(BlockATransferDescriptor<BlockATransferLengths>);

// Describe B block transfer thread cluster lengths.
struct BlockBTransferLengths
{
    int k0;
    int n;
    int k1;
};
static_assert(BlockBTransferDescriptor<BlockBTransferLengths>);

// Describe the thread cluster access order.
struct ThreadClusterAccessOrder
{
    // Order of the cluster dimensions. Must be a permutation of {0, 1, 2}.
    std::array<int, 3> order;
};
static_assert(ThreadClusterAccessOrderDescriptor<ThreadClusterAccessOrder>);

// Describe the source access order.
struct SourceAccessOrder
{
    // Order of the source dimensions. Must be a permutation of {0, 1, 2}.
    std::array<int, 3> order;
};
static_assert(SourceAccessOrderDescriptor<SourceAccessOrder>);

// Describe C block transfer thread cluster lengths.
struct BlockCTransferLengths
{
    int m_block;
    int m_wave_per_xdl;
    int n_block;
    int n_wave_per_xdl;
};
static_assert(BlockCTransferDescriptor<BlockCTransferLengths>);

struct VectorTransferAB
{
    size_t src_vector_dim;
    size_t src_scaler_per_vector;
    size_t dest_scaler_per_vector_k1;
    bool add_extra; 
};
static_assert(VectorTransferDescriptorAB<VectorTransferAB>);

struct VectorTransferC
{
    size_t m_xdl_per_wave_per_shuffle;
    size_t n_xdl_per_wave_per_shuffle;
    size_t scaler_per_vector;
};
static_assert(VectorTransferDescriptorC<VectorTransferC>);

struct BlockTransfer
{
    BlockATransferLengths thread_cluster_dims_a;
    BlockBTransferLengths thread_cluster_dims_b;
    BlockCTransferLengths thread_cluster_dims_c;
    VectorTransferAB vector_transfer_a;
    VectorTransferAB vector_transfer_b;
    VectorTransferC vector_transfer_c;
    ThreadClusterAccessOrder a_thread_cluster_access_order;
    ThreadClusterAccessOrder b_thread_cluster_access_order;
    SourceAccessOrder a_source_access_order;
    SourceAccessOrder b_source_access_order;
};

struct ConvAlgorithm
{
    ThreadBlock thread_block;
    ConvTuningParams tuning_params;
    BlockTransfer block_transfer;
    BlockGemmPipelineVersion pipeline_version;
};
static_assert(ConvAlgorithmDescriptor<ConvAlgorithm>);
static_assert(SpecifiesThreadBlock<ConvAlgorithm>);
static_assert(SpecifiesConvTuning<ConvAlgorithm>);
static_assert(SpecifiesGemmPipelineVersion<ConvAlgorithm>);
static_assert(SpecifiesBlockATransfer<ConvAlgorithm>);
static_assert(SpecifiesBlockBTransfer<ConvAlgorithm>);
static_assert(SpecifiesBlockCTransfer<ConvAlgorithm>);
static_assert(SpecifiesBlockAVectorTransfer<ConvAlgorithm>);
static_assert(SpecifiesBlockBVectorTransfer<ConvAlgorithm>);
static_assert(SpecifiesBlockCVectorTransfer<ConvAlgorithm>);
static_assert(SpecifiesAThreadClusterAccessOrder<ConvAlgorithm>);
static_assert(SpecifiesBThreadClusterAccessOrder<ConvAlgorithm>);
static_assert(SpecifiesASourceAccessOrder<ConvAlgorithm>);
static_assert(SpecifiesBSourceAccessOrder<ConvAlgorithm>);

} // namespace ck_tile::builder
