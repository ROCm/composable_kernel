#pragma once

#include <string_view>
#include <sstream>
#include <type_traits>

#include <ck_tile/builder/conv_signature.hpp>
#include <ck_tile/builder/conv_traits.hpp>
#include <ck_tile/builder/tree_formatter.hpp>

namespace ck_tile::reflect {

// Decoupled structs in the reflect namespace for runtime storage
struct SignatureInfo
{
    int spatial_dim;
    builder::ConvDirection direction;
    builder::GroupConvLayout layout;
    builder::DataType data_type;
};

struct BlockInfo
{
    int block_size;
    int m; // submatrix m dimension
    int n; // submatrix n dimension
    int k; // submatrix k dimension
};

struct TuningInfo
{
    int ak1;
    int bk1;
    int m_per_xdl;
    int n_per_xdl;
    int m_xdl_per_wave;
    int n_xdl_per_wave;
};

struct BlockTransferInfo
{
    int k0;
    int m_or_n; // m for A transfer, n for B transfer
    int k1;
};

struct CBlockTransferInfo
{
    int m_xdl_per_wave_per_shuffle;
    int n_xdl_per_wave_per_shuffle;
    int m_block;
    int m_wave_per_xdl;
    int n_block;
    int n_wave_per_xdl;
};

// Algorithm information - groups all algorithm-related configuration
struct AlgorithmInfo
{
    BlockInfo block;
    TuningInfo tuning;
    BlockTransferInfo a_transfer;
    BlockTransferInfo b_transfer;
    CBlockTransferInfo c_transfer;
    ck::BlockGemmPipelineVersion pipeline;
};

// Provides human-readable descriptions of ConvBuilder configurations.
struct Description
{
    // Public fields - runtime storage (allows future std::optional/variant)
    SignatureInfo signature;
    AlgorithmInfo algorithm;

    // Brief one-line summary
    std::string brief() const
    {
        std::ostringstream oss;
        oss << builder::ConvDirectionToString(signature.direction) << " convolution";
        return oss.str();
    }

    // Detailed hierarchical description
    std::string detailed() const
    {
        std::ostringstream oss;
        TreeFormatter tree;

        // Root line - no tree formatting
        oss << signature.spatial_dim << "D " << builder::ConvDirectionToString(signature.direction)
            << " Convolution Kernel\n";

        tree.writeLine(1, "Signature");
        tree.writeLine(2, "Tensor Type: ", signature.data_type);
        tree.writeLastLine(2, "Memory Layout: ", signature.layout);

        tree.writeLastLine(1, "Algorithm");
        // Compute Block section
        tree.writeLine(2,
                       "Compute Block: ",
                       algorithm.block.m,
                       "×",
                       algorithm.block.n,
                       "×",
                       algorithm.block.k,
                       " submatrix (",
                       algorithm.block.block_size,
                       " threads)");

        tree.writeLine(3,
                       "XDL Waves: ",
                       algorithm.tuning.m_xdl_per_wave,
                       "×",
                       algorithm.tuning.n_xdl_per_wave,
                       " mapping (",
                       (algorithm.tuning.m_xdl_per_wave * algorithm.tuning.n_xdl_per_wave),
                       " waves total)");
        tree.writeLastLine(3,
                           "Tuning: ak1=",
                           algorithm.tuning.ak1,
                           ", bk1=",
                           algorithm.tuning.bk1,
                           " (optimized for MI300 MFMA)");

        // Memory Layout section
        tree.writeLine(2, "Memory Layout:");
        tree.writeLine(3,
                       "A-Transfer: ",
                       algorithm.a_transfer.k0,
                       "×",
                       algorithm.a_transfer.m_or_n,
                       "×",
                       algorithm.a_transfer.k1,
                       " thread clusters (coalesced reads)");
        tree.writeLine(3,
                       "B-Transfer: ",
                       algorithm.b_transfer.k0,
                       "×",
                       algorithm.b_transfer.m_or_n,
                       "×",
                       algorithm.b_transfer.k1,
                       " thread clusters (broadcast-friendly)");
        tree.writeLastLine(3,
                           "C-Transfer: ",
                           algorithm.c_transfer.m_block,
                           "×",
                           algorithm.c_transfer.m_wave_per_xdl,
                           "×",
                           algorithm.c_transfer.n_block,
                           "×",
                           algorithm.c_transfer.n_wave_per_xdl,
                           " clusters (efficient writeback)");

        // Pipeline section
        tree.writeLastLine(2, "Pipeline: ", algorithm.pipeline);

        return oss.str() + tree.getString();
    }

    // Educational explanation of optimization choices
    std::string explain() const
    {
        std::ostringstream oss;
        // Placeholder for future implementation
        return oss.str();
    }

    // Performance characteristics and use case guidance
    std::string suggest() const
    {
        std::ostringstream oss;
        // Placeholder for future implementation
        return oss.str();
    }
};

// Factory function to create Description from a Builder type.
template <typename Builder>
Description Describe()
{
    using Traits = ConvTraits<Builder>;

    return Description{
        // signature
        SignatureInfo{Traits::spatial_dim, Traits::direction, Traits::layout, Traits::data_type},
        // algorithm
        AlgorithmInfo{// block
                      BlockInfo{Traits::block.block_size,
                                Traits::block.per_block.m,
                                Traits::block.per_block.n,
                                Traits::block.per_block.k},
                      // tuning
                      TuningInfo{Traits::tuning.ak1,
                                 Traits::tuning.bk1,
                                 Traits::tuning.m_per_xdl,
                                 Traits::tuning.n_per_dxl,
                                 Traits::tuning.m_xdl_per_wave,
                                 Traits::tuning.n_xdl_per_wave},
                      // a_transfer
                      BlockTransferInfo{Traits::a_block_transfer.thread_cluster_dims[0],
                                        Traits::a_block_transfer.thread_cluster_dims[1],
                                        Traits::a_block_transfer.thread_cluster_dims[2]},
                      // b_transfer
                      BlockTransferInfo{Traits::b_block_transfer.thread_cluster_dims[0],
                                        Traits::b_block_transfer.thread_cluster_dims[1],
                                        Traits::b_block_transfer.thread_cluster_dims[2]},
                      // c_transfer
                      CBlockTransferInfo{Traits::c_block_transfer.m_xdl_per_wave_per_shuffle,
                                         Traits::c_block_transfer.n_xdl_per_wave_per_shuffle,
                                         Traits::c_block_transfer.thread_cluster_dims[0],
                                         Traits::c_block_transfer.thread_cluster_dims[1],
                                         Traits::c_block_transfer.thread_cluster_dims[2],
                                         Traits::c_block_transfer.thread_cluster_dims[3]},
                      // pipeline
                      Traits::pipeline_version}};
}

} // namespace ck_tile::reflect
