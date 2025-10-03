#pragma once

#include "ck_tile/builder/conv_algorithm.hpp"
#include "ck_tile/builder/conv_signature.hpp"

namespace ck_tile::builder::examples
{
    namespace ckb = ck_tile::builder;

    struct ConvSignature {
        int spatial_dim;
        ckb::ConvDirection direction;
        ckb::GroupConvLayout layout;
        ckb::DataType data_type;
    };
    static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

    struct BlockTransfer
    {
        ckb::BlockATransferLengths thread_cluster_dims_a;
        ckb::BlockBTransferLengths thread_cluster_dims_b;
        ckb::BlockCTransferLengths thread_cluster_dims_c;
        ckb::VectorTransferAB vector_transfer_a;
        ckb::VectorTransferAB vector_transfer_b;
        ckb::VectorTransferC vector_transfer_c;
        ckb::ThreadClusterAccessOrder a_thread_cluster_access_order;
        ckb::ThreadClusterAccessOrder b_thread_cluster_access_order;
        ckb::SourceAccessOrder a_source_access_order;
        ckb::SourceAccessOrder b_source_access_order;
    };
    
    struct ConvAlgorithm
    {
        ckb::ThreadBlock thread_block;
        ckb::ConvTuningParams tuning_params;
        BlockTransfer block_transfer;
        ckb::BlockGemmPipelineVersion pipeline_version;
    };
    static_assert(ckb::ConvAlgorithmDescriptor<ConvAlgorithm>);
    static_assert(ckb::SpecifiesThreadBlock<ConvAlgorithm>);
    static_assert(ckb::SpecifiesConvTuning<ConvAlgorithm>);
    static_assert(ckb::SpecifiesGemmPipelineVersion<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBlockATransfer<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBlockBTransfer<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBlockCTransfer<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBlockAVectorTransfer<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBlockBVectorTransfer<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBlockCVectorTransfer<ConvAlgorithm>);
    static_assert(ckb::SpecifiesAThreadClusterAccessOrder<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBThreadClusterAccessOrder<ConvAlgorithm>);
    static_assert(ckb::SpecifiesASourceAccessOrder<ConvAlgorithm>);
    static_assert(ckb::SpecifiesBSourceAccessOrder<ConvAlgorithm>);
} // namespace ck_tile::builder::examples
