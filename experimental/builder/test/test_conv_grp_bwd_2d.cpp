#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>

namespace {

namespace ckb = ck_tile::builder;
using P       = ckb::BlockGemmPipelineVersion;

// Defines the signature of the convolution operation to be tested.
// This includes dimensionality, direction, data layout, and data type.
struct ConvSignature
{
    int spatial_dim              = 2;
    ckb::ConvDirection direction = ckb::ConvDirection::BACKWARD_DATA;
    ckb::GroupConvLayout layout  = ckb::GroupConvLayout::CHANNELS_LAST;
    ckb::DataType data_type      = ckb::DataType::FP16;
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

struct ConvAlgorithm
{
    ckb::ThreadBlock thread_block;
    // ckb::ConvTuningParams tuning_params;
    // struct BlockTransfer
    // {
    //     ckb::BlockATransferLengths thread_cluster_dims_a;
    //     ckb::BlockBTransferLengths thread_cluster_dims_b;
    //     // ckb::BlockCTransferLengths thread_cluster_dims_c;
    // } block_transfer;
};
static_assert(ckb::ConvAlgorithmDescriptor<ConvAlgorithm>);
static_assert(ckb::SpecifiesThreadBlock<ConvAlgorithm>);
// static_assert(ckb::SpecifiesConvTuning<ConvAlgorithm>);
// static_assert(ckb::SpecifiesBlockATransfer<ConvAlgorithm>);
// static_assert(ckb::SpecifiesBlockBTransfer<ConvAlgorithm>);
// static_assert(ckb::SpecifiesBlockCTransfer<ConvAlgorithm>);

// Comment out this test until it compiles.

TEST(ConvBuilderGrpBwd2d, TestFirstExample)
{
    // test first instance in
    // include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const ConvAlgorithm ALGORITHM{
        .thread_block   = {.block_size = 64, .submatrix = {.m = 16, .n = 64, .k = 32}},
        // .tuning_params  = {.ak1 = 8, .bk1 = 8, .m_xdl_per_wave = 4, .n_xdl_per_wave = 1},
        // .block_transfer = {
        //     .thread_cluster_dims_a = {.k0 = 4, .m = 16, .k1 = 1},
        //     .thread_cluster_dims_b = {.k0 = 4, .n = 8, .k1 = 1}}
        };
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
	"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Default, 16, 16, 1, 4, 8, 8, 1, 1, "
	"TransposeTransferInScalarPerVectorAligned: 1, TransposeTransferOutScalarPerVectorAligned: 1>");
}

} // namespace
