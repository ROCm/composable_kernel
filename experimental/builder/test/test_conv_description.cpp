#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/conv_description.hpp>
#include "testing_utils.hpp"

namespace {

namespace ckb = ck_tile::builder;
namespace ckr = ck_tile::reflect;
namespace ckt = ck_tile::test;
using P       = ckb::BlockGemmPipelineVersion;

// Defines the signature of the convolution operation to be tested.
// This includes dimensionality, direction, data layout, and data type.
struct ConvSignature
{
    int spatial_dim              = 2;
    ckb::ConvDirection direction = ckb::ConvDirection::FORWARD;
    ckb::GroupConvLayout layout  = ckb::GroupConvLayout::CHANNELS_LAST;
    ckb::DataType data_type      = ckb::DataType::FP16;
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

struct DefaultAlgorithm
{
};
static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

TEST(ConvDescriptionTest, DefaultInstanceHasBriefDescription)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    EXPECT_THAT(ckr::Description<Builder>::Brief(), ckt::StringEqWithDiff("Forward convolution"));
}

TEST(ConvDescriptionTest, DefaultInstanceHasDetailedDescription)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    EXPECT_THAT(ckr::Description<Builder>{}.detailed(),
                ckt::StringEqWithDiff( //
                    "2D Forward Convolution Kernel\n"
                    "├─ Signature\n"
                    "│  ├─ Tensor Type: FP16\n"
                    "│  └─ Memory Layout: Channels-last (NHWC)\n"
                    "└─ Algorithm\n"
                    "   ├─ Compute Block: 256×256×32 submatrix (256 threads)\n"
                    "   │  ├─ XDL Waves: 4×4 mapping (16 waves total)\n"
                    "   │  └─ Tuning: ak1=8, bk1=8 (optimized for MI300 MFMA)\n"
                    "   ├─ Memory Layout:\n"
                    "   │  ├─ A-Transfer: 4×64×1 thread clusters (coalesced reads)\n"
                    "   │  ├─ B-Transfer: 4×64×1 thread clusters (broadcast-friendly)\n"
                    "   │  └─ C-Transfer: 1×32×1×8 clusters (efficient writeback)\n"
                    "   └─ Pipeline: V4"));
}

TEST(ConvDescriptionTest, BackwardDataInstanceHasDetailedDescription)
{
    struct BackwardDataSignature
    {
        int spatial_dim              = 2;
        ckb::ConvDirection direction = ckb::ConvDirection::BACKWARD_DATA;
        ckb::GroupConvLayout layout  = ckb::GroupConvLayout::CHANNELS_LAST;
        ckb::DataType data_type      = ckb::DataType::FP16;
    };
    static_assert(ckb::ConvSignatureDescriptor<BackwardDataSignature>);

    static constexpr const BackwardDataSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    
    // Verify Brief works
    EXPECT_THAT(ckr::Description<Builder>::Brief(), 
                ckt::StringEqWithDiff("Backward Data convolution"));
    
    // Verify detailed works with backward data factory defaults
    EXPECT_THAT(ckr::Description<Builder>{}.detailed(),
                ckt::StringEqWithDiff( //
                    "2D Backward Data Convolution Kernel\n"
                    "├─ Signature\n"
                    "│  ├─ Tensor Type: FP16\n"
                    "│  └─ Memory Layout: Channels-last (NHWC)\n"
                    "└─ Algorithm\n"
                    "   ├─ Compute Block: 256×256×32 submatrix (256 threads)\n"
                    "   │  ├─ XDL Waves: 1×4 mapping (4 waves total)\n"
                    "   │  └─ Tuning: ak1=8, bk1=8 (optimized for MI300 MFMA)\n"
                    "   ├─ Memory Layout:\n"
                    "   │  ├─ A-Transfer: 4×16×1 thread clusters (coalesced reads)\n"
                    "   │  ├─ B-Transfer: 4×8×1 thread clusters (broadcast-friendly)\n"
                    "   │  └─ C-Transfer: 1×16×1×4 clusters (efficient writeback)\n"
                    "   └─ Pipeline: V4"));
}
} // namespace
