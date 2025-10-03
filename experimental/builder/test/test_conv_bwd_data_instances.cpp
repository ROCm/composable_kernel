// This test is designed to verify that the ConvBuilder can instantiate the same
// kernel classes that are used in production code. Production code may have
// hundreds or thousands of kernel instances, so this test uses a GTest typed
// test suite to efficiently test a representative set of these kernel examples.
// Each test case defines a specific convolution algorithm configuration and the
// expected kernel type string that the builder should generate.
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

// Defines the tunable algorithmic parameters for the convolution kernel.
// This includes thread block configuration, tuning parameters, data transfer
// settings, and the GEMM pipeline version.
struct BwdDataConvAlgorithm
{
    ckb::ThreadBlock thread_block;
    ckb::ConvTuningParams tuning_params;
    struct BlockTransfer
    {
        ckb::BlockATransferLengths thread_cluster_dims_a;
        ckb::BlockBTransferLengths thread_cluster_dims_b;
        ckb::BlockCTransferLengths thread_cluster_dims_c;
    } block_transfer;
};
static_assert(ckb::ConvAlgorithmDescriptor<BwdDataConvAlgorithm>);
static_assert(ckb::SpecifiesThreadBlock<BwdDataConvAlgorithm>);
static_assert(ckb::SpecifiesConvTuning<BwdDataConvAlgorithm>);
static_assert(ckb::SpecifiesBlockATransfer<BwdDataConvAlgorithm>);
static_assert(ckb::SpecifiesBlockBTransfer<BwdDataConvAlgorithm>);
static_assert(ckb::SpecifiesBlockCTransfer<BwdDataConvAlgorithm>);

// A container for a single test case, bundling a descriptive name, the
// algorithm configuration, and the expected generated kernel type string.
struct TestCase
{
    std::string_view name;
    BwdDataConvAlgorithm algorithm;
    std::string_view expected_type;
};

// Helper function to set the sub_matrix size.
constexpr ckb::ThreadBlock set_submatrix(int m, int n, int k)
{
    return {.block_size = 256, .submatrix = {.m = m, .n = n, .k = k}};
}

// An array of test cases that drive the typed test suite. Each entry
// represents a unique kernel instance to be verified.
constexpr std::array TEST_CASES = {
    TestCase{
        .name = "ConvFwdXdlBf16CompInstances2x_0",
        .algorithm =
            {.thread_block   = set_submatrix(256, 128, 64),
             .tuning_params  = {.ak1 = 8, .bk1 = 8, .m_xdl_per_wave = 2, .n_xdl_per_wave = 2},
             .block_transfer = {.thread_cluster_dims_a = {.k0 = 4, .m = 16, .k1 = 1},
                                .thread_cluster_dims_b = {.k0 = 4, .n = 8, .k1 = 1},
                                .thread_cluster_dims_c = {.m_block        = 1,
                                                          .m_wave_per_xdl = 16,
                                                          .n_block        = 1,
                                                          .n_wave_per_xdl = 4}}},
        .expected_type =
            "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 256, 128, 64, 8, 8, Default, "
            "16, 16, 1, 4, 8, 8, 1, 1, TransposeTransferInScalarPerVectorAligned: 1, "
            "TransposeTransferOutScalarPerVectorAligned: 1>",
    },
};

static constexpr int NUM_TEST_CASES = std::tuple_size_v<decltype(TEST_CASES)>;

// Helper to generate testing::Types<TestIndex<0>, TestIndex<1>, ..., TestIndex<N-1>>.
template <int N>
struct TestingIndices
{
    template <int INDEX>
    struct TestIndex
    {
        static constexpr int index = INDEX;
    };
    template <typename T, T... Indices>
    static auto GenerateTypes(std::integer_sequence<T, Indices...>)
    {
        return ::testing::Types<TestIndex<Indices>...>{};
    }
    // testing::Types sequence of TestIndex types.
    using Types = decltype(GenerateTypes(std::make_integer_sequence<int, N>{}));
};

// A typed test suite that will be instantiated for each type in TestingIndices::Types.
// This creates a separate test for each entry in the TEST_CASES array, allowing
// GTest to run and report on them individually.
template <typename T>
class ConvBuilderBwdInstancesTest : public ::testing::Test
{
    protected:
    static constexpr int N                                 = T::index;
    static constexpr const std::string_view& NAME          = TEST_CASES[N].name;
    static constexpr auto& ALGORITHM                       = TEST_CASES[N].algorithm;
    static constexpr const std::string_view& EXPECTED_TYPE = TEST_CASES[N].expected_type;
};

// Custom test name generator to provide more descriptive names for each
// typed test instance, incorporating the index and the name from the TestCase.
struct TestNameGenerator
{
    template <typename T>
    static std::string GetName(int index)
    {
        return std::to_string(index) + "." + std::string(TEST_CASES[index].name);
    }
};

TYPED_TEST_SUITE(ConvBuilderBwdInstancesTest,
                 TestingIndices<NUM_TEST_CASES>::Types,
                 TestNameGenerator);

// This is the body of the typed test. It will be executed for each TestCase.
// It verifies that the ConvBuilder, when configured with a specific algorithm,
// generates the correct kernel type string and correctly configures the
// underlying factory parameters.
TYPED_TEST(ConvBuilderBwdInstancesTest, KernelParamsConfigured)
{
    static constexpr const BwdDataConvAlgorithm& ALGORITHM =
        ConvBuilderBwdInstancesTest<TypeParam>::ALGORITHM;
    static constexpr const ConvSignature SIGNATURE;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    EXPECT_EQ(Builder::Instance::TypeString(),
              ConvBuilderBwdInstancesTest<TypeParam>::EXPECTED_TYPE);
    const auto& tp = ALGORITHM.tuning_params;
    EXPECT_EQ(Builder::Factory::TUNING.ak1, tp.ak1);
    EXPECT_EQ(Builder::Factory::TUNING.bk1, tp.bk1);
    const auto& tcda = ALGORITHM.block_transfer.thread_cluster_dims_a;
    EXPECT_EQ(Builder::Factory::A_BLOCK_TRANSFER.thread_cluster_dims[0], tcda.k0);
    EXPECT_EQ(Builder::Factory::A_BLOCK_TRANSFER.thread_cluster_dims[1], tcda.m);
    EXPECT_EQ(Builder::Factory::A_BLOCK_TRANSFER.thread_cluster_dims[2], tcda.k1);
    const auto& tcdb = ALGORITHM.block_transfer.thread_cluster_dims_b;
    EXPECT_EQ(Builder::Factory::B_BLOCK_TRANSFER.thread_cluster_dims[0], tcdb.k0);
    EXPECT_EQ(Builder::Factory::B_BLOCK_TRANSFER.thread_cluster_dims[1], tcdb.n);
    EXPECT_EQ(Builder::Factory::B_BLOCK_TRANSFER.thread_cluster_dims[2], tcdb.k1);
    const auto& tcdc = ALGORITHM.block_transfer.thread_cluster_dims_c;
    EXPECT_EQ(Builder::Factory::C_BLOCK_TRANSFER.thread_cluster_dims[0], tcdc.m_block);
    EXPECT_EQ(Builder::Factory::C_BLOCK_TRANSFER.thread_cluster_dims[1], tcdc.m_wave_per_xdl);
    EXPECT_EQ(Builder::Factory::C_BLOCK_TRANSFER.thread_cluster_dims[2], tcdc.n_block);
    EXPECT_EQ(Builder::Factory::C_BLOCK_TRANSFER.thread_cluster_dims[3], tcdc.n_wave_per_xdl);
}

// A standard GTest to ensure that all `expected_type` strings in the
// TEST_CASES array are unique. This helps prevent copy-paste errors and
// ensures that each test case is meaningful.
TEST(ConvBuilderBwdInstancesTest, TypeStringsAreUnique)
{
    std::set<std::string> strings;
    for(int i = 0; i < NUM_TEST_CASES; ++i)
    {
        const auto& [iter, inserted] = strings.insert(std::string(TEST_CASES[i].expected_type));
        EXPECT_TRUE(inserted) << "Duplicate expected_string " << *iter;
    }
    EXPECT_EQ(strings.size(), NUM_TEST_CASES)
        << "Found fewer unique expected_strings than test cases";
}

} // namespace
