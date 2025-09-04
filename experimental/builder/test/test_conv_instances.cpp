#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>

namespace {

namespace ckb = ck_tile::builder;
struct FwdConvSignature
{
    static constexpr int spatial_dim = 2;
    static constexpr auto direction  = ckb::ConvDirection::Forward;
    static constexpr auto layout     = ckb::GroupConvLayout::NHWGC_GKYXC_NHWGK;
    static constexpr auto data_type  = ckb::DataType::FP16;
};
static_assert(ckb::ConvSignature<FwdConvSignature>);

static constexpr char API_VERSION[] = "0.1.0";

struct FwdConvAlgorithm
{
    ckb::ThreadBlock thread_block;
    ckb::ConvTuningParams tuning_params;
    struct BlockTransfer
    {
        ckb::BlockATransferLengthsInfo thread_cluster_lengths_a;
        ckb::BlockBTransferLengthsInfo thread_cluster_lengths_b;
        ckb::BlockCTransferLengthsInfo thread_cluster_lengths_c;
    } block_transfer;
};
static_assert(ckb::ConvAlgorithm<FwdConvAlgorithm>);
static_assert(ckb::HasThreadBlockInfo<FwdConvAlgorithm>);
static_assert(ckb::HasConvTuningInfo<FwdConvAlgorithm>);
static_assert(ckb::HasABlockTransferInfo<FwdConvAlgorithm>);
static_assert(ckb::HasBBlockTransferInfo<FwdConvAlgorithm>);
static_assert(ckb::HasCBlockTransferInfo<FwdConvAlgorithm>);

struct TestCase
{
    std::string_view name;
    FwdConvAlgorithm algorithm;
    std::string_view expected_type;
};

// Test cases to drive the typed test suite.
constexpr std::array TEST_CASES = {
    TestCase{
        .name = "ConvFwdXdlBf16CompInstances2x_0",
        .algorithm =
            {.thread_block{.block_size = 256, .sub_matrix = {.m = 256, .n = 128, .k = 64}},
             .tuning_params{.ak1 = 16, .bk1 = 16, .m_xdl_per_wave = 2, .n_xdl_per_wave = 2},
             .block_transfer{
                 .thread_cluster_lengths_a = {.k0 = 4, .m = 64, .k1 = 1},
                 .thread_cluster_lengths_b = {.k0 = 4, .n = 64, .k1 = 1},
                 .thread_cluster_lengths_c =
                     {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
             }},
        .expected_type =
            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 64, Default, 32, 32, "
            "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
    },
    TestCase{
        .name      = "GroupedConvFwdXdlBf16CompInstance0",
        .algorithm = {.thread_block{.block_size = 256, .sub_matrix = {.m = 256, .n = 256, .k = 32}},
                      .tuning_params{.ak1 = 8, .bk1 = 8, .m_xdl_per_wave = 4, .n_xdl_per_wave = 4},
                      .block_transfer{
                          .thread_cluster_lengths_a = {.k0 = 4, .m = 64, .k1 = 1},
                          .thread_cluster_lengths_b = {.k0 = 4, .n = 64, .k1 = 1},
                          .thread_cluster_lengths_c = {.m_block        = 1,
                                                       .m_wave_per_xdl = 32,
                                                       .n_block        = 1,
                                                       .n_wave_per_xdl = 8},
                      }},
        .expected_type =
            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, "
            "4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
    },
    TestCase{
        .name      = "GroupedConvFwdXdlBf16CompInstance1",
        .algorithm = {.thread_block{.block_size = 256, .sub_matrix = {.m = 128, .n = 128, .k = 64}},
                      .tuning_params{.ak1 = 8, .bk1 = 8, .m_xdl_per_wave = 2, .n_xdl_per_wave = 2},
                      .block_transfer{
                          .thread_cluster_lengths_a = {.k0 = 8, .m = 32, .k1 = 1},
                          .thread_cluster_lengths_b = {.k0 = 8, .n = 32, .k1 = 1},
                          .thread_cluster_lengths_c = {.m_block        = 1,
                                                       .m_wave_per_xdl = 32,
                                                       .n_block        = 1,
                                                       .n_wave_per_xdl = 8},
                      }},
        .expected_type =
            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, "
            "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
    },
    TestCase{
        .name      = "GroupedConvFwdXdlBf16CompInstance1",
        .algorithm = {.thread_block{.block_size = 256, .sub_matrix = {.m = 128, .n = 128, .k = 32}},
                      .tuning_params{.ak1 = 8, .bk1 = 8, .m_xdl_per_wave = 2, .n_xdl_per_wave = 2},
                      .block_transfer{
                          .thread_cluster_lengths_a = {.k0 = 4, .m = 64, .k1 = 1},
                          .thread_cluster_lengths_b = {.k0 = 4, .n = 64, .k1 = 1},
                          .thread_cluster_lengths_c = {.m_block        = 1,
                                                       .m_wave_per_xdl = 32,
                                                       .n_block        = 1,
                                                       .n_wave_per_xdl = 8},
                      }},
        .expected_type =
            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Default, 32, 32, "
            "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
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

// A typed test suite so we can instantiate all the kernel builders.
template <typename T>
class ConvBuilderInstancesTest : public ::testing::Test
{
    protected:
    static constexpr int N                                 = T::index;
    static constexpr const std::string_view& NAME          = TEST_CASES[N].name;
    static constexpr auto& ALGORITHM                       = TEST_CASES[N].algorithm;
    static constexpr const std::string_view& EXPECTED_TYPE = TEST_CASES[N].expected_type;
};

struct TestNameGenerator
{
    template <typename T>
    static std::string GetName(int index)
    {
        return std::to_string(index) + "." + std::string(TEST_CASES[index].name);
    }
};

TYPED_TEST_SUITE(ConvBuilderInstancesTest,
                 TestingIndices<NUM_TEST_CASES>::Types,
                 TestNameGenerator);

// General test case, instantiated for each test case.
TYPED_TEST(ConvBuilderInstancesTest, KernelParamsConfigured)
{
    static constexpr const FwdConvAlgorithm& ALGORITHM =
        ConvBuilderInstancesTest<TypeParam>::ALGORITHM;
    static constexpr const FwdConvSignature SIGNATURE;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM, API_VERSION>;
    EXPECT_EQ(Builder::Instance::TypeString(), ConvBuilderInstancesTest<TypeParam>::EXPECTED_TYPE);
    const auto& tp = ALGORITHM.tuning_params;
    EXPECT_EQ(Builder::factory::TUNING.ak1, tp.ak1);
    EXPECT_EQ(Builder::factory::TUNING.bk1, tp.bk1);
    const auto& tcla = ALGORITHM.block_transfer.thread_cluster_lengths_a;
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[0], tcla.k0);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[1], tcla.m);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[2], tcla.k1);
    const auto& tclb = ALGORITHM.block_transfer.thread_cluster_lengths_b;
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[0], tclb.k0);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[1], tclb.n);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[2], tclb.k1);
    const auto& tclc = ALGORITHM.block_transfer.thread_cluster_lengths_c;
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[0], tclc.m_block);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[1], tclc.m_wave_per_xdl);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[2], tclc.n_block);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[3], tclc.n_wave_per_xdl);
}

TEST(ConvBuilderInstancesTest, TypeStringsAreUnique)
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
