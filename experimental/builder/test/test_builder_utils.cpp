#include <gtest/gtest.h>
#include <ck/utility/sequence.hpp>
#include <array>
#include <type_traits>

#include <ck_tile/builder/builder_utils.hpp>

namespace ck_tile::builder {
namespace {

TEST(BuilderUtils, ToSequence)
{
    static constexpr std::array<int, 4> test_arr{1, 2, 8, 16};
    using result_seq   = to_sequence_v<test_arr>;
    using expected_seq = ck::Sequence<1, 2, 8, 16>;
    EXPECT_TRUE((std::is_same_v<result_seq, expected_seq>));
}

TEST(BuilderUtils, ToSequenceEmpty)
{
    static constexpr std::array<int, 0> test_arr{};
    using result_seq   = to_sequence_v<test_arr>;
    using expected_seq = ck::Sequence<>;
    EXPECT_TRUE((std::is_same_v<result_seq, expected_seq>));
}

Test(BuilderUtils, ConstexprString)
{
    static constexpr ConstexprString str1{"hello"};
    static constexpr ConstexprString str2{"hello"};
    static constexpr ConstexprString str3{"world"};
    EXPECT_TRUE(str1 == str2);
    EXPECT_FALSE(str1 == str3);
};

} // namespace
} // namespace ck_tile::builder
