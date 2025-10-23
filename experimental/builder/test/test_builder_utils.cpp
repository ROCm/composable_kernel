// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <ck/utility/sequence.hpp>
#include <array>
#include <type_traits>

#include <ck_tile/builder/builder_utils.hpp>
#include "testing_utils.hpp"

namespace ck_tile::builder {
namespace {

class BuilderUtils : public ::testing::Test
{
};

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

TEST(BuilderUtils, StringLiteral)
{
    std::string str1{"hello"};
    std::string str2{"hello"};
    std::string str3{"world"};

    // some easy tests
    // you can veryfy the ungodly strings are meaningful by running echo -e "<string>"
    EXPECT_THAT(test::inlineDiff(str1, str2), "hello");
    EXPECT_THAT(test::inlineDiff(str1, str3),
                "[\x1B[36mwor\x1B[0m|\x1B[35mhel\x1B[0m]l[\x1B[36md\x1B[0m|\x1B[35mo\x1B[0m]");

    // now something more interesting
    std::string str4{"this part has changed, this part has been left out, this part, this part has "
                     "an extra letter"};
    std::string str5{
        "this part has degeahc, this part has, this part added, this part has ana extra letter"};

    EXPECT_THAT(
        test::inlineDiff(str5, str4),
        "this part has [\x1B[36mchanged\x1B[0m|\x1B[35mdegeahc\x1B[0m], this part has[\x1B[36m "
        "been left out\x1B[0m|\x1B[35m\x1B[0m], this part[\x1B[36m\x1B[0m|\x1B[35m added\x1B[0m], "
        "this part has an[\x1B[36m\x1B[0m|\x1B[35ma\x1B[0m] extra letter");
};

} // namespace
} // namespace ck_tile::builder
