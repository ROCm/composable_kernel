// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "ck_tile/core/utility/print.hpp"
#include "ck_tile/core/container/sequence.hpp"

namespace ck_tile {

class PrintSequenceTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PrintSequenceTest, PrintSimpleSequence)
{
    // Test printing sequence<1, 5, 8>
    constexpr auto seq = sequence<1, 5, 8>{};

    // Capture stdout
    testing::internal::CaptureStdout();

    // Call print function
    print(seq);

    // Get captured output
    std::string output = testing::internal::GetCapturedStdout();

    // Verify the output format
    EXPECT_EQ(output, "sequence<1, 5, 8>");
}

TEST_F(PrintSequenceTest, PrintSingleElementSequence)
{
    // Test printing sequence<42>
    constexpr auto seq = sequence<42>{};

    testing::internal::CaptureStdout();
    print(seq);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "sequence<42>");
}

TEST_F(PrintSequenceTest, PrintEmptySequence)
{
    // Test printing sequence<> (empty sequence)
    constexpr auto seq = sequence<>{};

    testing::internal::CaptureStdout();
    print(seq);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "sequence<>");
}

} // namespace ck_tile
