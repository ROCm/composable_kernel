// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <sstream>

namespace ck_tile::test {

static bool isTerminalOutput() { return isatty(fileno(stdout)) || isatty(fileno(stderr)); }

// Returns a string highlighting differences between actual and expected.
// Differences are enclosed in brackets with actual and expected parts separated by '|'.
std::string inlineDiff(const std::string& actual,
                       const std::string& expected,
                       bool use_color = isTerminalOutput());

// A convenience alias for inlineDiff to improve readability in test assertions.
// Note that the function has O(n^2) complexity both in compute and in memory - do not use for very
// long strings
std::string formatInlineDiff(const std::string& actual, const std::string& expected);

// Gmock matcher for string equality with inline diff output on failure
class StringEqWithDiffMatcher : public ::testing::MatcherInterface<std::string>
{
    public:
    explicit StringEqWithDiffMatcher(const std::string& expected);

    bool MatchAndExplain(std::string actual,
                         ::testing::MatchResultListener* listener) const override;

    void DescribeTo(std::ostream* os) const override;
    void DescribeNegationTo(std::ostream* os) const override;

    private:
    std::string expected_;
};

// Factory function for the StringEqWithDiff matcher
::testing::Matcher<std::string> StringEqWithDiff(const std::string& expected);

} // namespace ck_tile::test
