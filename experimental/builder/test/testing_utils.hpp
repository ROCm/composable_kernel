// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/device_operation_instance_factory.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <sstream>
#include <iosfwd>
#include <set>
#include <vector>
#include <array>

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

using ck::tensor_operation::device::instance::DeviceOperationInstanceFactory;

// This utility concept checks whether a type is a valid "Device Operation" -
// that is, there is a valid specialization of `DeviceOperationInstanceFactory`
// for it available.
template <typename DeviceOp>
concept HasCkFactory = requires {
    {
        DeviceOperationInstanceFactory<DeviceOp>::GetInstances()
    } -> std::convertible_to<std::vector<std::unique_ptr<DeviceOp>>>;
};

// This structure represents a (unique) set of instances, either a statically
// defined one (for testing) or one obtained from DeviceOperationInstanceFactory.
// The idea is that we use this structure as a utility to compare a set of
// instances. Instances are stored in a set so that they can be lexicographically
// compared, this helps generating readable error messages which just contain
// the differenses between sets.
struct InstanceSet
{
    explicit InstanceSet() {}

    explicit InstanceSet(std::initializer_list<const char*> items)
        : instances(items.begin(), items.end())
    {
    }

    template <HasCkFactory DeviceOp>
    static InstanceSet from_factory()
    {
        auto set = InstanceSet();

        const auto ops = DeviceOperationInstanceFactory<DeviceOp>::GetInstances();
        for(const auto& op : ops)
        {
            set.instances.insert(op->GetInstanceString());
        }

        return set;
    }

    std::set<std::string> instances;
};

std::ostream& operator<<(std::ostream& os, const InstanceSet& set);

// This is a custom Google Test matcher which can be used to compare two sets
// of instance names, with utility functions that print a helpful error
// message about the difference between the checked sets. Use `InstancesMatch`
// to obtain an instance of this type.
struct InstanceMatcher : public ::testing::MatcherInterface<InstanceSet>
{
    explicit InstanceMatcher(const InstanceSet& expected);

    bool MatchAndExplain(InstanceSet actual,
                         ::testing::MatchResultListener* listener) const override;
    void DescribeTo(std::ostream* os) const override;
    void DescribeNegationTo(std::ostream* os) const override;

    InstanceSet expected_;
};

::testing::Matcher<InstanceSet> InstancesMatch(const InstanceSet& expected);

} // namespace ck_tile::test
