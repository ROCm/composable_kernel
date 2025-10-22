// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "testing_utils.hpp"

#include <ostream>
#include <algorithm>

namespace ck_tile::test {

std::ostream& operator<<(std::ostream& os, const InstanceSet& set)
{
    // These sets can grow very large, and so its not very nice or useful to print them
    // in the event of a mismatch. Just print a brief description here, and use
    // InstancesMatcher to print a more useful message.
    return (os << "(set of " << set.instances.size() << " instances)");
}

InstanceMatcher::InstanceMatcher(const InstanceSet& expected) : expected_(expected) {}

::testing::Matcher<InstanceSet> InstancesMatch(const InstanceSet& expected)
{
    return ::testing::MakeMatcher(new InstanceMatcher(expected));
}

bool InstanceMatcher::MatchAndExplain(InstanceSet actual,
                                      ::testing::MatchResultListener* listener) const
{
    if(actual.instances == expected_.instances)
    {
        return true;
    }

    if(listener->IsInterested())
    {
        std::vector<std::string> instances;
        std::set_difference(expected_.instances.begin(),
                            expected_.instances.end(),
                            actual.instances.begin(),
                            actual.instances.end(),
                            std::back_inserter(instances));

        *listener << "\n";

        if(instances.size() > 0)
        {
            *listener << " Missing: " << instances.size() << "\n";
            for(const auto& instance : instances)
            {
                *listener << "- " << instance << "\n";
            }
        }

        instances.clear();
        std::set_difference(actual.instances.begin(),
                            actual.instances.end(),
                            expected_.instances.begin(),
                            expected_.instances.end(),
                            std::back_inserter(instances));

        if(instances.size() > 0)
        {
            *listener << "Unexpected: " << instances.size() << "\n";
            for(const auto& instance : instances)
            {
                if(instance == "")
                {
                    *listener << "- (empty string; indicates missing GetInstanceString() overload)"
                              << "\n";
                }
                else
                {
                    *listener << "- " << instance << "\n";
                }
            }
        }
    }

    return false;
}

void InstanceMatcher::DescribeTo(std::ostream* os) const { *os << expected_; }

void InstanceMatcher::DescribeNegationTo(std::ostream* os) const
{
    *os << "is not equal to " << expected_;
}

} // namespace ck_tile::test
