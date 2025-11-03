// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "impl/conv_algorithm_types.hpp"
#include "impl/conv_signature_types.hpp"
#include "ck_tile/builder/conv_builder.hpp"

namespace ck_tile::builder::test_utils {

using namespace ck_tile::builder;
using namespace test;

class InstanceNameAsserts
{
public:
    InstanceNameAsserts& StartsWith(const char* prefix)
    {
        prefixes_.push_back(std::string(prefix));
        return *this;
    }

    InstanceNameAsserts& Contains(const char* substring)
    {
        substrings_.push_back(std::string(substring));
        return *this;
    }

    void Check(const std::string& kernel_string) const
    {
        for (const auto& prefix : prefixes_)
        {
            EXPECT_THAT(kernel_string, ::testing::StartsWith(prefix));
        }
        for (const auto& substr : substrings_)
        {
            EXPECT_THAT(kernel_string, ::testing::HasSubstr(substr));   
        }
    }
private:
    std::vector<std::string> prefixes_;
    std::vector<std::string> substrings_;
};

// Common test implementation
template <typename Builder>
constexpr void run_test(const InstanceNameAsserts& asserts)
{
    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetTypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);

    const auto invoker_ptr = instance.MakeInvokerPointer();
    EXPECT_NE(invoker_ptr, nullptr);

    asserts.Check(kernel_string);
}

} // namespace ck_tile::builder::test_utils
