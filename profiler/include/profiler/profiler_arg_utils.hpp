// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <string>
#include "ck/ck.hpp"

namespace ck {
namespace profiler {

// Parse optional named arguments from argv
// Modifies instance_index and list_instances based on found arguments
inline void
parse_named_args(int argc, char* argv[], ck::index_t& instance_index, bool& list_instances)
{
    instance_index = -1;
    list_instances = false;

    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--instance") == 0 && i + 1 < argc)
        {
            instance_index = std::stoi(argv[i + 1]);
        }
        else if(std::strcmp(argv[i], "--list-instances") == 0)
        {
            list_instances = true;
        }
    }
}

// Count named arguments to adjust argc for positional arg checking
inline int count_named_args(int argc, char* argv[])
{
    int count = 0;
    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--instance") == 0)
        {
            count += 2;
            ++i; // skip the value
        }
        else if(std::strcmp(argv[i], "--list-instances") == 0)
        {
            count += 1;
        }
    }
    return count;
}

} // namespace profiler
} // namespace ck
