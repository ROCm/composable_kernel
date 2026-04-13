// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace ck_tile {
namespace dispatcher {

/// Log levels for dispatcher transparency:
///   0 = silent (default)
///   1 = print selected kernel name
///   2 = print all candidates considered and acceptance/rejection reasons
inline int get_log_level()
{
    static int level = []() {
        const char* env = std::getenv("CK_DISPATCHER_LOG_LEVEL");
        return env ? std::atoi(env) : 0;
    }();
    return level;
}

inline void log_kernel_selected(const std::string& kernel_name, const std::string& problem_desc)
{
    if(get_log_level() >= 1)
    {
        std::cerr << "[CK Dispatcher] Selected kernel: " << kernel_name << " for " << problem_desc
                  << std::endl;
    }
}

inline void
log_kernel_candidate(const std::string& kernel_name, bool accepted, const std::string& reason)
{
    if(get_log_level() >= 2)
    {
        std::cerr << "[CK Dispatcher]   Candidate: " << kernel_name << " -> "
                  << (accepted ? "ACCEPTED" : "REJECTED")
                  << (reason.empty() ? "" : " (" + reason + ")") << std::endl;
    }
}

inline void log_no_kernel_found(const std::string& problem_desc)
{
    if(get_log_level() >= 1)
    {
        std::cerr << "[CK Dispatcher] No kernel found for " << problem_desc << std::endl;
    }
}

} // namespace dispatcher
} // namespace ck_tile
