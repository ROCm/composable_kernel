// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

// Structure to hold kernel traits for dispatcher
struct KernelTraits
{
    std::string pipeline;  // compv3, compv4, mem
    std::string scheduler; // intrawave, interwave
    std::string epilogue;  // cshuffle, default
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool persistent;

    // Constructor with defaults
    KernelTraits()
        : pipeline("compv3"),
          scheduler("intrawave"),
          epilogue("cshuffle"),
          pad_m(false),
          pad_n(false),
          pad_k(false),
          persistent(false)
    {
    }
};

// Helper to extract traits from kernel name
inline KernelTraits extract_traits_from_name(const std::string& kernel_name)
{
    KernelTraits traits;

    // Extract pipeline
    if(kernel_name.find("compv3") != std::string::npos)
    {
        traits.pipeline = "compv3";
    }
    else if(kernel_name.find("compv4") != std::string::npos)
    {
        traits.pipeline = "compv4";
    }
    else if(kernel_name.find("mem") != std::string::npos)
    {
        traits.pipeline = "mem";
    }

    // Extract scheduler
    if(kernel_name.find("interwave") != std::string::npos)
    {
        traits.scheduler = "interwave";
    }
    else
    {
        traits.scheduler = "intrawave";
    }

    // Extract epilogue
    if(kernel_name.find("default") != std::string::npos &&
       kernel_name.find("default_") == std::string::npos)
    {
        traits.epilogue = "default";
    }
    else
    {
        traits.epilogue = "cshuffle";
    }

    // Padding flags would need to be extracted from the kernel configuration
    // For now, we'll leave them as false

    return traits;
}
