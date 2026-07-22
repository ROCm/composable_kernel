// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "gemm/gemm_common.hpp"

// Structure to hold kernel traits for dispatcher
struct PreshuffleKernelTraits : GemmKernelTraits
{

    // Constructor with defaults
    PreshuffleKernelTraits() : GemmKernelTraits() { this->pipeline = "preshufflev2"; }
};

// Helper to extract traits from kernel name
inline PreshuffleKernelTraits extract_traits_from_name(const std::string& kernel_name)
{
    PreshuffleKernelTraits traits;

    // Extract pipeline
    if(kernel_name.find("preshufflev2") != std::string::npos)
    {
        traits.pipeline = "preshufflev2";
    }

    // Extract scheduler
    if(kernel_name.find("interwave") != std::string::npos)
    {
        traits.scheduler = "interwave";
    }
    else if(kernel_name.find("intrawave") != std::string::npos)
    {
        traits.scheduler = "intrawave";
    }
    else
    {
        traits.scheduler = "default";
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
