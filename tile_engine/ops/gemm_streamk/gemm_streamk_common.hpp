// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "common/utils.hpp"

// Helper to extract traits from kernel name
inline GemmKernelTraits extract_traits_from_name(const std::string& kernel_name)
{
    GemmKernelTraits traits;

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
