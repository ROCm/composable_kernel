// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <sstream>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/pooling.hpp"

namespace ck_tile {

/// @brief Kernel trait parameters for pooling tile_engine configurations
struct PoolingKernelTraits
{
    std::string reduce_op; // "max", "min", or "avg"
    bool output_index;     // Whether to output indices (max pooling)
    bool propagate_nan;    // Whether to propagate NaN values
    bool cross_warp;       // Whether cross-warp reduction is used

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << reduce_op << "_" << (output_index ? "idx" : "noidx") << "_"
            << (propagate_nan ? "nan" : "nonan") << "_"
            << (cross_warp ? "crosswarp" : "nocrosswarp");
        return oss.str();
    }
};

/// @brief Extract traits from a kernel name string
inline PoolingKernelTraits extract_pooling_traits_from_name(const std::string& name)
{
    PoolingKernelTraits traits;
    if(name.find("max") != std::string::npos)
        traits.reduce_op = "max";
    else if(name.find("min") != std::string::npos)
        traits.reduce_op = "min";
    else
        traits.reduce_op = "avg";
    traits.output_index =
        (name.find("idx") != std::string::npos) && (name.find("noidx") == std::string::npos);
    traits.propagate_nan =
        (name.find("nan") != std::string::npos) && (name.find("nonan") == std::string::npos);
    traits.cross_warp = (name.find("crosswarp") != std::string::npos) &&
                        (name.find("nocrosswarp") == std::string::npos);
    return traits;
}

} // namespace ck_tile
