// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Provides a base class for generating human-readable descriptions of kernel instances.
///
/// This file contains the Description base class that defines a common interface for
/// all descriptor types. Derived classes implement specific formatting and explanation
/// logic for different kernel types (e.g., convolution, GEMM, etc.).

#pragma once

#include <string>

namespace ck_tile::reflect {

/// @brief Base class for generating human-readable descriptions of kernel instances
/// Defines a common interface for all descriptor types with methods for generating
/// descriptions at various levels of detail.
class Description
{
    public:
    /// @brief Virtual destructor for proper cleanup of derived classes
    virtual ~Description() = default;

    /// @brief Generate a brief one-line summary
    /// @return A concise description of the kernel configuration
    virtual std::string brief() const = 0;

    /// @brief Generate a detailed hierarchical description
    /// @return A multi-line tree-formatted description covering all configuration details
    virtual std::string detailed() const = 0;

    /// @brief Generate a string representation of the instance
    /// @return A string that represents the instance
    virtual std::string instance_string() const = 0;
};

} // namespace ck_tile::reflect
