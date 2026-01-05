// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Architecture-Specific Kernel Filtering for CK Tile Dispatcher
 *
 * Provides GPU architecture-aware validation of kernel configurations.
 * Uses arch_specs_generated.hpp as single source of truth (generated from arch_specs.json).
 *
 * Usage:
 *   ArchFilter filter("gfx942");
 *
 *   // Check if a kernel configuration is valid
 *   if (filter.is_valid(kernel_key)) {
 *       registry.register_kernel(kernel);
 *   }
 *
 *   // Get validation result with error details
 *   auto result = filter.validate(kernel_key);
 *   if (!result.valid) {
 *       for (const auto& error : result.errors) {
 *           std::cerr << error << "\n";
 *       }
 *   }
 *
 * Adding New GPU Support:
 *   1. Edit dispatcher/codegen/arch_specs.json
 *   2. Run: python dispatcher/codegen/generate_arch_specs.py
 *   3. Rebuild the dispatcher
 */

#pragma once

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/arch_specs_generated.hpp"
#include <array>
#include <string>
#include <vector>
#include <cstdint>

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// Re-export from generated header for convenience
// =============================================================================

// Use the generated types and functions from arch_specs namespace
using GpuArch        = arch_specs::GpuArch;
using WarpConfig     = arch_specs::WarpConfig;
using WarpTileConfig = std::array<int, 3>;

// Re-export string conversion functions
using arch_specs::arch_to_string;
using arch_specs::element_size;
using arch_specs::get_lds_capacity;
using arch_specs::get_supported_warp_configs;
using arch_specs::is_trait_unsupported;
using arch_specs::string_to_arch;

// =============================================================================
// Additional Helper Functions
// =============================================================================

/// Get supported warp tile configurations for arch and data types
/// This function wraps the generated data with runtime logic
inline std::vector<WarpTileConfig> get_supported_warp_tiles(GpuArch arch,
                                                            DataType dtype_a,
                                                            DataType dtype_b,
                                                            [[maybe_unused]] DataType dtype_c)
{
    // Common FP16 configurations (from arch_specs.json)
    std::vector<WarpTileConfig> fp16_configs = {
        {32, 32, 8}, {16, 16, 16}, {32, 32, 16}, {16, 16, 32}, {4, 64, 16}, {64, 4, 16}};

    // FP8 configurations
    std::vector<WarpTileConfig> fp8_gfx942 = {
        {32, 32, 16}, {32, 32, 32}, {16, 16, 32}, {16, 16, 64}};
    std::vector<WarpTileConfig> fp8_gfx950 = {
        {32, 32, 16}, {32, 32, 32}, {16, 16, 32}, {16, 16, 64}, {16, 16, 128}, {32, 32, 64}};

    // INT8 configurations
    std::vector<WarpTileConfig> int8_configs = {{16, 16, 32}, {32, 32, 16}};

    // GFX1201 only supports limited FP16
    std::vector<WarpTileConfig> rdna4_fp16 = {{16, 16, 16}};

    // Match based on architecture and data types
    if(dtype_a == DataType::FP16 && dtype_b == DataType::FP16)
    {
        if(arch == GpuArch::GFX_1201)
            return rdna4_fp16;
        return fp16_configs;
    }
    if(dtype_a == DataType::BF16 && dtype_b == DataType::BF16)
    {
        if(arch == GpuArch::GFX_1201)
            return {};       // Not supported on RDNA4
        return fp16_configs; // Same as FP16
    }
    if(dtype_a == DataType::FP8 || dtype_a == DataType::BF8)
    {
        if(arch == GpuArch::GFX_950)
            return fp8_gfx950;
        if(arch == GpuArch::GFX_942)
            return fp8_gfx942;
        if(arch == GpuArch::GFX_90A)
            return {{32, 32, 16}, {32, 32, 32}};
    }
    if(dtype_a == DataType::INT8 && dtype_b == DataType::INT8)
    {
        if(arch == GpuArch::GFX_942)
            return int8_configs;
    }

    return {}; // Unknown combination
}

// =============================================================================
// Validation Result
// =============================================================================

/// Result of kernel validation
struct ValidationResult
{
    bool valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    explicit operator bool() const { return valid; }

    void add_error(const std::string& msg)
    {
        errors.push_back(msg);
        valid = false;
    }

    void add_warning(const std::string& msg) { warnings.push_back(msg); }
};

// =============================================================================
// Architecture Filter
// =============================================================================

/**
 * Architecture-specific kernel filter.
 *
 * Validates kernel configurations against GPU architecture constraints
 * including warp configurations, warp tiles, LDS capacity, and traits.
 */
class ArchFilter
{
    public:
    /**
     * Create architecture filter.
     * @param arch Target GPU architecture
     * @param strict_mode If true, unknown configurations are rejected
     */
    explicit ArchFilter(GpuArch arch, bool strict_mode = false)
        : arch_(arch), strict_mode_(strict_mode)
    {
    }

    /**
     * Create architecture filter from string.
     * @param arch_str GPU architecture string (e.g., "gfx942")
     * @param strict_mode If true, unknown configurations are rejected
     */
    explicit ArchFilter(const std::string& arch_str, bool strict_mode = false)
        : arch_(string_to_arch(arch_str)), strict_mode_(strict_mode)
    {
    }

    /**
     * Quick validation check.
     * @param key Kernel configuration key
     * @return true if configuration is valid for this architecture
     */
    [[nodiscard]] bool is_valid(const KernelKey& key) const { return validate(key).valid; }

    /**
     * Detailed validation with error messages.
     * @param key Kernel configuration key
     * @return ValidationResult with valid flag and error/warning messages
     */
    [[nodiscard]] ValidationResult validate(const KernelKey& key) const
    {
        ValidationResult result;

        // Check architecture match
        if(!key.gfx_arch.empty() && string_to_arch(key.gfx_arch) != arch_)
        {
            result.add_warning("Kernel compiled for different architecture: " + key.gfx_arch);
        }

        // Validate dimensions
        validate_dimensions(key, result);

        // Validate warp configuration
        validate_warp_config(key, result);

        // Validate warp tile configuration
        validate_warp_tiles(key, result);

        // Validate trait combination
        validate_traits(key, result);

        // Validate LDS capacity
        validate_lds(key, result);

        return result;
    }

    /// Get target architecture
    [[nodiscard]] GpuArch get_arch() const { return arch_; }

    /// Get target architecture as string
    [[nodiscard]] std::string get_arch_string() const { return arch_to_string(arch_); }

    private:
    void validate_dimensions(const KernelKey& key, ValidationResult& result) const
    {
        const auto& alg = key.algorithm;

        // Check positive dimensions
        if(alg.tile_shape.m <= 0 || alg.tile_shape.n <= 0 || alg.tile_shape.k <= 0)
        {
            result.add_error("Tile dimensions must be positive");
            return;
        }

        // Check warp tiles fit in block tiles
        int warp_m_coverage = alg.wave_shape.m * alg.warp_tile_shape.m;
        int warp_n_coverage = alg.wave_shape.n * alg.warp_tile_shape.n;
        int warp_k_coverage = alg.wave_shape.k * alg.warp_tile_shape.k;

        if(warp_m_coverage > alg.tile_shape.m)
        {
            result.add_error("warp_m * warp_tile_m > tile_m: " + std::to_string(warp_m_coverage) +
                             " > " + std::to_string(alg.tile_shape.m));
        }
        if(warp_n_coverage > alg.tile_shape.n)
        {
            result.add_error("warp_n * warp_tile_n > tile_n: " + std::to_string(warp_n_coverage) +
                             " > " + std::to_string(alg.tile_shape.n));
        }
        if(warp_k_coverage > alg.tile_shape.k)
        {
            result.add_error("warp_k * warp_tile_k > tile_k: " + std::to_string(warp_k_coverage) +
                             " > " + std::to_string(alg.tile_shape.k));
        }

        // Check alignment
        if(alg.tile_shape.m % warp_m_coverage != 0)
        {
            result.add_error("tile_m must be divisible by warp_m * warp_tile_m");
        }
        if(alg.tile_shape.n % warp_n_coverage != 0)
        {
            result.add_error("tile_n must be divisible by warp_n * warp_tile_n");
        }
        if(alg.tile_shape.k % warp_k_coverage != 0)
        {
            result.add_error("tile_k must be divisible by warp_k * warp_tile_k");
        }
    }

    void validate_warp_config(const KernelKey& key, ValidationResult& result) const
    {
        auto supported = get_supported_warp_configs(arch_);
        if(supported.empty())
        {
            if(strict_mode_)
            {
                result.add_error("No warp configurations defined for " + get_arch_string());
            }
            else
            {
                result.add_warning("No warp configurations defined for " + get_arch_string());
            }
            return;
        }

        WarpConfig current = {
            key.algorithm.wave_shape.m, key.algorithm.wave_shape.n, key.algorithm.wave_shape.k};

        bool found = false;
        for(const auto& cfg : supported)
        {
            if(cfg == current)
            {
                found = true;
                break;
            }
        }

        if(!found)
        {
            result.add_error("Invalid warp configuration [" + std::to_string(current[0]) + ", " +
                             std::to_string(current[1]) + ", " + std::to_string(current[2]) +
                             "] for " + get_arch_string());
        }
    }

    void validate_warp_tiles(const KernelKey& key, ValidationResult& result) const
    {
        auto supported = get_supported_warp_tiles(
            arch_, key.signature.dtype_a, key.signature.dtype_b, key.signature.dtype_c);

        if(supported.empty())
        {
            // Unknown data type combination - allow with warning
            result.add_warning("No warp tile combinations defined for data types");
            return;
        }

        WarpTileConfig current = {key.algorithm.warp_tile_shape.m,
                                  key.algorithm.warp_tile_shape.n,
                                  key.algorithm.warp_tile_shape.k};

        bool found = false;
        for(const auto& cfg : supported)
        {
            if(cfg == current)
            {
                found = true;
                break;
            }
        }

        if(!found)
        {
            result.add_error("Invalid warp tile [" + std::to_string(current[0]) + ", " +
                             std::to_string(current[1]) + ", " + std::to_string(current[2]) +
                             "] for " + get_arch_string());
        }
    }

    void validate_traits(const KernelKey& key, ValidationResult& result) const
    {
        if(is_trait_unsupported(
               key.algorithm.pipeline, key.algorithm.epilogue, key.algorithm.scheduler))
        {
            result.add_error("Unsupported trait combination");
        }
    }

    void validate_lds(const KernelKey& key, ValidationResult& result) const
    {
        const auto& sig = key.signature;
        const auto& alg = key.algorithm;

        float elem_a = element_size(sig.dtype_a);
        float elem_b = element_size(sig.dtype_b);

        std::size_t matrix_a_size = alg.tile_shape.m * alg.tile_shape.k * elem_a;
        std::size_t matrix_b_size = alg.tile_shape.n * alg.tile_shape.k * elem_b;
        std::size_t total_lds     = matrix_a_size + matrix_b_size;

        std::size_t max_lds = get_lds_capacity(alg.pipeline);

        if(total_lds > max_lds)
        {
            result.add_error("LDS capacity exceeded: " + std::to_string(total_lds) + " bytes > " +
                             std::to_string(max_lds) + " bytes limit");
        }
    }

    GpuArch arch_;
    bool strict_mode_;
};

// =============================================================================
// Registry Integration Helper
// =============================================================================

/**
 * Create a filter function for use with Registry::filter()
 *
 * @tparam KernelT Kernel instance type with get_key() method
 * @param arch Target GPU architecture
 * @return Predicate function that returns true for valid kernels
 */
template <typename KernelT>
inline auto make_arch_filter_predicate(const std::string& arch)
{
    return [filter = ArchFilter(arch)](const KernelT& kernel) {
        return filter.is_valid(kernel.get_key());
    };
}

} // namespace dispatcher
} // namespace ck_tile
