// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file conv_utils.hpp
 * @brief CK Tile Convolution Dispatcher Utilities
 *
 * Common utilities for convolution kernel specification using the
 * Signature/Algorithm/Arch pattern from experimental/builder/reflect.
 *
 * Structure:
 *   - Signature: WHAT operation (types, layouts, direction, element ops)
 *   - Algorithm: HOW it's computed (tiles, warps, pipeline, scheduler, padding)
 *   - Arch:      WHERE it runs (target GPU architecture)
 *
 * Usage:
 *   #include "ck_tile/dispatcher/conv_utils.hpp"
 *
 *   using namespace ck_tile::dispatcher;
 *
 *   // Define signature (WHAT)
 *   auto sig = ConvSig().dtype("fp16").layout("nhwc").conv_type("forward");
 *
 *   // Define algorithm (HOW)
 *   auto algo = ConvAlgo().tile(1, 128, 128).wave(2, 2, 1).warp(32, 32, 16);
 *
 *   // Create config
 *   ConvKernelConfig config(sig, algo, "gfx942");
 */

#pragma once

// Core convolution headers
#include "ck_tile/dispatcher/conv_config.hpp"
#include "ck_tile/dispatcher/conv_kernel_decl.hpp"
#include "ck_tile/dispatcher/conv_problem.hpp"
#include "ck_tile/dispatcher/conv_registry.hpp"

// Common dispatcher utilities
#include "ck_tile/dispatcher/arch_filter.hpp"
#include "ck_tile/dispatcher/utils.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <functional>

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// TYPE ALIASES for cleaner example code
// =============================================================================

/// Signature alias (WHAT operation)
using ConvSig = conv_decl::ConvSignature;

/// Algorithm alias (HOW computed)
using ConvAlgo = conv_decl::ConvAlgorithm;

// =============================================================================
// CONVENIENCE CONFIG CREATORS
// =============================================================================

namespace conv_utils {

/**
 * @brief Create a 2D forward convolution config
 * @param dtype Data type (fp16, fp32, bf16)
 * @param tile_k K tile size
 * @param tile_c C tile size
 * @param arch Target architecture
 */
inline ConvKernelDecl create_conv2d_fwd(const std::string& dtype = "fp16",
                                        int tile_k               = 128,
                                        int tile_c               = 128,
                                        const std::string& arch  = "gfx942")
{
    return ConvKernelDecl(
        ConvSig().dtype(dtype).layout("nhwc").conv_type("forward").dims(2),
        ConvAlgo().tile(1, tile_k, tile_c).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
        arch);
}

/**
 * @brief Create a 3D forward convolution config
 */
inline ConvKernelDecl create_conv3d_fwd(const std::string& dtype = "fp16",
                                        int tile_k               = 64,
                                        int tile_c               = 64,
                                        const std::string& arch  = "gfx942")
{
    return ConvKernelDecl(
        ConvSig().dtype(dtype).layout("ndhwc").conv_type("forward").dims(3),
        ConvAlgo().tile(1, tile_k, tile_c).wave(2, 2, 1).warp(16, 16, 32).pipeline("compv3"),
        arch);
}

/**
 * @brief Create a 2D backward data convolution config
 */
inline ConvKernelDecl create_conv2d_bwd_data(const std::string& dtype = "fp16",
                                             int tile_k               = 128,
                                             int tile_c               = 128,
                                             const std::string& arch  = "gfx942")
{
    return ConvKernelDecl(
        ConvSig().dtype(dtype).layout("nhwc").conv_type("bwd_data").dims(2),
        ConvAlgo().tile(1, tile_k, tile_c).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
        arch);
}

/**
 * @brief Create a 2D backward weight convolution config
 */
inline ConvKernelDecl create_conv2d_bwd_weight(const std::string& dtype = "fp16",
                                               int tile_k               = 128,
                                               int tile_c               = 128,
                                               const std::string& arch  = "gfx942")
{
    return ConvKernelDecl(
        ConvSig().dtype(dtype).layout("nhwc").conv_type("bwd_weight").dims(2),
        ConvAlgo().tile(1, tile_k, tile_c).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
        arch);
}

// =============================================================================
// PROBLEM CREATION HELPERS
// =============================================================================

/**
 * @brief Create a standard 2D conv problem
 */
inline ConvProblem create_conv2d_problem(int N,
                                         int C,
                                         int K,
                                         int Hi,
                                         int Wi,
                                         int Y,
                                         int X,
                                         int stride  = 1,
                                         int padding = 0,
                                         ConvOp op   = ConvOp::Forward)
{
    ConvProblem p;
    p.N              = N;
    p.C              = C;
    p.K              = K;
    p.G              = 1;
    p.input_spatial  = {1, Hi, Wi};
    p.filter_spatial = {1, Y, X};
    p.stride         = {1, stride, stride};
    p.padding        = {0, padding, padding};
    p.dilation       = {1, 1, 1};
    p.op             = op;
    p.compute_output_size();
    return p;
}

/**
 * @brief Create a standard 3D conv problem
 */
inline ConvProblem create_conv3d_problem(int N,
                                         int C,
                                         int K,
                                         int Di,
                                         int Hi,
                                         int Wi,
                                         int Z,
                                         int Y,
                                         int X,
                                         int stride  = 1,
                                         int padding = 0,
                                         ConvOp op   = ConvOp::Forward)
{
    ConvProblem p;
    p.N              = N;
    p.C              = C;
    p.K              = K;
    p.G              = 1;
    p.input_spatial  = {Di, Hi, Wi};
    p.filter_spatial = {Z, Y, X};
    p.stride         = {stride, stride, stride};
    p.padding        = {padding, padding, padding};
    p.dilation       = {1, 1, 1};
    p.op             = op;
    p.compute_output_size();
    return p;
}

/**
 * @brief Create a depthwise 2D conv problem
 */
inline ConvProblem create_depthwise_conv2d_problem(
    int N, int C, int Hi, int Wi, int Y, int X, int stride = 1, int padding = 0)
{
    ConvProblem p;
    p.N              = N;
    p.C              = C;
    p.K              = C; // K = C for depthwise
    p.G              = C; // G = C for depthwise
    p.input_spatial  = {1, Hi, Wi};
    p.filter_spatial = {1, Y, X};
    p.stride         = {1, stride, stride};
    p.padding        = {0, padding, padding};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();
    return p;
}

// =============================================================================
// PRINTING UTILITIES
// =============================================================================

/**
 * @brief Print Signature/Algorithm/Arch pattern documentation
 */
inline void print_pattern_docs(std::ostream& os = std::cout)
{
    os << "SIGNATURE (WHAT operation):\n";
    os << "  - dtype_in_, dtype_wei_, dtype_out_, dtype_acc_  : Data types\n";
    os << "  - layout_                                         : nhwc, nchw\n";
    os << "  - conv_op_                                        : forward, bwd_data, bwd_weight\n";
    os << "  - num_dims_                                       : 1, 2, 3\n";
    os << "  - groups_                                         : Group count\n\n";

    os << "ALGORITHM (HOW it's computed):\n";
    os << "  - tile_n_, tile_k_, tile_c_         : Block tile dimensions\n";
    os << "  - tile_ho_, tile_wo_                : Output spatial tile\n";
    os << "  - wave_m_, wave_n_, wave_k_         : Warp distribution\n";
    os << "  - warp_m_, warp_n_, warp_k_         : Warp tile sizes\n";
    os << "  - pipeline_                         : compv3, compv4, compv5, mem\n";
    os << "  - scheduler_                        : intrawave, interwave\n\n";

    os << "ARCH (WHERE it runs):\n";
    os << "  - gfx942 (MI300X), gfx90a (MI200), gfx1100 (Navi31)\n";
}

/**
 * @brief Print a detailed view of a ConvKernelDecl
 */
inline void print_kernel_decl(const ConvKernelDecl& decl, std::ostream& os = std::cout)
{
    const auto& sig  = decl.signature;
    const auto& algo = decl.algorithm;

    os << "Convolution Kernel: " << decl.name() << "\n";
    os << "  Signature (WHAT):\n";
    os << "    Data Type:     " << sig.dtype_in_ << " -> " << sig.dtype_out_
       << " (acc: " << sig.dtype_acc_ << ")\n";
    os << "    Layout:        " << sig.layout_ << "\n";
    os << "    Direction:     " << sig.conv_op_ << "\n";
    os << "    Spatial Dims:  " << sig.num_dims_ << "D\n";
    os << "    Groups:        " << sig.groups_ << "\n";

    os << "  Algorithm (HOW):\n";
    os << "    Block Tile:    N=" << algo.tile_n_ << ", K=" << algo.tile_k_
       << ", C=" << algo.tile_c_ << "\n";
    os << "    Output Tile:   Ho=" << algo.tile_ho_ << ", Wo=" << algo.tile_wo_ << "\n";
    os << "    Wave Config:   " << algo.wave_m_ << "x" << algo.wave_n_ << "x" << algo.wave_k_
       << "\n";
    os << "    Warp Tile:     " << algo.warp_m_ << "x" << algo.warp_n_ << "x" << algo.warp_k_
       << "\n";
    os << "    Pipeline:      " << algo.pipeline_ << "\n";
    os << "    Scheduler:     " << algo.scheduler_ << "\n";

    os << "  Arch (WHERE):\n";
    os << "    Target:        " << decl.arch << "\n";
}

/**
 * @brief Print problem details
 */
inline void print_problem(const ConvProblem& p, std::ostream& os = std::cout)
{
    os << "ConvProblem:\n";
    os << "  Batch:     N=" << p.N << "\n";
    os << "  Channels:  C=" << p.C << ", K=" << p.K << ", G=" << p.G << "\n";
    os << "  Input:     ";
    for(size_t i = 0; i < p.input_spatial.size(); i++)
    {
        if(i > 0)
            os << "x";
        os << p.input_spatial[i];
    }
    os << "\n";
    os << "  Filter:    ";
    for(size_t i = 0; i < p.filter_spatial.size(); i++)
    {
        if(i > 0)
            os << "x";
        os << p.filter_spatial[i];
    }
    os << "\n";
    os << "  Output:    ";
    for(size_t i = 0; i < p.output_spatial.size(); i++)
    {
        if(i > 0)
            os << "x";
        os << p.output_spatial[i];
    }
    os << "\n";
    os << "  FLOPs:     " << std::scientific << std::setprecision(2) << p.get_flops() << "\n";
    os << "  Pointwise: " << (p.is_pointwise() ? "Yes" : "No") << "\n";
    os << "  Depthwise: " << (p.is_depthwise() ? "Yes" : "No") << "\n";
}

// =============================================================================
// KERNEL SET BUILDING UTILITIES
// =============================================================================

/**
 * @brief Build a standard 2D forward kernel set
 */
inline ConvKernelSet build_conv2d_fwd_set(const std::string& dtype = "fp16",
                                          const std::string& arch  = "gfx942")
{
    ConvKernelSet set;

    // Small tiles for latency
    set.add(ConvSig().dtype(dtype).layout("nhwc").conv_type("forward").dims(2),
            ConvAlgo().tile(1, 64, 64).wave(2, 2, 1).warp(16, 16, 32).pipeline("compv3"),
            arch);

    // Medium tiles for balanced
    set.add(ConvSig().dtype(dtype).layout("nhwc").conv_type("forward").dims(2),
            ConvAlgo().tile(1, 128, 128).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
            arch);

    // Large tiles for throughput
    set.add(ConvSig().dtype(dtype).layout("nhwc").conv_type("forward").dims(2),
            ConvAlgo().tile(1, 256, 256).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
            arch);

    return set;
}

/**
 * @brief Build a comprehensive kernel set for all 2D operations
 */
inline ConvKernelSet build_conv2d_full_set(const std::string& dtype = "fp16",
                                           const std::string& arch  = "gfx942")
{
    ConvKernelSet set;

    // Forward kernels
    set.add(ConvSig().dtype(dtype).layout("nhwc").conv_type("forward").dims(2),
            ConvAlgo().tile(1, 128, 128).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
            arch);

    // Backward data kernels
    set.add(ConvSig().dtype(dtype).layout("nhwc").conv_type("bwd_data").dims(2),
            ConvAlgo().tile(1, 128, 128).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
            arch);

    // Backward weight kernels
    set.add(ConvSig().dtype(dtype).layout("nhwc").conv_type("bwd_weight").dims(2),
            ConvAlgo().tile(1, 128, 128).wave(2, 2, 1).warp(32, 32, 16).pipeline("compv4"),
            arch);

    return set;
}

// =============================================================================
// VALIDATION UTILITIES
// =============================================================================

/**
 * @brief Validation result structure
 */
struct ValidationResult
{
    bool passed        = false;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float rtol         = 1e-3f;
    float atol         = 1e-3f;

    void print(std::ostream& os = std::cout) const
    {
        os << "Validation: " << (passed ? "PASSED" : "FAILED") << "\n";
        os << "  Max abs diff: " << std::scientific << max_abs_diff << "\n";
        os << "  Max rel diff: " << std::scientific << max_rel_diff << "\n";
        os << "  Tolerances:   rtol=" << rtol << ", atol=" << atol << "\n";
    }
};

/**
 * @brief Compare two buffers for equality within tolerance
 */
template <typename T>
inline ValidationResult validate_buffers(
    const T* result, const T* reference, size_t count, float rtol = 1e-3f, float atol = 1e-3f)
{
    ValidationResult res;
    res.rtol   = rtol;
    res.atol   = atol;
    res.passed = true;

    for(size_t i = 0; i < count; ++i)
    {
        float r   = static_cast<float>(result[i]);
        float ref = static_cast<float>(reference[i]);

        float abs_diff = std::abs(r - ref);
        float rel_diff = abs_diff / (std::abs(ref) + 1e-10f);

        res.max_abs_diff = std::max(res.max_abs_diff, abs_diff);
        res.max_rel_diff = std::max(res.max_rel_diff, rel_diff);

        if(abs_diff > atol + rtol * std::abs(ref))
        {
            res.passed = false;
        }
    }

    return res;
}

// =============================================================================
// BENCHMARK UTILITIES
// =============================================================================

/**
 * @brief Benchmark result structure
 */
struct BenchmarkResult
{
    std::string kernel_name;
    float time_ms      = 0.0f;
    float tflops       = 0.0f;
    int warmup_runs    = 0;
    int benchmark_runs = 0;

    void print(std::ostream& os = std::cout) const
    {
        os << "Benchmark: " << kernel_name << "\n";
        os << "  Time:    " << std::fixed << std::setprecision(3) << time_ms << " ms\n";
        os << "  TFLOPS:  " << std::fixed << std::setprecision(2) << tflops << "\n";
        os << "  Runs:    " << warmup_runs << " warmup, " << benchmark_runs << " timed\n";
    }
};

/**
 * @brief Calculate TFLOPS from time and FLOPs
 */
inline float calc_tflops(double flops, float time_ms)
{
    return static_cast<float>(flops / (time_ms * 1e9));
}

} // namespace conv_utils

// =============================================================================
// EXAMPLE TEMPLATES
// =============================================================================

namespace examples {

/**
 * @brief Template for a basic conv example main function
 */
inline int basic_conv_example_main(const std::string& example_name)
{
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Example: " << example_name << "\n";
    std::cout << std::string(70, '=') << "\n\n";

    // Print pattern documentation
    std::cout << "PATTERN STRUCTURE\n";
    std::cout << std::string(40, '-') << "\n";
    conv_utils::print_pattern_docs();
    std::cout << "\n";

    // Show declared kernel sets
    std::cout << "DECLARED KERNEL SETS\n";
    std::cout << std::string(40, '-') << "\n";
    ConvKernelSetRegistry::instance().print();
    std::cout << "\n";

    return 0;
}

} // namespace examples

} // namespace dispatcher
} // namespace ck_tile
