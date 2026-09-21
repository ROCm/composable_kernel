// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file grouped_conv_utils.hpp
 * @brief CK Tile Grouped Convolution Dispatcher Utilities
 */

#pragma once

#include "ck_tile/dispatcher/grouped_conv_config.hpp"
#include "ck_tile/dispatcher/grouped_conv_kernel_decl.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/arch_filter.hpp"
#include "ck_tile/dispatcher/utils.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <cmath>
#include <algorithm>

namespace ck_tile {
namespace dispatcher {

using GroupedConvSig  = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgo = grouped_conv_decl::GroupedConvAlgorithm;

namespace grouped_conv_utils {

inline GroupedConvKernelDecl create_grouped_conv2d_fwd(const std::string& dtype = "fp16",
                                                       int tile_n               = 128,
                                                       int tile_k               = 128,
                                                       const std::string& arch  = "gfx942")
{
    return GroupedConvKernelDecl(
        GroupedConvSig().dtype(dtype).layout("nhwc").conv_type("forward").dims(2),
        GroupedConvAlgo()
            .tile(1, tile_n, tile_k)
            .wave(2, 2, 1)
            .warp(32, 32, 16)
            .pipeline("compv4")
            .vector_sizes(4, 8, 8),
        arch);
}

inline GroupedConvKernelDecl create_grouped_conv3d_fwd(const std::string& dtype = "fp16",
                                                       int tile_n               = 64,
                                                       int tile_k               = 64,
                                                       const std::string& arch  = "gfx942")
{
    return GroupedConvKernelDecl(
        GroupedConvSig().dtype(dtype).layout("ndhwc").conv_type("forward").dims(3),
        GroupedConvAlgo()
            .tile(1, tile_n, tile_k)
            .wave(2, 2, 1)
            .warp(16, 16, 32)
            .pipeline("compv3")
            .vector_sizes(4, 8, 8),
        arch);
}

inline GroupedConvKernelDecl create_grouped_conv2d_bwd_data(const std::string& dtype = "fp16",
                                                            int tile_n               = 128,
                                                            int tile_k               = 128,
                                                            const std::string& arch  = "gfx942")
{
    return GroupedConvKernelDecl(
        GroupedConvSig().dtype(dtype).layout("nhwc").conv_type("bwd_data").dims(2),
        GroupedConvAlgo()
            .tile(1, tile_n, tile_k)
            .wave(2, 2, 1)
            .warp(32, 32, 16)
            .pipeline("compv3")
            .vector_sizes(4, 8, 8),
        arch);
}

inline GroupedConvKernelDecl create_grouped_conv2d_bwd_weight(const std::string& dtype = "fp16",
                                                              int tile_n               = 128,
                                                              int tile_k               = 128,
                                                              const std::string& arch  = "gfx942")
{
    return GroupedConvKernelDecl(
        GroupedConvSig().dtype(dtype).layout("nhwc").conv_type("bwd_weight").dims(2),
        GroupedConvAlgo()
            .tile(1, tile_n, tile_k)
            .wave(2, 2, 1)
            .warp(32, 32, 16)
            .pipeline("compv3")
            .memory_op("atomic_add")
            .vector_sizes(4, 8, 8),
        arch);
}

inline GroupedConvProblem create_grouped_conv2d_problem(int N,
                                                        int C,
                                                        int K,
                                                        int Hi,
                                                        int Wi,
                                                        int Y,
                                                        int X,
                                                        int stride       = 1,
                                                        int padding      = 0,
                                                        GroupedConvOp op = GroupedConvOp::Forward)
{
    GroupedConvProblem p;
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

inline GroupedConvProblem create_grouped_conv3d_problem(int N,
                                                        int C,
                                                        int K,
                                                        int Di,
                                                        int Hi,
                                                        int Wi,
                                                        int Z,
                                                        int Y,
                                                        int X,
                                                        int stride       = 1,
                                                        int padding      = 0,
                                                        GroupedConvOp op = GroupedConvOp::Forward)
{
    GroupedConvProblem p;
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

inline GroupedConvProblem create_depthwise_grouped_conv2d_problem(
    int N, int C, int Hi, int Wi, int Y, int X, int stride = 1, int padding = 0)
{
    GroupedConvProblem p;
    p.N              = N;
    p.C              = C;
    p.K              = C;
    p.G              = C;
    p.input_spatial  = {1, Hi, Wi};
    p.filter_spatial = {1, Y, X};
    p.stride         = {1, stride, stride};
    p.padding        = {0, padding, padding};
    p.dilation       = {1, 1, 1};
    p.op             = GroupedConvOp::Forward;
    p.compute_output_size();
    return p;
}

inline void print_pattern_docs(std::ostream& os = std::cout)
{
    os << "Grouped Convolution Pattern Documentation\n";
    os << "==========================================\n";
    os << "Signature patterns: dtype, layout, conv_type (forward/bwd_data/bwd_weight), dims "
          "(2/3)\n";
    os << "Algorithm patterns: tile(M,N,K), wave(M,N,K), warp(M,N,K), pipeline, vector_sizes\n";
    os << "Arch patterns: gfx942, gfx90a, gfx950, or '*' for all\n";
}

inline void print_grouped_conv_kernel_decl(const GroupedConvKernelDecl& decl,
                                           std::ostream& os = std::cout)
{
    os << "GroupedConvKernelDecl: " << decl.name() << "\n";
    os << "  Signature: dtype=" << decl.signature.dtype_in_ << ", layout=" << decl.signature.layout_
       << ", conv_type=" << decl.signature.conv_op_ << ", dims=" << decl.signature.num_dims_
       << "\n";
    os << "  Algorithm: tile=" << decl.algorithm.tile_m_ << "x" << decl.algorithm.tile_n_ << "x"
       << decl.algorithm.tile_k_ << ", wave=" << decl.algorithm.wave_m_ << "x"
       << decl.algorithm.wave_n_ << "x" << decl.algorithm.wave_k_
       << ", warp=" << decl.algorithm.warp_m_ << "x" << decl.algorithm.warp_n_ << "x"
       << decl.algorithm.warp_k_ << ", pipeline=" << decl.algorithm.pipeline_ << "\n";
    os << "  Arch: " << decl.arch << "\n";
}

inline void print_grouped_conv_problem(const GroupedConvProblem& p, std::ostream& os = std::cout)
{
    os << p.to_string() << "\n";
    os << "  FLOPs: " << std::scientific << p.get_flops() << "\n";
}

inline GroupedConvKernelSet build_grouped_conv2d_fwd_set(const std::string& dtype = "fp16",
                                                         const std::string& arch  = "gfx942")
{
    GroupedConvKernelSet set;
    auto decl1 = create_grouped_conv2d_fwd(dtype, 128, 128, arch);
    set.add(decl1.signature, decl1.algorithm, decl1.arch);
    auto decl2 = create_grouped_conv2d_fwd(dtype, 256, 256, arch);
    set.add(decl2.signature, decl2.algorithm, decl2.arch);
    return set;
}

inline GroupedConvKernelSet build_grouped_conv2d_full_set(const std::string& dtype = "fp16",
                                                          const std::string& arch  = "gfx942")
{
    GroupedConvKernelSet set;
    set.merge(build_grouped_conv2d_fwd_set(dtype, arch));
    auto bwd_data = create_grouped_conv2d_bwd_data(dtype, 128, 128, arch);
    set.add(bwd_data.signature, bwd_data.algorithm, bwd_data.arch);
    auto bwd_weight = create_grouped_conv2d_bwd_weight(dtype, 128, 128, arch);
    set.add(bwd_weight.signature, bwd_weight.algorithm, bwd_weight.arch);
    return set;
}

struct ValidationResult
{
    bool passed        = false;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float rtol         = 1e-3f;
    float atol         = 1e-3f;

    void print(std::ostream& os = std::cout) const
    {
        os << "ValidationResult: " << (passed ? "PASSED" : "FAILED") << "\n";
        os << "  max_abs_diff: " << max_abs_diff << ", max_rel_diff: " << max_rel_diff << "\n";
        os << "  rtol: " << rtol << ", atol: " << atol << "\n";
    }
};

template <typename T>
inline ValidationResult validate_buffers(
    const T* result, const T* reference, size_t count, float rtol = 1e-3f, float atol = 1e-3f)
{
    ValidationResult vr;
    vr.rtol   = rtol;
    vr.atol   = atol;
    vr.passed = true;

    for(size_t i = 0; i < count; ++i)
    {
        float r        = static_cast<float>(result[i]);
        float ref      = static_cast<float>(reference[i]);
        float abs_diff = std::abs(r - ref);
        float rel_diff = (std::abs(ref) > 1e-10f) ? (abs_diff / std::abs(ref)) : 0.0f;

        vr.max_abs_diff = std::max(vr.max_abs_diff, abs_diff);
        vr.max_rel_diff = std::max(vr.max_rel_diff, rel_diff);

        float threshold = atol + rtol * std::abs(ref);
        if(abs_diff > threshold)
        {
            vr.passed = false;
        }
    }

    return vr;
}

struct BenchmarkResult
{
    std::string kernel_name;
    float time_ms      = 0.0f;
    float tflops       = 0.0f;
    int warmup_runs    = 0;
    int benchmark_runs = 0;

    void print(std::ostream& os = std::cout) const
    {
        os << "BenchmarkResult: " << kernel_name << "\n";
        os << "  time_ms: " << time_ms << ", tflops: " << tflops << "\n";
        os << "  warmup_runs: " << warmup_runs << ", benchmark_runs: " << benchmark_runs << "\n";
    }
};

inline float calc_tflops(double flops, float time_ms)
{
    return static_cast<float>(flops / (time_ms * 1e9));
}

inline double calculate_conv_tflops(const GroupedConvProblem& problem, double time_ms)
{
    return problem.get_flops() / (time_ms * 1e9);
}

} // namespace grouped_conv_utils

namespace examples {
inline int basic_grouped_conv_example_main(const std::string& example_name)
{
    std::cout << "=== " << example_name << " ===\n";

    // Create a grouped convolution problem
    auto problem = grouped_conv_utils::create_grouped_conv2d_problem(
        32, 64, 128, 28, 28, 3, 3, 1, 1, GroupedConvOp::Forward);

    grouped_conv_utils::print_grouped_conv_problem(problem);

    // Create and print a kernel declaration
    auto decl = grouped_conv_utils::create_grouped_conv2d_fwd("fp16", 128, 128, "gfx942");
    grouped_conv_utils::print_grouped_conv_kernel_decl(decl);

    // Build and print kernel set
    auto kernel_set = grouped_conv_utils::build_grouped_conv2d_fwd_set("fp16", "gfx942");
    kernel_set.print();

    return 0;
}
} // namespace examples

} // namespace dispatcher
} // namespace ck_tile
