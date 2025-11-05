// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>

namespace ck_tile {
namespace dispatcher {

/// Data types supported by CK Tile GEMM kernels
enum class DataType : std::uint8_t {
    FP16,
    BF16,
    FP32,
    FP8,
    BF8,
    INT8,
    INT32,
    UNKNOWN
};

/// Memory layout tags for tensors
enum class LayoutTag : std::uint8_t {
    RowMajor,
    ColMajor,
    PackedExternal
};

/// Pipeline variants for memory/compute optimization
enum class Pipeline : std::uint8_t {
    Mem,      // Memory-bound pipeline
    CompV1,   // Compute pipeline v1
    CompV2,   // Compute pipeline v2
    CompV3,   // Compute pipeline v3
    CompV4,   // Compute pipeline v4 (double buffering)
    CompV5    // Compute pipeline v5
};

/// Epilogue strategies for output processing
enum class Epilogue : std::uint8_t {
    None,
    Bias,
    Activation,
    CShuffle,   // Cross-shuffle epilogue
    Default
};

/// Scheduler types for wave coordination
enum class Scheduler : std::uint8_t {
    Auto,
    Intrawave,
    Interwave
};

/// KernelKey: Compile-time kernel configuration metadata
/// Organized into Signature (what operation) and Algorithm (how it's implemented)
struct KernelKey {
    /// Signature: Describes WHAT operation is computed (mathematical semantics)
    /// Two kernels with different signatures compute different mathematical operations
    struct Signature {
        DataType dtype_a;
        DataType dtype_b;
        DataType dtype_c;
        DataType dtype_acc;
        LayoutTag layout_a;
        LayoutTag layout_b;
        LayoutTag layout_c;
        bool transpose_a;
        bool transpose_b;
        bool grouped;
        std::uint8_t split_k;
        
        // Element-wise fusion: Describes mathematical operation applied to GEMM output
        // Examples: PassThrough (C = A*B), MultiDAdd (E = C + D0 + D1),
        //           MultiDMultiply (E = C * D0 * D1), Clamp, Relu, Gelu, etc.
        // This affects the mathematical result, so it belongs in Signature
        std::string elementwise_op;  // e.g., "PassThrough", "MultiDAdd", "Relu"
        std::uint8_t num_d_tensors;  // Number of additional input tensors for fusion (0 for basic GEMM)
        
        bool structured_sparsity;  // 2:4 sparsity affects mathematical correctness
    } signature;

    /// Algorithm: Describes HOW it's implemented (performance tuning parameters)
    /// Two kernels with same signature but different algorithms compute the same result
    /// with different performance characteristics
    struct Algorithm {
        // Hierarchical tiling configuration (primary tuning knobs)
        struct TileShape {
            std::uint16_t m;
            std::uint16_t n;
            std::uint16_t k;
        } tile_shape;

        struct WaveShape {
            std::uint8_t m;  // WarpPerBlock_M in generated kernels
            std::uint8_t n;  // WarpPerBlock_N
            std::uint8_t k;  // WarpPerBlock_K
        } wave_shape;

        struct WarpTileShape {
            std::uint8_t m;  // WarpTileM in generated kernels
            std::uint8_t n;  // WarpTileN
            std::uint8_t k;  // WarpTileK
        } warp_tile_shape;

        // Pipeline and scheduling strategy
        Pipeline pipeline;
        Scheduler scheduler;
        Epilogue epilogue;
        
        // Block and memory configuration
        std::uint16_t block_size;  // BlockSize in generated kernels (typically 256)
        bool double_buffer;        // DoubleSmemBuffer (true for compv4)
        bool persistent;           // UsePersistentKernel
        bool preshuffle;           // Preshuffle (for weight preshuffle variants)
        bool transpose_c;          // TransposeC
        std::uint8_t num_wave_groups;  // NumWaveGroups
    } algorithm;

    std::uint16_t gfx_arch;   // e.g. 942 for gfx942

    /// Generate a unique string identifier for this kernel configuration
    /// Format matches tile_engine naming convention for registry lookup
    [[nodiscard]] std::string encode_identifier() const
    {
        std::ostringstream oss;
        
        // Match tile_engine naming: tile_m x tile_n x tile_k _ warp_m x warp_n x warp_k _ warp_tile_m x warp_tile_n x warp_tile_k
        oss << algorithm.tile_shape.m << "x" << algorithm.tile_shape.n << "x" << algorithm.tile_shape.k << "_"
            << unsigned(algorithm.wave_shape.m) << "x" << unsigned(algorithm.wave_shape.n) << "x" << unsigned(algorithm.wave_shape.k) << "_"
            << unsigned(algorithm.warp_tile_shape.m) << "x" << unsigned(algorithm.warp_tile_shape.n) << "x" << unsigned(algorithm.warp_tile_shape.k);
        
        // Add trait flags
        oss << "_" << (algorithm.persistent ? "persist" : "nopers");
        
        if(signature.split_k > 1)
            oss << "_splitk" << unsigned(signature.split_k);
        if(!signature.elementwise_op.empty() && signature.elementwise_op != "PassThrough")
            oss << "_" << signature.elementwise_op;
        if(signature.num_d_tensors > 0)
            oss << "_d" << unsigned(signature.num_d_tensors);
        if(signature.structured_sparsity)
            oss << "_sparse";
        if(algorithm.preshuffle)
            oss << "_preshuffle";
        
        return oss.str();
    }

    /// Create a tuple of all fields for comparison operators
    constexpr auto tie() const
    {
        return std::tie(signature.dtype_a,
                        signature.dtype_b,
                        signature.dtype_c,
                        signature.dtype_acc,
                        signature.layout_a,
                        signature.layout_b,
                        signature.layout_c,
                        signature.transpose_a,
                        signature.transpose_b,
                        signature.grouped,
                        signature.split_k,
                        signature.elementwise_op,
                        signature.num_d_tensors,
                        signature.structured_sparsity,
                        algorithm.tile_shape.m,
                        algorithm.tile_shape.n,
                        algorithm.tile_shape.k,
                        algorithm.wave_shape.m,
                        algorithm.wave_shape.n,
                        algorithm.wave_shape.k,
                        algorithm.warp_tile_shape.m,
                        algorithm.warp_tile_shape.n,
                        algorithm.warp_tile_shape.k,
                        algorithm.pipeline,
                        algorithm.epilogue,
                        algorithm.scheduler,
                        algorithm.block_size,
                        gfx_arch,
                        signature.structured_sparsity,
                        algorithm.persistent,
                        algorithm.double_buffer,
                        algorithm.preshuffle,
                        algorithm.transpose_c,
                        algorithm.num_wave_groups);
    }

    /// Equality comparison
    friend bool operator==(const KernelKey& lhs, const KernelKey& rhs)
    {
        return lhs.tie() == rhs.tie();
    }

    /// Inequality comparison
    friend bool operator!=(const KernelKey& lhs, const KernelKey& rhs)
    {
        return !(lhs == rhs);
    }
};

} // namespace dispatcher
} // namespace ck_tile

