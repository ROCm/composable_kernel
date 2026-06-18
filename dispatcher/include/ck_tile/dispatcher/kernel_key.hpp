// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>

namespace ck_tile {
namespace dispatcher {

/// Data types supported by CK Tile GEMM kernels
/// Matches tile_engine DATA_TYPE_MAP for full compatibility
enum class DataType : std::uint8_t
{
    FP16,  // ck_tile::half_t
    BF16,  // ck_tile::bf16_t
    FP32,  // float
    FP64,  // double
    FP8,   // ck_tile::fp8_t (E4M3)
    BF8,   // ck_tile::bf8_t (E5M2)
    INT8,  // ck_tile::int8_t
    INT4,  // ck_tile::pk_int4_t (packed int4)
    INT32, // ck_tile::int32_t
    UNKNOWN
};

/// Memory layout tags for tensors
enum class LayoutTag : std::uint8_t
{
    RowMajor,
    ColMajor,
    PackedExternal
};

/// Pipeline variants for memory/compute optimization
/// Matches tile_engine PIPELINE_MAP for full compatibility
enum class Pipeline : std::uint8_t
{
    Mem,          // Memory-bound pipeline
    CompV1,       // Compute pipeline v1
    CompV2,       // Compute pipeline v2
    CompV3,       // Compute pipeline v3
    CompV4,       // Compute pipeline v4 (double buffering)
    CompV5,       // Compute pipeline v5
    CompV6,       // Compute pipeline v6
    PreShuffleV1, // Weight preshuffle pipeline v1
    PreShuffleV2, // Weight preshuffle pipeline v2 (optimized)
    Wavelet       // Wavelet pipeline (specialized math + load waves)
};

/// Epilogue strategies for output processing
/// Matches tile_engine epilogue options for full compatibility
enum class Epilogue : std::uint8_t
{
    None,
    Default,       // DefaultGemm2DEpilogue
    CShuffle,      // CShuffleEpilogue (cross-shuffle)
    Bias,          // Bias addition
    Activation,    // Fused activation
    BiasActivation // Fused bias + activation
};

/// Scheduler types for wave coordination
enum class Scheduler : std::uint8_t
{
    Auto,
    Intrawave,
    Interwave
};

/// KernelKey: Compile-time kernel configuration metadata
/// Organized into Signature (what operation) and Algorithm (how it's implemented)
struct KernelKey
{
    /// Signature: Describes WHAT operation is computed (mathematical semantics)
    /// Two kernels with different signatures compute different mathematical operations
    struct Signature
    {
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
        std::string elementwise_op; // e.g., "PassThrough", "MultiDAdd", "Relu"
        std::uint8_t
            num_d_tensors; // Number of additional input tensors for fusion (0 for basic GEMM)

        bool structured_sparsity; // 2:4 sparsity affects mathematical correctness
    } signature;

    /// Algorithm: Describes HOW it's implemented (performance tuning parameters)
    /// Two kernels with same signature but different algorithms compute the same result
    /// with different performance characteristics
    struct Algorithm
    {
        // Hierarchical tiling configuration (primary tuning knobs)
        struct TileShape
        {
            std::uint16_t m;
            std::uint16_t n;
            std::uint16_t k;
        } tile_shape;

        struct WaveShape
        {
            std::uint8_t m; // WarpPerBlock_M in generated kernels
            std::uint8_t n; // WarpPerBlock_N
            std::uint8_t k; // WarpPerBlock_K
        } wave_shape;

        struct WarpTileShape
        {
            std::uint8_t m; // WarpTileM in generated kernels
            std::uint8_t n; // WarpTileN
            std::uint8_t k; // WarpTileK
        } warp_tile_shape;

        // Pipeline and scheduling strategy
        Pipeline pipeline;
        Scheduler scheduler;
        Epilogue epilogue;

        // Block and memory configuration
        std::uint16_t block_size;     // BlockSize in generated kernels (typically 256)
        bool double_buffer;           // DoubleSmemBuffer (true for compv4)
        bool persistent;              // UsePersistentKernel
        bool preshuffle;              // Preshuffle (for weight preshuffle variants)
        bool transpose_c;             // TransposeC
        std::uint8_t num_wave_groups; // NumWaveGroups

        // Padding support flags (kPadM, kPadN, kPadK in generated kernels)
        bool pad_m = true; // Support arbitrary M dimensions via padding
        bool pad_n = true; // Support arbitrary N dimensions via padding
        bool pad_k = true; // Support arbitrary K dimensions via padding
    } algorithm;

    std::string gfx_arch; // e.g. "gfx942", "gfx90a", "gfx908"

    /// Generate a unique string identifier for this kernel configuration
    /// Format matches tile_engine naming convention for registry lookup
    /// Note: Defined after to_string() functions to use them
    [[nodiscard]] std::string encode_identifier() const;

    /// Create a tuple of all fields for comparison operators
    auto tie() const
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
                        algorithm.num_wave_groups,
                        algorithm.pad_m,
                        algorithm.pad_n,
                        algorithm.pad_k);
    }

    /// Equality comparison
    friend bool operator==(const KernelKey& lhs, const KernelKey& rhs)
    {
        return lhs.tie() == rhs.tie();
    }

    /// Inequality comparison
    friend bool operator!=(const KernelKey& lhs, const KernelKey& rhs) { return !(lhs == rhs); }
};

// =============================================================================
// String Conversion Helpers (for serialization and debugging)
// =============================================================================

/// Convert DataType to string
inline std::string to_string(DataType dtype)
{
    switch(dtype)
    {
    case DataType::FP16: return "fp16";
    case DataType::BF16: return "bf16";
    case DataType::FP32: return "fp32";
    case DataType::FP64: return "fp64";
    case DataType::FP8: return "fp8";
    case DataType::BF8: return "bf8";
    case DataType::INT8: return "int8";
    case DataType::INT4: return "int4";
    case DataType::INT32: return "int32";
    default: return "unknown";
    }
}

/// Convert string to DataType
inline DataType string_to_dtype(const std::string& str)
{
    if(str == "fp16")
        return DataType::FP16;
    if(str == "bf16")
        return DataType::BF16;
    if(str == "fp32")
        return DataType::FP32;
    if(str == "fp64")
        return DataType::FP64;
    if(str == "fp8")
        return DataType::FP8;
    if(str == "bf8")
        return DataType::BF8;
    if(str == "int8")
        return DataType::INT8;
    if(str == "int4")
        return DataType::INT4;
    if(str == "int32")
        return DataType::INT32;
    return DataType::UNKNOWN;
}

/// Convert LayoutTag to string
inline std::string to_string(LayoutTag layout)
{
    switch(layout)
    {
    case LayoutTag::RowMajor: return "r";
    case LayoutTag::ColMajor: return "c";
    case LayoutTag::PackedExternal: return "p";
    default: return "?";
    }
}

/// Convert string to LayoutTag
inline LayoutTag string_to_layout(const std::string& str)
{
    if(str == "r" || str == "row" || str == "RowMajor")
        return LayoutTag::RowMajor;
    if(str == "c" || str == "col" || str == "ColMajor")
        return LayoutTag::ColMajor;
    if(str == "p" || str == "packed")
        return LayoutTag::PackedExternal;
    return LayoutTag::RowMajor; // Default
}

/// Convert Pipeline to string
inline std::string to_string(Pipeline pipeline)
{
    switch(pipeline)
    {
    case Pipeline::Mem: return "mem";
    case Pipeline::CompV1: return "compv1";
    case Pipeline::CompV2: return "compv2";
    case Pipeline::CompV3: return "compv3";
    case Pipeline::CompV4: return "compv4";
    case Pipeline::CompV5: return "compv5";
    case Pipeline::CompV6: return "compv6";
    case Pipeline::PreShuffleV1: return "preshufflev1";
    case Pipeline::PreShuffleV2: return "preshufflev2";
    case Pipeline::Wavelet: return "wavelet";
    default: return "unknown";
    }
}

/// Convert string to Pipeline
inline Pipeline string_to_pipeline(const std::string& str)
{
    if(str == "mem")
        return Pipeline::Mem;
    if(str == "compv1")
        return Pipeline::CompV1;
    if(str == "compv2")
        return Pipeline::CompV2;
    if(str == "compv3")
        return Pipeline::CompV3;
    if(str == "compv4")
        return Pipeline::CompV4;
    if(str == "compv5")
        return Pipeline::CompV5;
    if(str == "compv6")
        return Pipeline::CompV6;
    if(str == "preshufflev1")
        return Pipeline::PreShuffleV1;
    if(str == "preshufflev2")
        return Pipeline::PreShuffleV2;
    if(str == "wavelet")
        return Pipeline::Wavelet;
    return Pipeline::Mem; // Default
}

/// Convert Epilogue to string
inline std::string to_string(Epilogue epilogue)
{
    switch(epilogue)
    {
    case Epilogue::None: return "none";
    case Epilogue::Default: return "default";
    case Epilogue::CShuffle: return "cshuffle";
    case Epilogue::Bias: return "bias";
    case Epilogue::Activation: return "activation";
    case Epilogue::BiasActivation: return "bias_activation";
    default: return "unknown";
    }
}

/// Convert string to Epilogue
inline Epilogue string_to_epilogue(const std::string& str)
{
    if(str == "none")
        return Epilogue::None;
    if(str == "default")
        return Epilogue::Default;
    if(str == "cshuffle")
        return Epilogue::CShuffle;
    if(str == "bias")
        return Epilogue::Bias;
    if(str == "activation")
        return Epilogue::Activation;
    if(str == "bias_activation")
        return Epilogue::BiasActivation;
    return Epilogue::Default; // Default
}

/// Convert Scheduler to string
inline std::string to_string(Scheduler scheduler)
{
    switch(scheduler)
    {
    case Scheduler::Auto: return "auto";
    case Scheduler::Intrawave: return "intrawave";
    case Scheduler::Interwave: return "interwave";
    default: return "unknown";
    }
}

/// Convert string to Scheduler
inline Scheduler string_to_scheduler(const std::string& str)
{
    if(str == "auto")
        return Scheduler::Auto;
    if(str == "intrawave")
        return Scheduler::Intrawave;
    if(str == "interwave")
        return Scheduler::Interwave;
    return Scheduler::Intrawave; // Default
}

/// Common elementwise operations (for reference in elementwise_op field)
/// These match CK Tile's ck_tile::element_wise namespace
namespace ElementwiseOps {
constexpr const char* PassThrough    = "PassThrough";
constexpr const char* Add            = "Add";
constexpr const char* Multiply       = "Multiply";
constexpr const char* MultiDAdd      = "MultiDAdd";
constexpr const char* MultiDMultiply = "MultiDMultiply";
constexpr const char* Relu           = "Relu";
constexpr const char* Gelu           = "Gelu";
constexpr const char* Clamp          = "Clamp";
constexpr const char* Sigmoid        = "Sigmoid";
constexpr const char* Tanh           = "Tanh";
constexpr const char* Swish          = "Swish";
constexpr const char* HardSwish      = "HardSwish";
} // namespace ElementwiseOps

// =============================================================================
// KernelKey::encode_identifier() implementation
// Defined after to_string() functions to use them
// =============================================================================

inline std::string KernelKey::encode_identifier() const
{
    std::ostringstream oss;

    // Include data types and layout for uniqueness across different signatures
    oss << to_string(signature.dtype_a) << "_";
    oss << to_string(signature.layout_a) << to_string(signature.layout_b)
        << to_string(signature.layout_c) << "_";

    // Include pipeline, scheduler, epilogue for uniqueness
    oss << to_string(algorithm.pipeline) << "_";
    oss << to_string(algorithm.epilogue) << "_";
    oss << to_string(algorithm.scheduler) << "_";

    // Match tile_engine naming: padding flags (True/False) then persistent flag
    oss << (algorithm.pad_m ? "True" : "False") << "_";
    oss << (algorithm.pad_n ? "True" : "False") << "_";
    oss << (algorithm.pad_k ? "True" : "False") << "_";
    oss << (algorithm.persistent ? "True" : "False") << "_";

    // Match tile_engine naming: tile_m x tile_n x tile_k _ warp_m x warp_n x warp_k _
    // warp_tile_m x warp_tile_n x warp_tile_k
    oss << algorithm.tile_shape.m << "x" << algorithm.tile_shape.n << "x" << algorithm.tile_shape.k
        << "_" << unsigned(algorithm.wave_shape.m) << "x" << unsigned(algorithm.wave_shape.n) << "x"
        << unsigned(algorithm.wave_shape.k) << "_" << unsigned(algorithm.warp_tile_shape.m) << "x"
        << unsigned(algorithm.warp_tile_shape.n) << "x" << unsigned(algorithm.warp_tile_shape.k);

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

} // namespace dispatcher
} // namespace ck_tile
