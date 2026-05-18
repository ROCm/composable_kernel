// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file grouped_conv_config.hpp
 * @brief CK Tile Grouped Convolution Configuration with Builder-style naming
 *
 * This adopts the Signature/Algorithm/Arch pattern from:
 *   experimental/builder/include/ck_tile/builder/reflect/conv_description.hpp
 *
 * Structure:
 *   - Signature: WHAT operation (types, layouts, direction, element ops)
 *   - Algorithm: HOW it's computed (tiles, warps, pipeline, scheduler, padding)
 *   - Arch: Target GPU architecture
 */

#pragma once

// Use common kernel_key types for DataType, Pipeline, etc.
#include "ck_tile/dispatcher/kernel_key.hpp"

#include <string>
#include <sstream>
#include <array>
#include <cstdint>

namespace ck_tile {
namespace dispatcher {

// DataType, Pipeline, Scheduler, Epilogue are defined in kernel_key.hpp
// No need to redefine them here

// =============================================================================
// Data Type Enum (matching CK Tile numeric types)
// =============================================================================

enum class ConvDataType
{
    // Standard floating point
    FP32, // float
    FP64, // double
    FP16, // half_t
    BF16, // bf16_t

    // 8-bit float variants (FP8/BF8)
    FP8,      // fp8_t (E4M3)
    BF8,      // bf8_t (E5M2)
    FP8_E4M3, // Explicit E4M3 format
    FP8_E5M2, // Explicit E5M2 format

    // Integer types
    INT8,  // int8_t
    UINT8, // uint8_t
    INT32, // int32_t (accumulator)

    // 4-bit types (gfx950+ only)
    FP4, // MXFP4
    INT4 // pk_int4_t
};

// =============================================================================
// Direction and Layout Enums
// =============================================================================

enum class GroupedConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHT
};

enum class ConvLayout2D
{
    GNHWC_GKYXC_GNHWK, // NHWC-style
    NHWGC_GKYXC_NHWGK,
    NGCHW_GKYXC_NGKHW, // NCHW-style
    NGCHW_GKCYX_NGKHW
};

enum class ConvLayout3D
{
    GNDHWC_GKZYXC_GNDHWK,
    NDHWGC_GKZYXC_NDHWGK,
    NGCDHW_GKZYXC_NGKDHW,
    NGCDHW_GKCZYX_NGKDHW
};

// =============================================================================
// Element-wise Operations
// =============================================================================

enum class ElementwiseOp
{
    PASS_THROUGH,
    BIAS,
    BIAS_CLAMP,
    SCALE,
    BILINEAR,
    RELU,
    GELU,
    SIGMOID,
    TANH
};

// =============================================================================
// Grouped Convolution Specialization
// =============================================================================

enum class ConvSpecialization
{
    DEFAULT,
    FILTER_1X1_PAD0,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_3X3,
    FILTER_5X5,
    FILTER_7X7
};

// =============================================================================
// Memory Operation Types (for accumulator operations)
// =============================================================================

enum class MemoryOperation
{
    SET,        // Direct write (=)
    ATOMIC_ADD, // Atomic addition (+=)
    ATOMIC_MAX, // Atomic max
    ADD         // Non-atomic addition
};

// =============================================================================
// Epilogue Types
// =============================================================================

enum class EpilogueType
{
    CSHUFFLE,        // C-shuffle epilogue
    DEFAULT_2D,      // Default 2D epilogue
    DEFAULT_GEMM_2D, // Default GEMM 2D epilogue
    DIRECT_STORE,    // Direct store without shuffle
    BIAS_ADD,        // Add bias
    BIAS_ADD_RELU,   // Add bias + ReLU
    BIAS_ADD_GELU    // Add bias + GELU
};

// =============================================================================
// Algorithm Enums (matching builder/types.hpp and CK Tile pipelines)
// =============================================================================

enum class PipelineVersion
{
    V1,            // Basic pipeline V1
    V2,            // Basic pipeline V2
    V3,            // Compute V3 (intrawave only)
    V4,            // Compute V4 (double buffer, ping-pong LDS)
    V5,            // Compute V5 (wave groups)
    V6,            // Compute V6 (newest)
    MEMORY,        // Memory pipeline
    COMPUTE_ASYNC, // Compute with async copy
    PRESHUFFLE_V2  // Preshuffle V2 pipeline
};

enum class PipelineScheduler
{
    DEFAULT,
    INTRAWAVE,
    INTERWAVE
};

enum class GemmPadding
{
    DEFAULT,
    NO_PADDING, // No padding
    M_PADDING,
    N_PADDING,
    K_PADDING,
    MN_PADDING,
    MK_PADDING,
    NK_PADDING,
    MNK_PADDING
};

// =============================================================================
// Signature Info (WHAT operation)
// =============================================================================

struct GroupedConvSignatureInfo
{
    int spatial_dim                = 2; // 1, 2, or 3
    GroupedConvDirection direction = GroupedConvDirection::FORWARD;
    std::string in_type            = "fp16";
    std::string wei_type           = "fp16";
    std::string out_type           = "fp16";
    std::string acc_type           = "fp32";
    std::string workspace_type     = "fp32"; // For two-stage algorithms
    std::string bias_type          = "fp16"; // For bias epilogue
    ElementwiseOp in_element_op    = ElementwiseOp::PASS_THROUGH;
    ElementwiseOp wei_element_op   = ElementwiseOp::PASS_THROUGH;
    ElementwiseOp out_element_op   = ElementwiseOp::PASS_THROUGH;
    ConvSpecialization conv_spec   = ConvSpecialization::DEFAULT;
    int num_groups                 = 1;

    // String helpers
    static const char* direction_str(GroupedConvDirection dir)
    {
        switch(dir)
        {
        case GroupedConvDirection::FORWARD: return "fwd";
        case GroupedConvDirection::BACKWARD_DATA: return "bwd_data";
        case GroupedConvDirection::BACKWARD_WEIGHT: return "bwd_weight";
        default: return "unknown";
        }
    }

    static const char* datatype_str(ConvDataType dt)
    {
        switch(dt)
        {
        case ConvDataType::FP32: return "fp32";
        case ConvDataType::FP64: return "fp64";
        case ConvDataType::FP16: return "fp16";
        case ConvDataType::BF16: return "bf16";
        case ConvDataType::FP8: return "fp8";
        case ConvDataType::BF8: return "bf8";
        case ConvDataType::FP8_E4M3: return "fp8_e4m3";
        case ConvDataType::FP8_E5M2: return "fp8_e5m2";
        case ConvDataType::INT8: return "int8";
        case ConvDataType::UINT8: return "uint8";
        case ConvDataType::INT32: return "int32";
        case ConvDataType::FP4: return "fp4";
        case ConvDataType::INT4: return "int4";
        default: return "unknown";
        }
    }
};

// =============================================================================
// Algorithm Info (HOW it's computed)
// =============================================================================

struct DataTileInfo
{
    int m = 128; // M tile (output spatial * N)
    int n = 128; // N tile (K output channels)
    int k = 64;  // K tile (C input channels)
};

struct WarpGemmParams
{
    int gemm_m = 16; // MFMA M dimension (MPerXDL)
    int gemm_n = 16; // MFMA N dimension (NPerXDL)
    int m_iter = 2;  // M iterations per warp (MXdlPerWave)
    int n_iter = 2;  // N iterations per warp (NXdlPerWave)
};

struct BlockWarpConfig
{
    int m_warp      = 2;  // Warps along M
    int n_warp      = 2;  // Warps along N
    int k_warp      = 1;  // Warps along K
    int m_warp_tile = 32; // Warp tile M
    int n_warp_tile = 32; // Warp tile N
    int k_warp_tile = 16; // Warp tile K
};

struct VectorSizeInfo
{
    int a = 4; // Input vector size
    int b = 8; // Weight vector size
    int c = 8; // Output vector size
};

struct GroupedConvAlgorithmInfo
{
    DataTileInfo tile;
    BlockWarpConfig warp;
    VectorSizeInfo vector_size;

    PipelineVersion pipeline    = PipelineVersion::V4;
    PipelineScheduler scheduler = PipelineScheduler::INTRAWAVE;
    GemmPadding padding         = GemmPadding::MNK_PADDING;
    MemoryOperation memory_op   = MemoryOperation::SET;
    EpilogueType epilogue       = EpilogueType::CSHUFFLE;

    int thread_block_size   = 256;
    bool double_smem_buffer = false;
    int num_wave_groups     = 1;
    int block_per_cu        = 1;
    int num_groups_to_merge = 1;

    // Pipeline string
    static const char* pipeline_str(PipelineVersion pv)
    {
        switch(pv)
        {
        case PipelineVersion::V1: return "v1";
        case PipelineVersion::V2: return "v2";
        case PipelineVersion::V3: return "compv3";
        case PipelineVersion::V4: return "compv4";
        case PipelineVersion::V5: return "compv5";
        case PipelineVersion::V6: return "compv6";
        case PipelineVersion::MEMORY: return "mem";
        case PipelineVersion::COMPUTE_ASYNC: return "comp_async";
        case PipelineVersion::PRESHUFFLE_V2: return "preshuffle_v2";
        default: return "unknown";
        }
    }

    static const char* scheduler_str(PipelineScheduler ps)
    {
        switch(ps)
        {
        case PipelineScheduler::DEFAULT: return "default";
        case PipelineScheduler::INTRAWAVE: return "intrawave";
        case PipelineScheduler::INTERWAVE: return "interwave";
        default: return "unknown";
        }
    }

    static const char* memory_op_str(MemoryOperation mo)
    {
        switch(mo)
        {
        case MemoryOperation::SET: return "set";
        case MemoryOperation::ATOMIC_ADD: return "atomic_add";
        case MemoryOperation::ATOMIC_MAX: return "atomic_max";
        case MemoryOperation::ADD: return "add";
        default: return "unknown";
        }
    }

    static const char* epilogue_str(EpilogueType et)
    {
        switch(et)
        {
        case EpilogueType::CSHUFFLE: return "cshuffle";
        case EpilogueType::DEFAULT_2D: return "default_2d";
        case EpilogueType::DEFAULT_GEMM_2D: return "default_gemm_2d";
        case EpilogueType::DIRECT_STORE: return "direct_store";
        case EpilogueType::BIAS_ADD: return "bias_add";
        case EpilogueType::BIAS_ADD_RELU: return "bias_add_relu";
        case EpilogueType::BIAS_ADD_GELU: return "bias_add_gelu";
        default: return "unknown";
        }
    }
};

// =============================================================================
// Arch Info (Target GPU)
// =============================================================================

struct ArchInfo
{
    std::string name     = "gfx942"; // MI300X default
    int max_waves_per_cu = 8;
    int lds_size_kb      = 64;
    int sgpr_count       = 108;
    int vgpr_count       = 512;

    bool supports_mfma_fp16() const { return name.find("gfx9") != std::string::npos; }
    bool supports_wmma() const { return name.find("gfx11") != std::string::npos; }
};

// =============================================================================
// Full Grouped Conv Config (combines Signature + Algorithm + Arch)
// =============================================================================

struct GroupedConvConfig
{
    GroupedConvSignatureInfo signature;
    GroupedConvAlgorithmInfo algorithm;
    ArchInfo arch;

    // Generate unique kernel name
    std::string name() const
    {
        std::ostringstream oss;
        oss << "grouped_conv_" << GroupedConvSignatureInfo::direction_str(signature.direction)
            << "_" << signature.in_type << "_" << signature.spatial_dim << "d" << "_"
            << GroupedConvAlgorithmInfo::pipeline_str(algorithm.pipeline) << "_" << algorithm.tile.m
            << "x" << algorithm.tile.n << "x" << algorithm.tile.k;
        return oss.str();
    }

    // Brief description
    std::string brief() const
    {
        std::ostringstream oss;
        oss << signature.spatial_dim << "D "
            << GroupedConvSignatureInfo::direction_str(signature.direction)
            << " Grouped Convolution (" << signature.in_type << ")";
        return oss.str();
    }

    // Detailed description (tree-like)
    std::string detailed() const
    {
        std::ostringstream oss;
        oss << signature.spatial_dim << "D "
            << GroupedConvSignatureInfo::direction_str(signature.direction)
            << " Grouped Convolution Kernel\n";

        oss << "  Signature:\n";
        oss << "    Data Type: " << signature.in_type << "\n";
        oss << "    Accumulator: " << signature.acc_type << "\n";
        oss << "    Groups: " << signature.num_groups << "\n";

        oss << "  Algorithm:\n";
        oss << "    Thread Block Size: " << algorithm.thread_block_size << "\n";
        oss << "    Data Tile: " << algorithm.tile.m << "x" << algorithm.tile.n << "x"
            << algorithm.tile.k << "\n";
        oss << "    Warp Config: " << algorithm.warp.m_warp << "x" << algorithm.warp.n_warp << "x"
            << algorithm.warp.k_warp << "\n";
        oss << "    Warp Tile: " << algorithm.warp.m_warp_tile << "x" << algorithm.warp.n_warp_tile
            << "x" << algorithm.warp.k_warp_tile << "\n";
        oss << "    Pipeline: " << GroupedConvAlgorithmInfo::pipeline_str(algorithm.pipeline)
            << "\n";
        oss << "    Scheduler: " << GroupedConvAlgorithmInfo::scheduler_str(algorithm.scheduler)
            << "\n";

        oss << "  Arch:\n";
        oss << "    Target: " << arch.name << "\n";

        return oss.str();
    }
};

// =============================================================================
// Predefined Configs
// =============================================================================

namespace configs {

// Memory-bound config
template <typename PrecType>
struct Memory : public GroupedConvConfig
{
    Memory()
    {
        algorithm.tile               = {128, 32, 128 / (int)sizeof(PrecType)};
        algorithm.warp               = {4, 1, 1, 32, 32, 16};
        algorithm.pipeline           = PipelineVersion::MEMORY;
        algorithm.double_smem_buffer = false;
    }
};

// Compute V3 - Small
template <typename PrecType>
struct CompV3_Small : public GroupedConvConfig
{
    CompV3_Small()
    {
        algorithm.tile     = {16, 64, 64};
        algorithm.warp     = {1, 4, 1, 16, 16, 32};
        algorithm.pipeline = PipelineVersion::V3;
    }
};

// Compute V3 - Medium
template <typename PrecType>
struct CompV3_Medium : public GroupedConvConfig
{
    CompV3_Medium()
    {
        algorithm.tile         = {128, 128, 128 / (int)sizeof(PrecType)};
        algorithm.warp         = {2, 2, 1, 16, 16, 32};
        algorithm.pipeline     = PipelineVersion::V3;
        algorithm.block_per_cu = 2;
    }
};

// Compute V3 - Large
template <typename PrecType>
struct CompV3_Large : public GroupedConvConfig
{
    CompV3_Large()
    {
        algorithm.tile     = {256, 256, 128 / (int)sizeof(PrecType)};
        algorithm.warp     = {2, 2, 1, 32, 32, 16};
        algorithm.pipeline = PipelineVersion::V3;
    }
};

// Compute V4 - Double buffered
template <typename PrecType>
struct CompV4 : public GroupedConvConfig
{
    CompV4()
    {
        algorithm.tile               = {256, 256, 64 / (int)sizeof(PrecType)};
        algorithm.warp               = {2, 2, 1, 32, 32, 16};
        algorithm.pipeline           = PipelineVersion::V4;
        algorithm.double_smem_buffer = true;
    }
};

// Compute V5 - Wave groups
template <typename PrecType>
struct CompV5 : public GroupedConvConfig
{
    CompV5()
    {
        algorithm.tile            = {128, 128, 64 / (int)sizeof(PrecType)};
        algorithm.warp            = {1, 1, 2, 32, 32, 16};
        algorithm.pipeline        = PipelineVersion::V5;
        algorithm.num_wave_groups = 2;
    }
};

// WMMA config for gfx11xx
template <typename PrecType>
struct WMMA : public GroupedConvConfig
{
    WMMA()
    {
        algorithm.tile         = {128, 128, 64 / (int)sizeof(PrecType)};
        algorithm.warp         = {4, 2, 1, 16, 16, 16};
        algorithm.pipeline     = PipelineVersion::V3;
        algorithm.block_per_cu = 2;
        arch.name              = "gfx1100";
    }
};

// Merged groups config
template <typename PrecType>
struct CompV3_MergedGroups : public GroupedConvConfig
{
    CompV3_MergedGroups()
    {
        algorithm.tile                = {16, 32, 32};
        algorithm.warp                = {1, 2, 1, 16, 16, 32};
        algorithm.vector_size         = {4, 8, 8};
        algorithm.pipeline            = PipelineVersion::V3;
        algorithm.num_groups_to_merge = 2;
    }
};

} // namespace configs

// =============================================================================
// DataType Traits (compile-time type info for CK Tile types)
// =============================================================================

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
    static constexpr int size_bytes   = 4;
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
    static constexpr int size_bytes   = 8;
};

// Forward declare CK Tile types for traits
// Note: actual ck_tile types are defined in ck_tile/core/numeric/
// These traits allow working with type names at compile time

// =============================================================================
// ConvTypeConfig (input/weight/acc/output type combinations)
// =============================================================================

template <typename InDataType,
          typename WeiDataType = InDataType,
          typename OutDataType = InDataType,
          typename AccDataType = float>
struct ConvTypeConfig
{
    using input_type       = InDataType;
    using weight_type      = WeiDataType;
    using output_type      = OutDataType;
    using accumulator_type = AccDataType;
};

// Common type configurations as type aliases
// FP16 -> FP32 accumulator -> FP16 output (most common)
// BF16 -> FP32 accumulator -> BF16 output
// FP8 -> FP32 accumulator -> FP8 output
// INT8 -> INT32 accumulator -> INT8 output

} // namespace dispatcher
} // namespace ck_tile
