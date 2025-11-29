// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file conv_config.hpp
 * @brief CK Tile Convolution Configuration with Builder-style naming
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

enum class ConvDirection
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

enum class ElementwiseOp
{
    PASS_THROUGH,
    BIAS,
    BIAS_CLAMP,
    SCALE,
    BILINEAR
};

enum class ConvSpecialization
{
    DEFAULT,
    FILTER_1X1_PAD0,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_3X3
};

// =============================================================================
// Algorithm Enums (matching builder/types.hpp)
// =============================================================================

enum class PipelineVersion
{
    V1,    // Basic pipeline
    V2,    // Improved pipeline
    V3,    // Compute V3 (intrawave only)
    V4,    // Compute V4 (double buffer)
    V5,    // Compute V5 (wave groups)
    MEMORY // Memory pipeline
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

struct ConvSignatureInfo
{
    int spatial_dim              = 2; // 1, 2, or 3
    ConvDirection direction      = ConvDirection::FORWARD;
    std::string in_type          = "fp16";
    std::string wei_type         = "fp16";
    std::string out_type         = "fp16";
    std::string acc_type         = "fp32";
    ElementwiseOp in_element_op  = ElementwiseOp::PASS_THROUGH;
    ElementwiseOp wei_element_op = ElementwiseOp::PASS_THROUGH;
    ElementwiseOp out_element_op = ElementwiseOp::PASS_THROUGH;
    ConvSpecialization conv_spec = ConvSpecialization::DEFAULT;
    int num_groups               = 1;

    // String helpers
    static const char* direction_str(ConvDirection dir)
    {
        switch(dir)
        {
        case ConvDirection::FORWARD: return "fwd";
        case ConvDirection::BACKWARD_DATA: return "bwdd";
        case ConvDirection::BACKWARD_WEIGHT: return "bwdw";
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

struct ConvAlgorithmInfo
{
    DataTileInfo tile;
    BlockWarpConfig warp;
    VectorSizeInfo vector_size;

    PipelineVersion pipeline    = PipelineVersion::V4;
    PipelineScheduler scheduler = PipelineScheduler::INTRAWAVE;
    GemmPadding padding         = GemmPadding::MNK_PADDING;

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
        case PipelineVersion::MEMORY: return "mem";
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
// Full Conv Config (combines Signature + Algorithm + Arch)
// =============================================================================

struct ConvConfig
{
    ConvSignatureInfo signature;
    ConvAlgorithmInfo algorithm;
    ArchInfo arch;

    // Generate unique kernel name
    std::string name() const
    {
        std::ostringstream oss;
        oss << "conv_" << ConvSignatureInfo::direction_str(signature.direction) << "_"
            << signature.in_type << "_" << signature.spatial_dim << "d" << "_"
            << ConvAlgorithmInfo::pipeline_str(algorithm.pipeline) << "_" << algorithm.tile.m << "x"
            << algorithm.tile.n << "x" << algorithm.tile.k;
        return oss.str();
    }

    // Brief description
    std::string brief() const
    {
        std::ostringstream oss;
        oss << signature.spatial_dim << "D "
            << ConvSignatureInfo::direction_str(signature.direction) << " convolution ("
            << signature.in_type << ")";
        return oss.str();
    }

    // Detailed description (tree-like)
    std::string detailed() const
    {
        std::ostringstream oss;
        oss << signature.spatial_dim << "D "
            << ConvSignatureInfo::direction_str(signature.direction) << " Convolution Kernel\n";

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
        oss << "    Pipeline: " << ConvAlgorithmInfo::pipeline_str(algorithm.pipeline) << "\n";
        oss << "    Scheduler: " << ConvAlgorithmInfo::scheduler_str(algorithm.scheduler) << "\n";

        oss << "  Arch:\n";
        oss << "    Target: " << arch.name << "\n";

        return oss.str();
    }
};

// =============================================================================
// Predefined Configs (like conv_configs.hpp)
// =============================================================================

namespace configs {

// Memory-bound config
template <typename PrecType>
struct Memory : public ConvConfig
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
struct CompV3_Small : public ConvConfig
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
struct CompV3_Medium : public ConvConfig
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
struct CompV3_Large : public ConvConfig
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
struct CompV4 : public ConvConfig
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
struct CompV5 : public ConvConfig
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
struct WMMA : public ConvConfig
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

} // namespace configs

} // namespace dispatcher
} // namespace ck_tile
