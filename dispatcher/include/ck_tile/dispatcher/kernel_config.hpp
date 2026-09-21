// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file kernel_config.hpp
 * @brief Explicit kernel configuration for CK Tile Dispatcher
 *
 * This header provides a KernelConfig struct that mirrors the Python API,
 * allowing explicit, self-contained kernel configuration without relying
 * on force-included generated headers.
 *
 * Usage:
 *   #include "ck_tile/dispatcher/kernel_config.hpp"
 *   using namespace ck_tile::dispatcher;
 *
 *   // Step 1: Define explicit config
 *   auto config = KernelConfig::fp16_rcr()
 *       .tile(128, 128, 32)
 *       .wave(2, 2, 1)
 *       .warp_tile(32, 32, 16)
 *       .pipeline(Pipeline::CompV4)
 *       .scheduler(Scheduler::Intrawave);
 *
 *   // Step 2: Create registry and register
 *   Registry registry;
 *   registry.register_kernel(config.build_key(), config.get_name());
 *
 *   // Step 3: Create dispatcher
 *   Dispatcher dispatcher(&registry);
 *
 *   // Step 4: Run GEMM
 *   dispatcher.run(a, b, c, Problem(M, N, K));
 */

#pragma once

#include "ck_tile/dispatcher/kernel_key.hpp"
#include <sstream>
#include <string>
#include <iostream>

namespace ck_tile {
namespace dispatcher {

/**
 * @brief Explicit kernel configuration matching Python's KernelConfig
 *
 * This provides a fluent builder API for creating kernel configurations
 * with all parameters visible and explicit.
 */
class KernelConfig
{
    public:
    // =========================================================================
    // Data types
    // =========================================================================
    DataType dtype_a   = DataType::FP16;
    DataType dtype_b   = DataType::FP16;
    DataType dtype_c   = DataType::FP16;
    DataType dtype_acc = DataType::FP32;

    // =========================================================================
    // Layouts
    // =========================================================================
    LayoutTag layout_a = LayoutTag::RowMajor;
    LayoutTag layout_b = LayoutTag::ColMajor;
    LayoutTag layout_c = LayoutTag::RowMajor;

    // =========================================================================
    // Tile shape
    // =========================================================================
    int tile_m = 128;
    int tile_n = 128;
    int tile_k = 32;

    // =========================================================================
    // Wave shape (warps per block)
    // =========================================================================
    int wave_m = 2;
    int wave_n = 2;
    int wave_k = 1;

    // =========================================================================
    // Warp tile shape
    // =========================================================================
    int warp_m = 32;
    int warp_n = 32;
    int warp_k = 16;

    // =========================================================================
    // Block and pipeline
    // =========================================================================
    int block_size           = 256;
    Pipeline pipeline_type   = Pipeline::CompV4;
    Scheduler scheduler_type = Scheduler::Intrawave;
    Epilogue epilogue_type   = Epilogue::CShuffle;

    // =========================================================================
    // Padding and features
    // =========================================================================
    bool pad_m      = true;
    bool pad_n      = true;
    bool pad_k      = true;
    bool preshuffle = false;

    // =========================================================================
    // Target architecture
    // =========================================================================
    std::string gfx_arch = "gfx942";

    // =========================================================================
    // Fluent builder methods
    // =========================================================================

    /// Set tile dimensions (M x N x K)
    KernelConfig& tile(int m, int n, int k)
    {
        tile_m = m;
        tile_n = n;
        tile_k = k;
        return *this;
    }

    /// Set wave dimensions (warps per block M x N x K)
    KernelConfig& wave(int m, int n, int k)
    {
        wave_m = m;
        wave_n = n;
        wave_k = k;
        return *this;
    }

    /// Set warp tile dimensions (M x N x K)
    KernelConfig& warp_tile(int m, int n, int k)
    {
        warp_m = m;
        warp_n = n;
        warp_k = k;
        return *this;
    }

    /// Set block size
    KernelConfig& block(int size)
    {
        block_size = size;
        return *this;
    }

    /// Set pipeline type
    KernelConfig& pipeline(Pipeline p)
    {
        pipeline_type = p;
        return *this;
    }

    /// Set scheduler type
    KernelConfig& scheduler(Scheduler s)
    {
        scheduler_type = s;
        return *this;
    }

    /// Set epilogue type
    KernelConfig& epilogue(Epilogue e)
    {
        epilogue_type = e;
        return *this;
    }

    /// Set data types for A, B, C
    KernelConfig& dtypes(DataType a, DataType b, DataType c, DataType acc = DataType::FP32)
    {
        dtype_a   = a;
        dtype_b   = b;
        dtype_c   = c;
        dtype_acc = acc;
        return *this;
    }

    /// Set layouts for A, B, C
    KernelConfig& layouts(LayoutTag a, LayoutTag b, LayoutTag c)
    {
        layout_a = a;
        layout_b = b;
        layout_c = c;
        return *this;
    }

    /// Set padding flags
    KernelConfig& padding(bool m, bool n, bool k)
    {
        pad_m = m;
        pad_n = n;
        pad_k = k;
        return *this;
    }

    /// Set target GPU architecture
    KernelConfig& arch(const std::string& gpu)
    {
        gfx_arch = gpu;
        return *this;
    }

    // =========================================================================
    // Preset configurations
    // =========================================================================

    /// FP16 Row-Column-Row layout (most common)
    static KernelConfig fp16_rcr() { return KernelConfig{}; }

    /// FP16 Row-Row-Row layout
    static KernelConfig fp16_rrr()
    {
        KernelConfig cfg;
        cfg.layout_b = LayoutTag::RowMajor;
        return cfg;
    }

    /// BF16 Row-Column-Row layout
    static KernelConfig bf16_rcr()
    {
        KernelConfig cfg;
        cfg.dtype_a = DataType::BF16;
        cfg.dtype_b = DataType::BF16;
        cfg.dtype_c = DataType::BF16;
        return cfg;
    }

    /// FP32 Row-Column-Row layout
    static KernelConfig fp32_rcr()
    {
        KernelConfig cfg;
        cfg.dtype_a   = DataType::FP32;
        cfg.dtype_b   = DataType::FP32;
        cfg.dtype_c   = DataType::FP32;
        cfg.dtype_acc = DataType::FP32;
        return cfg;
    }

    // =========================================================================
    // Build KernelKey
    // =========================================================================

    /// Build a KernelKey from this configuration
    [[nodiscard]] KernelKey build_key() const
    {
        KernelKey key;

        // Signature
        key.signature.dtype_a             = dtype_a;
        key.signature.dtype_b             = dtype_b;
        key.signature.dtype_c             = dtype_c;
        key.signature.dtype_acc           = dtype_acc;
        key.signature.layout_a            = layout_a;
        key.signature.layout_b            = layout_b;
        key.signature.layout_c            = layout_c;
        key.signature.transpose_a         = false;
        key.signature.transpose_b         = false;
        key.signature.grouped             = false;
        key.signature.split_k             = 1;
        key.signature.elementwise_op      = "PassThrough";
        key.signature.num_d_tensors       = 0;
        key.signature.structured_sparsity = false;

        // Algorithm
        key.algorithm.tile_shape      = {static_cast<std::uint16_t>(tile_m),
                                         static_cast<std::uint16_t>(tile_n),
                                         static_cast<std::uint16_t>(tile_k)};
        key.algorithm.wave_shape      = {static_cast<std::uint8_t>(wave_m),
                                         static_cast<std::uint8_t>(wave_n),
                                         static_cast<std::uint8_t>(wave_k)};
        key.algorithm.warp_tile_shape = {static_cast<std::uint8_t>(warp_m),
                                         static_cast<std::uint8_t>(warp_n),
                                         static_cast<std::uint8_t>(warp_k)};
        key.algorithm.pipeline        = pipeline_type;
        key.algorithm.scheduler       = scheduler_type;
        key.algorithm.epilogue        = epilogue_type;
        key.algorithm.block_size      = block_size;
        key.algorithm.double_buffer   = true;
        key.algorithm.persistent      = false;
        key.algorithm.preshuffle      = preshuffle;
        key.algorithm.transpose_c     = false;
        key.algorithm.num_wave_groups = 1;

        key.gfx_arch = gfx_arch;

        return key;
    }

    // =========================================================================
    // String representations
    // =========================================================================

    /// Get tile string (e.g., "128x128x32")
    [[nodiscard]] std::string tile_str() const
    {
        std::ostringstream oss;
        oss << tile_m << "x" << tile_n << "x" << tile_k;
        return oss.str();
    }

    /// Get wave string (e.g., "2x2x1")
    [[nodiscard]] std::string wave_str() const
    {
        std::ostringstream oss;
        oss << wave_m << "x" << wave_n << "x" << wave_k;
        return oss.str();
    }

    /// Get warp tile string (e.g., "32x32x16")
    [[nodiscard]] std::string warp_tile_str() const
    {
        std::ostringstream oss;
        oss << warp_m << "x" << warp_n << "x" << warp_k;
        return oss.str();
    }

    /// Get layout string (e.g., "rcr")
    [[nodiscard]] std::string layout_str() const
    {
        std::ostringstream oss;
        oss << to_string(layout_a) << to_string(layout_b) << to_string(layout_c);
        return oss.str();
    }

    /// Get kernel name for generated code lookup
    [[nodiscard]] std::string get_name() const
    {
        std::ostringstream oss;
        oss << "gemm_" << to_string(dtype_a) << "_" << layout_str() << "_"
            << to_string(pipeline_type) << "_" << to_string(epilogue_type) << "_"
            << to_string(scheduler_type) << "_" << (pad_m ? "True" : "False") << "_"
            << (pad_n ? "True" : "False") << "_" << (pad_k ? "True" : "False") << "_"
            << "False" // preshuffle
            << "_" << tile_str() << "_" << wave_str() << "_" << warp_tile_str();
        return oss.str();
    }

    /// Print configuration to stdout
    void print_config(std::ostream& os = std::cout) const
    {
        os << "  Data types:\n";
        os << "    dtype_a   = " << to_string(dtype_a) << "\n";
        os << "    dtype_b   = " << to_string(dtype_b) << "\n";
        os << "    dtype_c   = " << to_string(dtype_c) << "\n";
        os << "    dtype_acc = " << to_string(dtype_acc) << "\n";
        os << "  Layouts:\n";
        os << "    layout_a = " << to_string(layout_a) << "\n";
        os << "    layout_b = " << to_string(layout_b) << "\n";
        os << "    layout_c = " << to_string(layout_c) << "\n";
        os << "  Tile shape:\n";
        os << "    tile = " << tile_str() << "\n";
        os << "    wave = " << wave_str() << "\n";
        os << "    warp_tile = " << warp_tile_str() << "\n";
        os << "  Pipeline:\n";
        os << "    pipeline  = " << to_string(pipeline_type) << "\n";
        os << "    scheduler = " << to_string(scheduler_type) << "\n";
        os << "    epilogue  = " << to_string(epilogue_type) << "\n";
        os << "  Padding:\n";
        os << "    pad_m = " << (pad_m ? "true" : "false") << "\n";
        os << "    pad_n = " << (pad_n ? "true" : "false") << "\n";
        os << "    pad_k = " << (pad_k ? "true" : "false") << "\n";
        os << "  Target:\n";
        os << "    gfx_arch = " << gfx_arch << "\n";
    }
};

} // namespace dispatcher
} // namespace ck_tile
