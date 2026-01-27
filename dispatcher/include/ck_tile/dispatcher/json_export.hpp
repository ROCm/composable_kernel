// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * JSON Export Utilities for Dispatcher Registry
 *
 * Provides functionality to export kernel registry metadata to JSON format,
 * similar to the tile engine benchmarking JSON export.
 *
 * Features:
 * - Export all registered kernels with full metadata
 * - Include kernel configuration (tile shapes, pipeline, scheduler, etc.)
 * - Group kernels by various properties (data type, layout, pipeline, etc.)
 * - Export to string or file
 *
 * Usage:
 *   auto& registry = Registry::instance();
 *   std::string json = export_registry_json(registry);
 *   // or
 *   export_registry_json_to_file(registry, "kernels.json");
 */

#pragma once

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <iomanip>
#include <ctime>
#include <chrono>

namespace ck_tile {
namespace dispatcher {

/// Convert DataType enum to string
inline std::string datatype_to_string(DataType dtype)
{
    switch(dtype)
    {
    case DataType::FP16: return "fp16";
    case DataType::BF16: return "bf16";
    case DataType::FP32: return "fp32";
    case DataType::FP8: return "fp8";
    case DataType::BF8: return "bf8";
    case DataType::INT8: return "int8";
    case DataType::INT32: return "int32";
    default: return "unknown";
    }
}

/// Convert LayoutTag enum to string
inline std::string layout_to_string(LayoutTag layout)
{
    switch(layout)
    {
    case LayoutTag::RowMajor: return "row_major";
    case LayoutTag::ColMajor: return "col_major";
    case LayoutTag::PackedExternal: return "packed_external";
    default: return "unknown";
    }
}

/// Convert Pipeline enum to string
inline std::string pipeline_to_string(Pipeline pipeline)
{
    switch(pipeline)
    {
    case Pipeline::Mem: return "mem";
    case Pipeline::CompV1: return "compv1";
    case Pipeline::CompV2: return "compv2";
    case Pipeline::CompV3: return "compv3";
    case Pipeline::CompV4: return "compv4";
    case Pipeline::CompV5: return "compv5";
    default: return "unknown";
    }
}

/// Convert Epilogue enum to string
inline std::string epilogue_to_string(Epilogue epilogue)
{
    switch(epilogue)
    {
    case Epilogue::None: return "none";
    case Epilogue::Bias: return "bias";
    case Epilogue::Activation: return "activation";
    case Epilogue::CShuffle: return "cshuffle";
    case Epilogue::Default: return "default";
    default: return "unknown";
    }
}

/// Convert Scheduler enum to string
inline std::string scheduler_to_string(Scheduler scheduler)
{
    switch(scheduler)
    {
    case Scheduler::Auto: return "auto";
    case Scheduler::Intrawave: return "intrawave";
    case Scheduler::Interwave: return "interwave";
    default: return "unknown";
    }
}

/// Escape string for JSON
inline std::string json_escape(const std::string& str)
{
    std::ostringstream oss;
    for(char c : str)
    {
        switch(c)
        {
        case '"': oss << "\\\""; break;
        case '\\': oss << "\\\\"; break;
        case '\b': oss << "\\b"; break;
        case '\f': oss << "\\f"; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:
            if(c < 0x20)
            {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
            }
            else
            {
                oss << c;
            }
        }
    }
    return oss.str();
}

/// Get current timestamp in ISO 8601 format
inline std::string get_iso_timestamp()
{
    auto now    = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&time_t, &tm_buf);

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

/// Export a single kernel's metadata to JSON
inline std::string export_kernel_json(const KernelInstance& kernel)
{
    std::ostringstream json;
    const auto& key = kernel.get_key();

    json << "    {\n";
    json << "      \"name\": \"" << json_escape(kernel.get_name()) << "\",\n";
    json << "      \"identifier\": \"" << json_escape(key.encode_identifier()) << "\",\n";

    // Signature (what operation is computed)
    json << "      \"signature\": {\n";
    json << "        \"dtype_a\": \"" << datatype_to_string(key.signature.dtype_a) << "\",\n";
    json << "        \"dtype_b\": \"" << datatype_to_string(key.signature.dtype_b) << "\",\n";
    json << "        \"dtype_c\": \"" << datatype_to_string(key.signature.dtype_c) << "\",\n";
    json << "        \"dtype_acc\": \"" << datatype_to_string(key.signature.dtype_acc) << "\",\n";
    json << "        \"layout_a\": \"" << layout_to_string(key.signature.layout_a) << "\",\n";
    json << "        \"layout_b\": \"" << layout_to_string(key.signature.layout_b) << "\",\n";
    json << "        \"layout_c\": \"" << layout_to_string(key.signature.layout_c) << "\",\n";
    json << "        \"transpose_a\": " << (key.signature.transpose_a ? "true" : "false") << ",\n";
    json << "        \"transpose_b\": " << (key.signature.transpose_b ? "true" : "false") << ",\n";
    json << "        \"grouped\": " << (key.signature.grouped ? "true" : "false") << ",\n";
    json << "        \"split_k\": " << (int)key.signature.split_k << ",\n";
    json << "        \"elementwise_op\": \"" << json_escape(key.signature.elementwise_op)
         << "\",\n";
    json << "        \"num_d_tensors\": " << (int)key.signature.num_d_tensors << ",\n";
    json << "        \"structured_sparsity\": "
         << (key.signature.structured_sparsity ? "true" : "false") << "\n";
    json << "      },\n";

    // Algorithm (how it's implemented)
    json << "      \"algorithm\": {\n";
    json << "        \"tile_shape\": {\n";
    json << "          \"m\": " << key.algorithm.tile_shape.m << ",\n";
    json << "          \"n\": " << key.algorithm.tile_shape.n << ",\n";
    json << "          \"k\": " << key.algorithm.tile_shape.k << "\n";
    json << "        },\n";
    json << "        \"wave_shape\": {\n";
    json << "          \"m\": " << (int)key.algorithm.wave_shape.m << ",\n";
    json << "          \"n\": " << (int)key.algorithm.wave_shape.n << ",\n";
    json << "          \"k\": " << (int)key.algorithm.wave_shape.k << "\n";
    json << "        },\n";
    json << "        \"warp_tile_shape\": {\n";
    json << "          \"m\": " << (int)key.algorithm.warp_tile_shape.m << ",\n";
    json << "          \"n\": " << (int)key.algorithm.warp_tile_shape.n << ",\n";
    json << "          \"k\": " << (int)key.algorithm.warp_tile_shape.k << "\n";
    json << "        },\n";
    json << "        \"pipeline\": \"" << pipeline_to_string(key.algorithm.pipeline) << "\",\n";
    json << "        \"scheduler\": \"" << scheduler_to_string(key.algorithm.scheduler) << "\",\n";
    json << "        \"epilogue\": \"" << epilogue_to_string(key.algorithm.epilogue) << "\",\n";
    json << "        \"block_size\": " << key.algorithm.block_size << ",\n";
    json << "        \"double_buffer\": " << (key.algorithm.double_buffer ? "true" : "false")
         << ",\n";
    json << "        \"persistent\": " << (key.algorithm.persistent ? "true" : "false") << ",\n";
    json << "        \"preshuffle\": " << (key.algorithm.preshuffle ? "true" : "false") << ",\n";
    json << "        \"transpose_c\": " << (key.algorithm.transpose_c ? "true" : "false") << ",\n";
    json << "        \"num_wave_groups\": " << (int)key.algorithm.num_wave_groups << "\n";
    json << "      },\n";

    json << "      \"gfx_arch\": \"" << json_escape(key.gfx_arch) << "\"\n";
    json << "    }";

    return json.str();
}

/// Export registry metadata and statistics to JSON
inline std::string export_registry_json(const Registry& registry, bool include_statistics = true)
{
    std::ostringstream json;

    auto all_kernels = registry.get_all();

    json << "{\n";

    // Metadata
    json << "  \"metadata\": {\n";
    json << "    \"timestamp\": \"" << get_iso_timestamp() << "\",\n";
    json << "    \"registry_name\": \"" << json_escape(registry.get_name()) << "\",\n";
    json << "    \"total_kernels\": " << all_kernels.size() << ",\n";
    json << "    \"export_version\": \"1.0.0\"\n";
    json << "  },\n";

    // Statistics (if enabled)
    if(include_statistics && !all_kernels.empty())
    {
        std::map<std::string, int> by_datatype;
        std::map<std::string, int> by_pipeline;
        std::map<std::string, int> by_scheduler;
        std::map<std::string, int> by_layout;
        std::map<std::string, int> by_gfx_arch;

        for(const auto& kernel : all_kernels)
        {
            const auto& key = kernel->get_key();

            // Count by data type
            std::string dtype_key = datatype_to_string(key.signature.dtype_a) + "_" +
                                    datatype_to_string(key.signature.dtype_b) + "_" +
                                    datatype_to_string(key.signature.dtype_c);
            by_datatype[dtype_key]++;

            // Count by pipeline
            by_pipeline[pipeline_to_string(key.algorithm.pipeline)]++;

            // Count by scheduler
            by_scheduler[scheduler_to_string(key.algorithm.scheduler)]++;

            // Count by layout
            std::string layout_key = layout_to_string(key.signature.layout_a) + "_" +
                                     layout_to_string(key.signature.layout_b) + "_" +
                                     layout_to_string(key.signature.layout_c);
            by_layout[layout_key]++;

            // Count by GFX architecture
            by_gfx_arch[key.gfx_arch]++;
        }

        json << "  \"statistics\": {\n";

        // Data type breakdown
        json << "    \"by_datatype\": {\n";
        bool first = true;
        for(const auto& [dtype, count] : by_datatype)
        {
            if(!first)
                json << ",\n";
            json << "      \"" << dtype << "\": " << count;
            first = false;
        }
        json << "\n    },\n";

        // Pipeline breakdown
        json << "    \"by_pipeline\": {\n";
        first = true;
        for(const auto& [pipeline, count] : by_pipeline)
        {
            if(!first)
                json << ",\n";
            json << "      \"" << pipeline << "\": " << count;
            first = false;
        }
        json << "\n    },\n";

        // Scheduler breakdown
        json << "    \"by_scheduler\": {\n";
        first = true;
        for(const auto& [scheduler, count] : by_scheduler)
        {
            if(!first)
                json << ",\n";
            json << "      \"" << scheduler << "\": " << count;
            first = false;
        }
        json << "\n    },\n";

        // Layout breakdown
        json << "    \"by_layout\": {\n";
        first = true;
        for(const auto& [layout, count] : by_layout)
        {
            if(!first)
                json << ",\n";
            json << "      \"" << layout << "\": " << count;
            first = false;
        }
        json << "\n    },\n";

        // GFX architecture breakdown
        json << "    \"by_gfx_arch\": {\n";
        first = true;
        for(const auto& [arch, count] : by_gfx_arch)
        {
            if(!first)
                json << ",\n";
            json << "      \"" << arch << "\": " << count;
            first = false;
        }
        json << "\n    }\n";

        json << "  },\n";
    }

    // Kernels list
    json << "  \"kernels\": [\n";
    for(size_t i = 0; i < all_kernels.size(); ++i)
    {
        json << export_kernel_json(*all_kernels[i]);
        if(i < all_kernels.size() - 1)
        {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";

    json << "}\n";

    return json.str();
}

/// Export registry to a JSON file
inline bool export_registry_json_to_file(const Registry& registry,
                                         const std::string& filename,
                                         bool include_statistics = true)
{
    std::string json = export_registry_json(registry, include_statistics);

    std::ofstream file(filename);
    if(!file.is_open())
    {
        return false;
    }

    file << json;
    file.close();

    return true;
}

} // namespace dispatcher
} // namespace ck_tile
