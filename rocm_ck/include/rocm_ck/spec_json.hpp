// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host -- JSON serialization for spec types. No CK deps.
//
// Runtime to_json() functions for GemmSpec.
// Used by build-time extractors to emit .spec.json files that pack.py
// reads to embed structured metadata in the kpack TOC.
//
// Hand-written JSON -- no library dependency. The schema is fixed and small.

#pragma once

#include <rocm_ck/arch_properties.hpp>
#include <rocm_ck/gemm_spec.hpp>

#include <string>

namespace rocm_ck {

// ============================================================================
// Enum-to-string (GemmSpec enums not already covered by existing functions)
// ============================================================================

inline const char* pipelineName(Pipeline p)
{
    switch(p)
    {
    case Pipeline::V1: return "V1";
    case Pipeline::V3: return "V3";
    case Pipeline::V4: return "V4";
    case Pipeline::Memory: return "Memory";
    case Pipeline::Preshuffle: return "Preshuffle";
    }
    return "???";
}

inline const char* schedulerName(PipelineScheduler s)
{
    switch(s)
    {
    case PipelineScheduler::Intrawave: return "Intrawave";
    case PipelineScheduler::Interwave: return "Interwave";
    }
    return "???";
}

inline const char* partitionerName(TilePartitioner tp)
{
    switch(tp)
    {
    case TilePartitioner::Direct: return "Direct";
    case TilePartitioner::Linear: return "Linear";
    case TilePartitioner::SpatiallyLocal: return "SpatiallyLocal";
    case TilePartitioner::StreamK: return "StreamK";
    }
    return "???";
}

inline const char* storeStrategyName(StoreStrategy ss)
{
    switch(ss)
    {
    case StoreStrategy::CShuffle: return "CShuffle";
    case StoreStrategy::Direct2D: return "Direct2D";
    }
    return "???";
}

inline const char* epilogueOpName(EpilogueOp op)
{
    switch(op)
    {
    case EpilogueOp::Add: return "Add";
    case EpilogueOp::Mul: return "Mul";
    case EpilogueOp::Relu: return "Relu";
    case EpilogueOp::FastGelu: return "FastGelu";
    case EpilogueOp::Gelu: return "Gelu";
    case EpilogueOp::Silu: return "Silu";
    case EpilogueOp::Sigmoid: return "Sigmoid";
    }
    return "???";
}

inline const char* gpuTargetName(GpuTarget t)
{
    switch(t)
    {
    case GpuTarget::gfx90a: return "gfx90a";
    case GpuTarget::gfx942: return "gfx942";
    case GpuTarget::gfx950: return "gfx950";
    case GpuTarget::gfx1100: return "gfx1100";
    case GpuTarget::gfx1101: return "gfx1101";
    case GpuTarget::gfx1102: return "gfx1102";
    case GpuTarget::gfx1150: return "gfx1150";
    case GpuTarget::gfx1151: return "gfx1151";
    case GpuTarget::_count: break;
    }
    return "???";
}

// ============================================================================
// JSON helpers
// ============================================================================

namespace detail {

inline std::string tensor_name_str(const FixedString<16>& tn)
{
    return std::string(tn.data, static_cast<std::size_t>(tn.len));
}

inline std::string quoted(const std::string& s) { return "\"" + s + "\""; }

inline std::string quoted(const char* s) { return quoted(std::string(s)); }

inline std::string dim3_json(const Dim3& d)
{
    return "{\"m\": " + std::to_string(d.m) + ", \"n\": " + std::to_string(d.n) +
           ", \"k\": " + std::to_string(d.k) + "}";
}

inline std::string physical_tensor_json(const PhysicalTensor& pt)
{
    return "{\"name\": " + quoted(tensor_name_str(pt.name)) +
           ", \"dtype\": " + quoted(dataTypeName(pt.dtype)) +
           ", \"layout\": " + quoted(layoutName(pt.layout)) +
           ", \"args_slot\": " + std::to_string(pt.args_slot) + "}";
}

inline std::string targets_json(TargetSet targets)
{
    std::string result = "[";
    bool first         = true;
    targets.for_each([&](GpuTarget t) {
        if(!first)
            result += ", ";
        result += quoted(gpuTargetName(t));
        first = false;
    });
    result += "]";
    return result;
}

} // namespace detail

// ============================================================================
// to_json: full variant output (name + spec_type + targets + spec)
// ============================================================================

/// Serialize a GEMM variant to JSON for the build-time extractor.
inline std::string to_json(const char* name, const GemmSpec& spec, TargetSet targets)
{
    std::string j;
    j += "{\n";
    j += "  \"name\": " + detail::quoted(name) + ",\n";
    j += "  \"spec_type\": \"GemmSpec\",\n";
    j += "  \"targets\": " + detail::targets_json(targets) + ",\n";
    j += "  \"spec\": {\n";

    // Physical tensors
    j += "    \"physical_tensors\": [\n";
    for(int i = 0; i < spec.num_physical_tensors; ++i)
    {
        j += "      " + detail::physical_tensor_json(spec.physical_tensors[i]);
        if(i + 1 < spec.num_physical_tensors)
            j += ",";
        j += "\n";
    }
    j += "    ],\n";

    j += "    \"acc_dtype\": " + detail::quoted(dataTypeName(spec.acc_dtype)) + ",\n";
    j += "    \"block_tile\": " + detail::dim3_json(spec.block_tile) + ",\n";
    j += "    \"block_waves\": " + detail::dim3_json(spec.block_waves) + ",\n";
    j += "    \"wave_tile\": " + detail::dim3_json(spec.wave_tile) + ",\n";
    j += "    \"workgroup_size\": " + std::to_string(spec.workgroup_size) + ",\n";
    j += "    \"k_batch\": " + std::to_string(spec.k_batch) + ",\n";
    j += "    \"pipeline\": " + detail::quoted(pipelineName(spec.pipeline)) + ",\n";
    j += "    \"pipeline_scheduler\": " + detail::quoted(schedulerName(spec.pipeline_scheduler)) +
         ",\n";
    j += "    \"tile_partitioner\": " + detail::quoted(partitionerName(spec.tile_partitioner)) +
         ",\n";

    // Epilogue ops
    j += "    \"epilogue_ops\": [";
    for(int i = 0; i < spec.num_epilogue_ops; ++i)
    {
        if(i > 0)
            j += ", ";
        j += detail::quoted(epilogueOpName(spec.epilogue_ops[i]));
    }
    j += "],\n";

    j +=
        "    \"store_strategy\": " + detail::quoted(storeStrategyName(spec.store_strategy)) + ",\n";
    j += "    \"pad_m\": " + std::string(spec.pad_m ? "true" : "false") + ",\n";
    j += "    \"pad_n\": " + std::string(spec.pad_n ? "true" : "false") + ",\n";
    j += "    \"group_size\": " + std::to_string(spec.group_size) + "\n";

    j += "  }\n";
    j += "}\n";
    return j;
}

} // namespace rocm_ck
