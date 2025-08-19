# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# Script to generate gemm_dispatcher.cpp from available kernels

# Find all kernel header files
file(GLOB_RECURSE KERNEL_HEADERS 
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/rcr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/rrr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/rrc/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/rcc/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/crr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/crc/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/ccr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp16/ccc/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/rcr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/rrr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/rrc/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/rcc/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/crr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/crc/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/ccr/*.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/fp32/ccc/*.hpp"
)

# Generate dispatcher header
set(DISPATCHER_HPP "${CMAKE_CURRENT_BINARY_DIR}/gemm_dispatcher.hpp")
file(WRITE ${DISPATCHER_HPP} "// Auto-generated file - do not edit\n")
file(APPEND ${DISPATCHER_HPP} "#pragma once\n\n")
file(APPEND ${DISPATCHER_HPP} "#include <functional>\n")
file(APPEND ${DISPATCHER_HPP} "#include <vector>\n")
file(APPEND ${DISPATCHER_HPP} "#include <string>\n")
file(APPEND ${DISPATCHER_HPP} "#include \"gemm_common.hpp\"\n")
file(APPEND ${DISPATCHER_HPP} "#include \"ck_tile/host.hpp\"\n\n")

file(APPEND ${DISPATCHER_HPP} "struct KernelInfo {\n")
file(APPEND ${DISPATCHER_HPP} "    std::string name;\n")
file(APPEND ${DISPATCHER_HPP} "    KernelTraits traits;\n")
file(APPEND ${DISPATCHER_HPP} "    std::function<float(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&)> launch_func;\n")
file(APPEND ${DISPATCHER_HPP} "};\n\n")

file(APPEND ${DISPATCHER_HPP} "class GemmDispatcher {\n")
file(APPEND ${DISPATCHER_HPP} "public:\n")
file(APPEND ${DISPATCHER_HPP} "    static std::vector<KernelInfo> get_all_kernels();\n")
file(APPEND ${DISPATCHER_HPP} "    static std::function<float(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>\n")
file(APPEND ${DISPATCHER_HPP} "        dispatch(bool structured_sparsity, const KernelTraits& traits);\n")
file(APPEND ${DISPATCHER_HPP} "};\n")

# Generate dispatcher implementation
set(DISPATCHER_CPP "${CMAKE_CURRENT_BINARY_DIR}/gemm_dispatcher.cpp")
file(WRITE ${DISPATCHER_CPP} "// Auto-generated file - do not edit\n")
file(APPEND ${DISPATCHER_CPP} "#include \"gemm_dispatcher.hpp\"\n\n")

# Include all kernel headers
foreach(KERNEL_HEADER ${KERNEL_HEADERS})
    get_filename_component(KERNEL_NAME ${KERNEL_HEADER} NAME_WE)
    file(APPEND ${DISPATCHER_CPP} "#include \"${KERNEL_HEADER}\"\n")
endforeach()

file(APPEND ${DISPATCHER_CPP} "\n")

# Generate get_all_kernels function
file(APPEND ${DISPATCHER_CPP} "std::vector<KernelInfo> GemmDispatcher::get_all_kernels() {\n")
file(APPEND ${DISPATCHER_CPP} "    std::vector<KernelInfo> kernels;\n\n")

foreach(KERNEL_HEADER ${KERNEL_HEADERS})
    get_filename_component(KERNEL_NAME ${KERNEL_HEADER} NAME_WE)
    file(APPEND ${DISPATCHER_CPP} "    {\n")
    file(APPEND ${DISPATCHER_CPP} "        KernelInfo info;\n")
    file(APPEND ${DISPATCHER_CPP} "        info.name = \"${KERNEL_NAME}\";\n")
    file(APPEND ${DISPATCHER_CPP} "        info.traits = extract_traits_from_name(\"${KERNEL_NAME}\");\n")
    file(APPEND ${DISPATCHER_CPP} "        info.launch_func = [](const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream) {\n")
    file(APPEND ${DISPATCHER_CPP} "            return SelectedKernel::launch(args, stream);\n")
    file(APPEND ${DISPATCHER_CPP} "        };\n")
    file(APPEND ${DISPATCHER_CPP} "        kernels.push_back(info);\n")
    file(APPEND ${DISPATCHER_CPP} "    }\n\n")
endforeach()

file(APPEND ${DISPATCHER_CPP} "    return kernels;\n")
file(APPEND ${DISPATCHER_CPP} "}\n\n")

# Generate dispatch function
file(APPEND ${DISPATCHER_CPP} "std::function<float(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>\n")
file(APPEND ${DISPATCHER_CPP} "GemmDispatcher::dispatch(bool structured_sparsity, const KernelTraits& traits) {\n")
file(APPEND ${DISPATCHER_CPP} "    auto all_kernels = get_all_kernels();\n")
file(APPEND ${DISPATCHER_CPP} "    \n")
file(APPEND ${DISPATCHER_CPP} "    // Find matching kernel\n")
file(APPEND ${DISPATCHER_CPP} "    for(const auto& kernel : all_kernels) {\n")
file(APPEND ${DISPATCHER_CPP} "        if(kernel.traits.pipeline == traits.pipeline &&\n")
file(APPEND ${DISPATCHER_CPP} "           kernel.traits.scheduler == traits.scheduler &&\n")
file(APPEND ${DISPATCHER_CPP} "           kernel.traits.epilogue == traits.epilogue &&\n")
file(APPEND ${DISPATCHER_CPP} "           kernel.traits.pad_m == traits.pad_m &&\n")
file(APPEND ${DISPATCHER_CPP} "           kernel.traits.pad_n == traits.pad_n &&\n")
file(APPEND ${DISPATCHER_CPP} "           kernel.traits.pad_k == traits.pad_k &&\n")
file(APPEND ${DISPATCHER_CPP} "           kernel.traits.persistent == traits.persistent) {\n")
file(APPEND ${DISPATCHER_CPP} "            return kernel.launch_func;\n")
file(APPEND ${DISPATCHER_CPP} "        }\n")
file(APPEND ${DISPATCHER_CPP} "    }\n")
file(APPEND ${DISPATCHER_CPP} "    \n")
file(APPEND ${DISPATCHER_CPP} "    // Return first kernel if no match found\n")
file(APPEND ${DISPATCHER_CPP} "    if(!all_kernels.empty()) {\n")
file(APPEND ${DISPATCHER_CPP} "        return all_kernels[0].launch_func;\n")
file(APPEND ${DISPATCHER_CPP} "    }\n")
file(APPEND ${DISPATCHER_CPP} "    \n")
file(APPEND ${DISPATCHER_CPP} "    return nullptr;\n")
file(APPEND ${DISPATCHER_CPP} "}\n")
