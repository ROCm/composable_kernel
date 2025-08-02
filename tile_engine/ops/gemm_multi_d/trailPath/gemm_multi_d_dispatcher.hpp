
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <unordered_map>
#include <functional>
#include <vector>

#include "gemm_multi_d_common.hpp"
#include "gemm_multi_d_instances.hpp"

/// @brief Defines the configuration parameters for a GEMM Multi D operation, enabling the selection of a
/// specific kernel instance based on the provided settings.
struct KernelTraits
{
    /// @brief The name of the pipeline.
    std::string pipeline;
    /// @brief The name of the scheduler (e.g., "intrawave", "interwave").
    std::string scheduler;
    /// @brief The name of the epilogue (e.g., "cshuffle", "default").
    std::string epilogue;
    /// @brief Indicates whether padding is applied to the M dimension.
    bool pad_m;
    /// @brief Indicates whether padding is applied to the N dimension.
    bool pad_n;
    /// @brief Indicates whether padding is applied to the K dimension.
    bool pad_k;
};

struct GemmDispatcher {
    static auto& get_kernel_map() {
        // Use a static local variable
        static std::unordered_map<
            std::string,
            std::vector<std::function<std::tuple<std::string, float>(ck_tile::GemmHostArgs<DsDataType::size()>&, const ck_tile::stream_config&)>>>
            kernel_map;
        return kernel_map;
    }

    static void init() {
        auto& kernel_map = get_kernel_map();
        if(!kernel_map.empty()) return;
        
         kernel_map["compv3_cshuffle_intrawave_false_false_false"] = {[=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                } 
            };
          kernel_map["mem_cshuffle_intrawave_false_false_false"] = {[=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                } 
            };
          kernel_map["compv4_cshuffle_intrawave_false_false_false"] = {[=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<compv4_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                } 
            };
          kernel_map["mem_cshuffle_interwave_false_false_false"] = {[=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 8>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 16, 16, 32>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 1, 4, 1, 32, 32, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 256, 32, 4, 1, 1, 4, 64, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 4, 1, 1, 16, 16, 16>>(args, stream);
                                }, [=](ck_tile::GemmHostArgs<DsDataType::size()>& args, const ck_tile::stream_config& stream) { 
                        return run_kernel<mem_cshuffle_interwave_false_false_false::GemmKernel<256, 128, 32, 2, 2, 1, 4, 64, 16>>(args, stream);
                                } 
            };
     }

    template <typename Kernel>
    static std::tuple<std::string, float> run_kernel(ck_tile::GemmHostArgs<>& args, const ck_tile::stream_config& stream)
    {
        std::string name = Kernel::get_name();
        float avg_time = Kernel::launch(args, stream);
        
        return std::make_tuple(name, avg_time);
    }
    
    
    static auto dispatch(const KernelTraits& trait) {
        init();
        const std::string key = assemble_key(trait);
        auto& kernel_map = get_kernel_map();
        if(auto it = kernel_map.find(key); it != kernel_map.end())
        {
            return it->second;
        }
        throw std::runtime_error("No suitable kernel found: " + key);
    }

private:
    static std::string assemble_key(const KernelTraits &trait) {
        return std::string(trait.pipeline) + "_" +
               trait.epilogue + "_" +
               trait.scheduler + "_" +
               (trait.pad_m ? "true" : "false") + "_" +
               (trait.pad_n ? "true" : "false") + "_" +
               (trait.pad_k ? "true" : "false");
    }
};

