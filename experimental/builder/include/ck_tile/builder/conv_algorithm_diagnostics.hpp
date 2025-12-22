// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::diagnostics {

#define CHECK_MARK(cond) (cond ? "[✓]" : "[✗]")

// Macro to check a concept and generate both the boolean and the string representation
#define CHECK_CONCEPT(Type, Concept) \
    static constexpr bool c_##Concept = Concept<Type>; \
    static constexpr const char* s_##Concept = #Concept;

// Helper to create diagnostic message line
#define DIAGNOSTIC_LINE(Concept) \
    "  " + std::string(s_##Concept) + ": " + std::string(CHECK_MARK(c_##Concept)) + "\n" + \
    (c_##Concept ? std::string("") : detailed_diagnostic_##Concept<T>())

namespace detail {

// ThreadBlockDescriptor diagnostics
template <typename T>
consteval auto diagnose_thread_block_descriptor() -> std::string {
    if constexpr (!requires { T::thread_block; }) {
        return "      → T::thread_block member: [✗] (not found)\n";
    } else {
        using TB = decltype(T::thread_block);
        std::string msg;
        
        constexpr bool has_block_size = requires(TB t) { { t.block_size } -> std::convertible_to<size_t>; };
        constexpr bool has_tile_m = requires(TB t) { { t.tile_size.m } -> std::convertible_to<size_t>; };
        constexpr bool has_tile_n = requires(TB t) { { t.tile_size.n } -> std::convertible_to<size_t>; };
        constexpr bool has_tile_k = requires(TB t) { { t.tile_size.k } -> std::convertible_to<size_t>; };
        
        msg += "      → thread_block.block_size: " + std::string(CHECK_MARK(has_block_size)) + 
               (has_block_size ? "\n" : " (missing or wrong type)\n");
        msg += "      → thread_block.tile_size.m: " + std::string(CHECK_MARK(has_tile_m)) + 
               (has_tile_m ? "\n" : " (missing or wrong type)\n");
        msg += "      → thread_block.tile_size.n: " + std::string(CHECK_MARK(has_tile_n)) + 
               (has_tile_n ? "\n" : " (missing or wrong type)\n");
        msg += "      → thread_block.tile_size.k: " + std::string(CHECK_MARK(has_tile_k)) + 
               (has_tile_k ? "\n" : " (missing or wrong type)\n");
        
        return msg;
    }
}

// GridwiseXdlGemmDescriptor diagnostics
template <typename T, typename XdlParams>
consteval auto diagnose_xdl_params() -> std::string {
    std::string msg;
    
    constexpr bool has_m_per_xdl = requires(XdlParams t) { { t.m_per_xdl } -> std::convertible_to<size_t>; };
    constexpr bool has_n_per_xdl = requires(XdlParams t) { { t.n_per_xdl } -> std::convertible_to<size_t>; };
    constexpr bool has_m_xdl_per_wave = requires(XdlParams t) { { t.m_xdl_per_wave } -> std::convertible_to<size_t>; };
    constexpr bool has_n_xdl_per_wave = requires(XdlParams t) { { t.n_xdl_per_wave } -> std::convertible_to<size_t>; };
    
    msg += "      → xdl_params.m_per_xdl: " + std::string(CHECK_MARK(has_m_per_xdl)) + 
           (has_m_per_xdl ? "\n" : " (missing or wrong type)\n");
    msg += "      → xdl_params.n_per_xdl: " + std::string(CHECK_MARK(has_n_per_xdl)) + 
           (has_n_per_xdl ? "\n" : " (missing or wrong type)\n");
    msg += "      → xdl_params.m_xdl_per_wave: " + std::string(CHECK_MARK(has_m_xdl_per_wave)) + 
           (has_m_xdl_per_wave ? "\n" : " (missing or wrong type)\n");
    msg += "      → xdl_params.n_xdl_per_wave: " + std::string(CHECK_MARK(has_n_xdl_per_wave)) + 
           (has_n_xdl_per_wave ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

// BlockTransferDescriptor diagnostics
template <typename T, typename BT>
consteval auto diagnose_block_transfer(const char* prefix) -> std::string {
    std::string msg;
    
    constexpr bool has_k0 = requires(BT t) { { t.k0 } -> std::convertible_to<size_t>; };
    constexpr bool has_m_n = requires(BT t) { { t.m_n } -> std::convertible_to<size_t>; };
    constexpr bool has_k1 = requires(BT t) { { t.k1 } -> std::convertible_to<size_t>; };
    
    msg += std::string("      → ") + prefix + ".k0: " + std::string(CHECK_MARK(has_k0)) + 
           (has_k0 ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".m_n: " + std::string(CHECK_MARK(has_m_n)) + 
           (has_m_n ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".k1: " + std::string(CHECK_MARK(has_k1)) + 
           (has_k1 ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

// LdsTransferDescriptor diagnostics
template <typename T, typename LT>
consteval auto diagnose_lds_transfer(const char* prefix) -> std::string {
    std::string msg;
    
    constexpr bool has_src_vector_dim = requires(LT t) { { t.src_vector_dim } -> std::convertible_to<size_t>; };
    constexpr bool has_src_scalar_per_vector = requires(LT t) { { t.src_scalar_per_vector } -> std::convertible_to<size_t>; };
    constexpr bool has_lds_dst_scalar_per_vector = requires(LT t) { { t.lds_dst_scalar_per_vector } -> std::convertible_to<size_t>; };
    constexpr bool has_is_direct_load = requires(LT t) { { t.is_direct_load } -> std::convertible_to<bool>; };
    constexpr bool has_lds_padding = requires(LT t) { { t.lds_padding } -> std::convertible_to<bool>; };
    
    msg += std::string("      → ") + prefix + ".src_vector_dim: " + std::string(CHECK_MARK(has_src_vector_dim)) + 
           (has_src_vector_dim ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".src_scalar_per_vector: " + std::string(CHECK_MARK(has_src_scalar_per_vector)) + 
           (has_src_scalar_per_vector ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".lds_dst_scalar_per_vector: " + std::string(CHECK_MARK(has_lds_dst_scalar_per_vector)) + 
           (has_lds_dst_scalar_per_vector ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".is_direct_load: " + std::string(CHECK_MARK(has_is_direct_load)) + 
           (has_is_direct_load ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".lds_padding: " + std::string(CHECK_MARK(has_lds_padding)) + 
           (has_lds_padding ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

// ThreadClusterDescriptor diagnostics
template <typename T, typename TC>
consteval auto diagnose_thread_cluster(const char* prefix) -> std::string {
    std::string msg;
    
    constexpr bool has_m_block = requires(TC t) { { t.m_block } -> std::convertible_to<size_t>; };
    constexpr bool has_m_wave_per_xdl = requires(TC t) { { t.m_wave_per_xdl } -> std::convertible_to<size_t>; };
    constexpr bool has_n_block = requires(TC t) { { t.n_block } -> std::convertible_to<size_t>; };
    constexpr bool has_n_wave_per_xdl = requires(TC t) { { t.n_wave_per_xdl } -> std::convertible_to<size_t>; };
    
    msg += std::string("      → ") + prefix + ".m_block: " + std::string(CHECK_MARK(has_m_block)) + 
           (has_m_block ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".m_wave_per_xdl: " + std::string(CHECK_MARK(has_m_wave_per_xdl)) + 
           (has_m_wave_per_xdl ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".n_block: " + std::string(CHECK_MARK(has_n_block)) + 
           (has_n_block ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".n_wave_per_xdl: " + std::string(CHECK_MARK(has_n_wave_per_xdl)) + 
           (has_n_wave_per_xdl ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

// AccessOrderDescriptor diagnostics
template <typename T, typename AO>
consteval auto diagnose_access_order(const char* prefix) -> std::string {
    std::string msg;
    
    constexpr bool has_order = requires(AO t) { { t.order } -> std::convertible_to<std::array<size_t, 3>>; };
    
    msg += std::string("      → ") + prefix + ".order: " + std::string(CHECK_MARK(has_order)) + 
           (has_order ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

// EpilogueDescriptor diagnostics
template <typename T, typename E>
consteval auto diagnose_epilogue(const char* prefix) -> std::string {
    std::string msg;
    
    constexpr bool has_m_xdl = requires(E t) { { t.m_xdl_per_wave_per_shuffle } -> std::convertible_to<size_t>; };
    constexpr bool has_n_per_wave = requires(E t) { { t.n_per_wave_per_shuffle } -> std::convertible_to<size_t>; };
    constexpr bool has_scalar_per_vector = requires(E t) { { t.scalar_per_vector } -> std::convertible_to<size_t>; };
    
    msg += std::string("      → ") + prefix + ".m_xdl_per_wave_per_shuffle: " + std::string(CHECK_MARK(has_m_xdl)) + 
           (has_m_xdl ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".n_per_wave_per_shuffle: " + std::string(CHECK_MARK(has_n_per_wave)) + 
           (has_n_per_wave ? "\n" : " (missing or wrong type)\n");
    msg += std::string("      → ") + prefix + ".scalar_per_vector: " + std::string(CHECK_MARK(has_scalar_per_vector)) + 
           (has_scalar_per_vector ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

} // namespace detail

// Detailed diagnostic functions for high-level concepts
template <typename T>
consteval auto detailed_diagnostic_ConvAlgorithmDescriptor() -> std::string {
    return ""; // Base concept, no sub-requirements to check
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesThreadBlock() -> std::string {
    if constexpr (!requires { T::thread_block; }) {
        return "      → T::thread_block member: [✗] (not found)\n";
    } else {
        return "      → T::thread_block member: [✓]\n" + 
               detail::diagnose_thread_block_descriptor<T>();
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGridwiseFwdXdlGemm() -> std::string {
    std::string msg;
    
    constexpr bool has_ak1 = requires { { T::ak1 } -> std::convertible_to<size_t>; };
    constexpr bool has_bk1 = requires { { T::bk1 } -> std::convertible_to<size_t>; };
    constexpr bool has_xdl_params = requires { T::xdl_params; };
    
    msg += "      → T::ak1: " + std::string(CHECK_MARK(has_ak1)) + 
           (has_ak1 ? "\n" : " (missing or wrong type)\n");
    msg += "      → T::bk1: " + std::string(CHECK_MARK(has_bk1)) + 
           (has_bk1 ? "\n" : " (missing or wrong type)\n");
    msg += "      → T::xdl_params member: " + std::string(CHECK_MARK(has_xdl_params)) + 
            (has_xdl_params ? "\n" : " (missing or wrong type)\n");
    
    if constexpr (has_xdl_params) {
        msg += detail::diagnose_xdl_params<T, decltype(T::xdl_params)>();
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGridwiseBwdXdlGemm() -> std::string {
    std::string msg;
    
    constexpr bool has_k0 = requires { { T::k0_per_block } -> std::convertible_to<size_t>; };
    constexpr bool has_k1 = requires { { T::k1 } -> std::convertible_to<size_t>; };
    constexpr bool has_xdl_params = requires { T::xdl_params; };
    
    msg += "      → T::k0_per_block: " + std::string(CHECK_MARK(has_k0)) + 
           (has_k0 ? "\n" : " (missing or wrong type)\n");
    msg += "      → T::k1: " + std::string(CHECK_MARK(has_k1)) + 
           (has_k1 ? "\n" : " (missing or wrong type)\n");
    msg += "      → T::xdl_params member: " + std::string(CHECK_MARK(has_xdl_params)) + 
           (has_xdl_params ? "\n" : " (missing or wrong type)\n");
    
    if constexpr (has_xdl_params) {
        msg += detail::diagnose_xdl_params<T, decltype(T::xdl_params)>();
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesBlockTransfer() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a = requires { T::transfer.a; };
    constexpr bool has_b = requires { T::transfer.b; };
    constexpr bool has_c = requires { T::transfer.c; };
    
    msg += "      → T::transfer.a: " + std::string(CHECK_MARK(has_a)) + "\n";
    msg += "      → T::transfer.b: " + std::string(CHECK_MARK(has_b)) + "\n";
    msg += "      → T::transfer.c: " + std::string(CHECK_MARK(has_c)) + "\n";
    
    if constexpr (has_a && requires { T::transfer.a.block_transfer; }) {
        msg += detail::diagnose_block_transfer<T, decltype(T::transfer.a.block_transfer)>("transfer.a.block_transfer");
    } else if constexpr (has_a) {
        msg += "      → T::transfer.a.block_transfer: [✗] (missing)\n";
    }
    
    if constexpr (has_b && requires { T::transfer.b.block_transfer; }) {
        msg += detail::diagnose_block_transfer<T, decltype(T::transfer.b.block_transfer)>("transfer.b.block_transfer");
    } else if constexpr (has_b) {
        msg += "      → T::transfer.b.block_transfer: [✗] (missing)\n";
    }
    
    if constexpr (has_c && requires { T::transfer.c.thread_cluster_dims; }) {
        msg += detail::diagnose_thread_cluster<T, decltype(T::transfer.c.thread_cluster_dims)>("transfer.c.thread_cluster_dims");
    } else if constexpr (has_c) {
        msg += "      → T::transfer.c.thread_cluster_dims: [✗] (missing)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesLdsTransfer() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a = requires { T::transfer.a; };
    constexpr bool has_b = requires { T::transfer.b; };
    constexpr bool has_c = requires { T::transfer.c; };
    
    if constexpr (has_a && requires { T::transfer.a.lds_transfer; }) {
        msg += detail::diagnose_lds_transfer<T, decltype(T::transfer.a.lds_transfer)>("transfer.a.lds_transfer");
    } else if constexpr (has_a) {
        msg += "      → T::transfer.a.lds_transfer: [✗] (missing)\n";
    }
    
    if constexpr (has_b && requires { T::transfer.b.lds_transfer; }) {
        msg += detail::diagnose_lds_transfer<T, decltype(T::transfer.b.lds_transfer)>("transfer.b.lds_transfer");
    } else if constexpr (has_b) {
        msg += "      → T::transfer.b.lds_transfer: [✗] (missing)\n";
    }
    
    if constexpr (has_c && requires { T::transfer.c.epilogue; }) {
        msg += detail::diagnose_epilogue<T, decltype(T::transfer.c.epilogue)>("transfer.c.epilogue");
    } else if constexpr (has_c) {
        msg += "      → T::transfer.c.epilogue: [✗] (missing)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesThreadClusterAccessOrder() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    if constexpr (!has_transfer) {
        return "      → T::transfer member: [✗] (not found)\n";
    }
    
    constexpr bool has_a = requires { T::transfer.a; };
    constexpr bool has_b = requires { T::transfer.b; };
    
    if constexpr (has_a && requires { T::transfer.a.block_transfer_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.a.block_transfer_access_order)>("transfer.a.block_transfer_access_order");
    } else if constexpr (has_a) {
        msg += "      → T::transfer.a.block_transfer_access_order: [✗] (missing)\n";
    }
    
    if constexpr (has_b && requires { T::transfer.b.block_transfer_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.b.block_transfer_access_order)>("transfer.b.block_transfer_access_order");
    } else if constexpr (has_b) {
        msg += "      → T::transfer.b.block_transfer_access_order: [✗] (missing)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesSourceAccessOrder() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    if constexpr (!has_transfer) {
        return "      → T::transfer member: [✗] (not found)\n";
    }
    
    constexpr bool has_a = requires { T::transfer.a; };
    constexpr bool has_b = requires { T::transfer.b; };
    
    if constexpr (has_a && requires { T::transfer.a.src_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.a.src_access_order)>("transfer.a.src_access_order");
    } else if constexpr (has_a) {
        msg += "      → T::transfer.a.src_access_order: [✗] (missing)\n";
    }
    
    if constexpr (has_b && requires { T::transfer.b.src_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.b.src_access_order)>("transfer.b.src_access_order");
    } else if constexpr (has_b) {
        msg += "      → T::transfer.b.src_access_order: [✗] (missing)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesBlockGemm() -> std::string {
    std::string msg;
    
    constexpr bool has_block_gemm = requires { T::block_gemm; };
    msg += "      → T::block_gemm member: " + std::string(CHECK_MARK(has_block_gemm)) + "\n";
    
    if constexpr (!has_block_gemm) {
        return msg;
    }
    
    constexpr bool has_pipeline = requires { { T::block_gemm.pipeline_version } -> std::convertible_to<PipelineVersion>; };
    constexpr bool has_scheduler = requires { { T::block_gemm.scheduler } -> std::convertible_to<PipelineScheduler>; };
    
    msg += "      → block_gemm.pipeline_version: " + std::string(CHECK_MARK(has_pipeline)) + 
           (has_pipeline ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.scheduler: " + std::string(CHECK_MARK(has_scheduler)) + 
           (has_scheduler ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesFwdConvSpecialization() -> std::string {
    constexpr bool has_member = requires { { T::fwd_specialization } -> std::convertible_to<ConvSpecialization>; };
    return "      → T::fwd_specialization: " + std::string(CHECK_MARK(has_member)) + 
           (has_member ? "\n" : " (missing or wrong type)\n");
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesBwdWeightConvSpecialization() -> std::string {
    constexpr bool has_member = requires { { T::bwd_weight_specialization } -> std::convertible_to<ConvSpecialization>; };
    return "      → T::bwd_weight_specialization: " + std::string(CHECK_MARK(has_member)) + 
           (has_member ? "\n" : " (missing or wrong type)\n");
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGemmSpecialization() -> std::string {
    constexpr bool has_member = requires { { T::gemm_specialization } -> std::convertible_to<GemmSpecialization>; };
    return "      → T::gemm_specialization: " + std::string(CHECK_MARK(has_member)) + 
           (has_member ? "\n" : " (missing or wrong type)\n");
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesNumPrefetchStages() -> std::string {
    constexpr bool has_member = requires { { T::num_gemm_k_prefetch_stages } -> std::convertible_to<size_t>; };
    return "      → T::num_gemm_k_prefetch_stages: " + std::string(CHECK_MARK(has_member)) + 
           (has_member ? "\n" : " (missing or wrong type)\n");
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesNumGroupsToMerge() -> std::string {
    constexpr bool has_member = requires { { T::num_groups_to_merge } -> std::convertible_to<size_t>; };
    return "      → T::num_groups_to_merge: " + std::string(CHECK_MARK(has_member)) + 
           (has_member ? "\n" : " (missing or wrong type)\n");
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesLoopScheduler() -> std::string {
    constexpr bool has_member = requires { { T::loop_scheduler } -> std::convertible_to<PipelineScheduler>; };
    return "      → T::loop_scheduler: " + std::string(CHECK_MARK(has_member)) + 
           (has_member ? "\n" : " (missing or wrong type)\n");
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesLargeTensorSupport() -> std::string {
    std::string msg;
    constexpr bool has_specialization = requires { { T::specialization } -> std::convertible_to<ConvAlgorithmSpecialization>; };
    msg += "      → T::specialization: " + std::string(CHECK_MARK(has_specialization)) + 
           (has_specialization ? "\n" : " (missing or wrong type)\n");
    
    if constexpr (has_specialization) {
        constexpr bool is_large_tensor = (T::specialization == ConvAlgorithmSpecialization::LARGE_TENSOR);
        msg += "      → specialization == LARGE_TENSOR: " + std::string(CHECK_MARK(is_large_tensor)) + "\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTransposeTransfer() -> std::string {
    std::string msg;
    constexpr bool has_src = requires { { T::max_transpose_transfer_src_scalar_per_vector } -> std::convertible_to<size_t>; };
    constexpr bool has_dst = requires { { T::max_transpose_transfer_dst_scalar_per_vector } -> std::convertible_to<size_t>; };
    
    msg += "      → T::max_transpose_transfer_src_scalar_per_vector: " + std::string(CHECK_MARK(has_src)) + 
           (has_src ? "\n" : " (missing or wrong type)\n");
    msg += "      → T::max_transpose_transfer_dst_scalar_per_vector: " + std::string(CHECK_MARK(has_dst)) + 
           (has_dst ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGridwiseWmmaGemm() -> std::string {
    std::string msg;
    constexpr bool has_gridwise_gemm = requires { T::gridwise_gemm; };
    msg += "      → T::gridwise_gemm member: " + std::string(CHECK_MARK(has_gridwise_gemm)) + "\n";
    
    if constexpr (!has_gridwise_gemm) {
        return msg;
    }
    
    using GG = decltype(T::gridwise_gemm);
    constexpr bool has_k1 = requires(GG t) { { t.k1 } -> std::convertible_to<size_t>; };
    constexpr bool has_m_per_wmma = requires(GG t) { { t.m_per_wmma } -> std::convertible_to<size_t>; };
    constexpr bool has_n_per_wmma = requires(GG t) { { t.n_per_wmma } -> std::convertible_to<size_t>; };
    constexpr bool has_m_wmma_per_wave = requires(GG t) { { t.m_wmma_per_wave } -> std::convertible_to<size_t>; };
    constexpr bool has_n_wmma_per_wave = requires(GG t) { { t.n_wmma_per_wave } -> std::convertible_to<size_t>; };
    constexpr bool has_pipeline = requires(GG t) { { t.pipeline_version } -> std::convertible_to<PipelineVersion>; };
    
    msg += "      → gridwise_gemm.k1: " + std::string(CHECK_MARK(has_k1)) + (has_k1 ? "\n" : " (missing or wrong type)\n");
    msg += "      → gridwise_gemm.m_per_wmma: " + std::string(CHECK_MARK(has_m_per_wmma)) + (has_m_per_wmma ? "\n" : " (missing or wrong type)\n");
    msg += "      → gridwise_gemm.n_per_wmma: " + std::string(CHECK_MARK(has_n_per_wmma)) + (has_n_per_wmma ? "\n" : " (missing or wrong type)\n");
    msg += "      → gridwise_gemm.m_wmma_per_wave: " + std::string(CHECK_MARK(has_m_wmma_per_wave)) + (has_m_wmma_per_wave ? "\n" : " (missing or wrong type)\n");
    msg += "      → gridwise_gemm.n_wmma_per_wave: " + std::string(CHECK_MARK(has_n_wmma_per_wave)) + (has_n_wmma_per_wave ? "\n" : " (missing or wrong type)\n");
    msg += "      → gridwise_gemm.pipeline_version: " + std::string(CHECK_MARK(has_pipeline)) + (has_pipeline ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

// Tile-specific diagnostics
template <typename T>
consteval auto detailed_diagnostic_SpecifiesTileThreadBlock() -> std::string {
    if constexpr (!requires { T::thread_block; }) {
        return "      → T::thread_block member: [✗] (not found)\n";
    } else {
        using TB = decltype(T::thread_block);
        std::string msg = "      → T::thread_block member: [✓]\n";
        
        constexpr bool has_tile_m = requires(TB t) { { t.tile_size.m } -> std::convertible_to<size_t>; };
        constexpr bool has_tile_n = requires(TB t) { { t.tile_size.n } -> std::convertible_to<size_t>; };
        constexpr bool has_tile_k = requires(TB t) { { t.tile_size.k } -> std::convertible_to<size_t>; };
        
        msg += "      → thread_block.tile_size.m: " + std::string(CHECK_MARK(has_tile_m)) + (has_tile_m ? "\n" : " (missing or wrong type)\n");
        msg += "      → thread_block.tile_size.n: " + std::string(CHECK_MARK(has_tile_n)) + (has_tile_n ? "\n" : " (missing or wrong type)\n");
        msg += "      → thread_block.tile_size.k: " + std::string(CHECK_MARK(has_tile_k)) + (has_tile_k ? "\n" : " (missing or wrong type)\n");
        
        return msg;
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTileTransfer() -> std::string {
    std::string msg;
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a_scalar = requires { { T::transfer.a_scalar_per_vector } -> std::convertible_to<size_t>; };
    constexpr bool has_b_scalar = requires { { T::transfer.b_scalar_per_vector } -> std::convertible_to<size_t>; };
    constexpr bool has_c_scalar = requires { { T::transfer.c_scalar_per_vector } -> std::convertible_to<size_t>; };
    
    msg += "      → transfer.a_scalar_per_vector: " + std::string(CHECK_MARK(has_a_scalar)) + (has_a_scalar ? "\n" : " (missing or wrong type)\n");
    msg += "      → transfer.b_scalar_per_vector: " + std::string(CHECK_MARK(has_b_scalar)) + (has_b_scalar ? "\n" : " (missing or wrong type)\n");
    msg += "      → transfer.c_scalar_per_vector: " + std::string(CHECK_MARK(has_c_scalar)) + (has_c_scalar ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTileConvSpecialization() -> std::string {
    constexpr bool has_member = requires { { T::specialization } -> std::convertible_to<TileConvSpecialization>; };
    return "      → T::specialization: " + std::string(CHECK_MARK(has_member)) + (has_member ? "\n" : " (missing or wrong type)\n");
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTileBlockGemm() -> std::string {
    std::string msg;
    constexpr bool has_block_gemm = requires { T::block_gemm; };
    msg += "      → T::block_gemm member: " + std::string(CHECK_MARK(has_block_gemm)) + "\n";
    
    if constexpr (!has_block_gemm) {
        return msg;
    }
    
    using BG = decltype(T::block_gemm);
    constexpr bool has_warps_m = requires(BG t) { { t.warps.m } -> std::convertible_to<int>; };
    constexpr bool has_warps_n = requires(BG t) { { t.warps.n } -> std::convertible_to<int>; };
    constexpr bool has_warps_k = requires(BG t) { { t.warps.k } -> std::convertible_to<int>; };
    constexpr bool has_warp_tile_m = requires(BG t) { { t.warp_tile.m } -> std::convertible_to<int>; };
    constexpr bool has_warp_tile_n = requires(BG t) { { t.warp_tile.n } -> std::convertible_to<int>; };
    constexpr bool has_warp_tile_k = requires(BG t) { { t.warp_tile.k } -> std::convertible_to<int>; };
    constexpr bool has_double_smem = requires(BG t) { { t.double_smem_buffer } -> std::convertible_to<bool>; };
    constexpr bool has_num_wave_groups = requires(BG t) { { t.num_wave_groups } -> std::convertible_to<int>; };
    constexpr bool has_pipeline = requires(BG t) { { t.pipeline_version } -> std::convertible_to<PipelineVersion>; };
    constexpr bool has_scheduler = requires(BG t) { { t.scheduler } -> std::convertible_to<PipelineScheduler>; };
    
    msg += "      → block_gemm.warps.m: " + std::string(CHECK_MARK(has_warps_m)) + (has_warps_m ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.warps.n: " + std::string(CHECK_MARK(has_warps_n)) + (has_warps_n ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.warps.k: " + std::string(CHECK_MARK(has_warps_k)) + (has_warps_k ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.warp_tile.m: " + std::string(CHECK_MARK(has_warp_tile_m)) + (has_warp_tile_m ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.warp_tile.n: " + std::string(CHECK_MARK(has_warp_tile_n)) + (has_warp_tile_n ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.warp_tile.k: " + std::string(CHECK_MARK(has_warp_tile_k)) + (has_warp_tile_k ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.double_smem_buffer: " + std::string(CHECK_MARK(has_double_smem)) + (has_double_smem ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.num_wave_groups: " + std::string(CHECK_MARK(has_num_wave_groups)) + (has_num_wave_groups ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.pipeline_version: " + std::string(CHECK_MARK(has_pipeline)) + (has_pipeline ? "\n" : " (missing or wrong type)\n");
    msg += "      → block_gemm.scheduler: " + std::string(CHECK_MARK(has_scheduler)) + (has_scheduler ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTileOptimizations() -> std::string {
    std::string msg;
    constexpr bool has_optimizations = requires { T::optimizations; };
    msg += "      → T::optimizations member: " + std::string(CHECK_MARK(has_optimizations)) + "\n";
    
    if constexpr (!has_optimizations) {
        return msg;
    }
    
    using OPT = decltype(T::optimizations);
    constexpr bool has_num_groups = requires(OPT t) { { t.num_groups_to_merge } -> std::convertible_to<int>; };
    constexpr bool has_split_image = requires(OPT t) { { t.split_image } -> std::convertible_to<bool>; };
    constexpr bool has_explicit_gemm = requires(OPT t) { { t.explicit_gemm } -> std::convertible_to<bool>; };
    
    msg += "      → optimizations.num_groups_to_merge: " + std::string(CHECK_MARK(has_num_groups)) + (has_num_groups ? "\n" : " (missing or wrong type)\n");
    msg += "      → optimizations.split_image: " + std::string(CHECK_MARK(has_split_image)) + (has_split_image ? "\n" : " (missing or wrong type)\n");
    msg += "      → optimizations.explicit_gemm: " + std::string(CHECK_MARK(has_explicit_gemm)) + (has_explicit_gemm ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

// DL-specific diagnostics
template <typename T>
consteval auto detailed_diagnostic_SpecifiesDlThreadConfig() -> std::string {
    std::string msg;
    constexpr bool has_thread_config = requires { T::thread_config; };
    msg += "      → T::thread_config member: " + std::string(CHECK_MARK(has_thread_config)) + "\n";
    
    if constexpr (!has_thread_config) {
        return msg;
    }
    
    using TC = decltype(T::thread_config);
    constexpr bool has_k0 = requires(TC t) { { t.k0_per_block } -> std::convertible_to<size_t>; };
    constexpr bool has_k1 = requires(TC t) { { t.k1 } -> std::convertible_to<size_t>; };
    constexpr bool has_m1 = requires(TC t) { { t.m1_per_thread } -> std::convertible_to<size_t>; };
    constexpr bool has_n1 = requires(TC t) { { t.n1_per_thread } -> std::convertible_to<size_t>; };
    constexpr bool has_k = requires(TC t) { { t.k_per_thread } -> std::convertible_to<size_t>; };
    
    msg += "      → thread_config.k0_per_block: " + std::string(CHECK_MARK(has_k0)) + (has_k0 ? "\n" : " (missing or wrong type)\n");
    msg += "      → thread_config.k1: " + std::string(CHECK_MARK(has_k1)) + (has_k1 ? "\n" : " (missing or wrong type)\n");
    msg += "      → thread_config.m1_per_thread: " + std::string(CHECK_MARK(has_m1)) + (has_m1 ? "\n" : " (missing or wrong type)\n");
    msg += "      → thread_config.n1_per_thread: " + std::string(CHECK_MARK(has_n1)) + (has_n1 ? "\n" : " (missing or wrong type)\n");
    msg += "      → thread_config.k_per_thread: " + std::string(CHECK_MARK(has_k)) + (has_k ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesDlThreadCluster() -> std::string {
    std::string msg;
    constexpr bool has_thread_cluster = requires { T::thread_cluster; };
    msg += "      → T::thread_cluster member: " + std::string(CHECK_MARK(has_thread_cluster)) + "\n";
    
    if constexpr (!has_thread_cluster) {
        return msg;
    }
    
    using TC = decltype(T::thread_cluster);
    constexpr bool has_m1_xs = requires(TC t) { { t.m1_xs } -> std::convertible_to<std::array<size_t, 2>>; };
    constexpr bool has_n1_xs = requires(TC t) { { t.n1_xs } -> std::convertible_to<std::array<size_t, 2>>; };
    
    msg += "      → thread_cluster.m1_xs: " + std::string(CHECK_MARK(has_m1_xs)) + (has_m1_xs ? "\n" : " (missing or wrong type)\n");
    msg += "      → thread_cluster.n1_xs: " + std::string(CHECK_MARK(has_n1_xs)) + (has_n1_xs ? "\n" : " (missing or wrong type)\n");
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesDlBlockTransfer() -> std::string {
    std::string msg;
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a = requires { T::transfer.a; };
    constexpr bool has_b = requires { T::transfer.b; };
    msg += "      → T::transfer.a: " + std::string(CHECK_MARK(has_a)) + "\n";
    msg += "      → T::transfer.b: " + std::string(CHECK_MARK(has_b)) + "\n";
    
    if constexpr (has_a && requires { T::transfer.a.block_transfer; }) {
        using ABT = decltype(T::transfer.a.block_transfer);
        constexpr bool has_thread_slice = requires(ABT t) { { t.thread_slice_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_thread_cluster = requires(ABT t) { { t.thread_cluster_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_cluster_arrange = requires(ABT t) { { t.thread_cluster_arrange_order } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_src_access = requires(ABT t) { { t.src_access_order } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_src_vector = requires(ABT t) { { t.src_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_src_contiguous = requires(ABT t) { { t.src_vector_tensor_contiguous_dim_order } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_dst_vector = requires(ABT t) { { t.dst_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        
        msg += "      → transfer.a.block_transfer.thread_slice_lengths: " + std::string(CHECK_MARK(has_thread_slice)) + (has_thread_slice ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.a.block_transfer.thread_cluster_lengths: " + std::string(CHECK_MARK(has_thread_cluster)) + (has_thread_cluster ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.a.block_transfer.thread_cluster_arrange_order: " + std::string(CHECK_MARK(has_cluster_arrange)) + (has_cluster_arrange ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.a.block_transfer.src_access_order: " + std::string(CHECK_MARK(has_src_access)) + (has_src_access ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.a.block_transfer.src_vector_tensor_lengths: " + std::string(CHECK_MARK(has_src_vector)) + (has_src_vector ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.a.block_transfer.src_vector_tensor_contiguous_dim_order: " + std::string(CHECK_MARK(has_src_contiguous)) + (has_src_contiguous ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.a.block_transfer.dst_vector_tensor_lengths: " + std::string(CHECK_MARK(has_dst_vector)) + (has_dst_vector ? "\n" : " (missing or wrong type)\n");
    } else if constexpr (has_a) {
        msg += "      → T::transfer.a.block_transfer: [✗] (missing)\n";
    }
    
    // Similar checks for transfer.b
    if constexpr (has_b && requires { T::transfer.b.block_transfer; }) {
        msg += "      → T::transfer.b.block_transfer: [✓] (similar fields as transfer.a)\n";
    } else if constexpr (has_b) {
        msg += "      → T::transfer.b.block_transfer: [✗] (missing)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesDlEpilogue() -> std::string {
    std::string msg;
    constexpr bool has_transfer = requires { T::transfer; };
    if constexpr (!has_transfer) {
        return "      → T::transfer member: [✗] (not found)\n";
    }
    
    constexpr bool has_c = requires { T::transfer.c; };
    msg += "      → T::transfer.c: " + std::string(CHECK_MARK(has_c)) + "\n";
    
    if constexpr (has_c && requires { T::transfer.c.epilogue; }) {
        using E = decltype(T::transfer.c.epilogue);
        constexpr bool has_src_dst_access = requires(E t) { { t.src_dst_access_order } -> std::convertible_to<std::array<size_t, 6>>; };
        constexpr bool has_src_dst_vector_dim = requires(E t) { { t.src_dst_vector_dim } -> std::convertible_to<size_t>; };
        constexpr bool has_dst_scalar = requires(E t) { { t.dst_scalar_per_vector } -> std::convertible_to<size_t>; };
        
        msg += "      → transfer.c.epilogue.src_dst_access_order: " + std::string(CHECK_MARK(has_src_dst_access)) + (has_src_dst_access ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.c.epilogue.src_dst_vector_dim: " + std::string(CHECK_MARK(has_src_dst_vector_dim)) + (has_src_dst_vector_dim ? "\n" : " (missing or wrong type)\n");
        msg += "      → transfer.c.epilogue.dst_scalar_per_vector: " + std::string(CHECK_MARK(has_dst_scalar)) + (has_dst_scalar ? "\n" : " (missing or wrong type)\n");
    } else if constexpr (has_c) {
        msg += "      → T::transfer.c.epilogue: [✗] (missing)\n";
    }
    
    return msg;
}

} // namespace ck::detail::diagnostics
