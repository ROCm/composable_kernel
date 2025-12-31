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

// Helper to get type information
template<typename MemberType>
consteval auto get_type_info() -> const char* {
    // Returns a descriptive string about the type
    if constexpr (std::is_same_v<MemberType, size_t>) {
        return " (type: size_t)";
    } else if constexpr (std::is_same_v<MemberType, int>) {
        return " (type: int)";
    } else if constexpr (std::is_same_v<MemberType, bool>) {
        return " (type: bool)";
    } else if constexpr (std::is_same_v<MemberType, PipelineVersion>) {
        return " (type: PipelineVersion)";
    } else if constexpr (std::is_same_v<MemberType, PipelineScheduler>) {
        return " (type: PipelineScheduler)";
    } else if constexpr (std::is_same_v<MemberType, ConvSpecialization>) {
        return " (type: ConvSpecialization)";
    } else if constexpr (std::is_same_v<MemberType, GemmSpecialization>) {
        return " (type: GemmSpecialization)";
    } else if constexpr (std::is_same_v<MemberType, TileConvSpecialization>) {
        return " (type: TileConvSpecialization)";
    } else if constexpr (std::is_same_v<MemberType, ConvAlgorithmSpecialization>) {
        return " (type: ConvAlgorithmSpecialization)";
    } else if constexpr (std::is_same_v<MemberType, std::array<size_t, 2>>) {
        return " (type: std::array<size_t, 2>)";
    } else if constexpr (std::is_same_v<MemberType, std::array<size_t, 3>>) {
        return " (type: std::array<size_t, 3>)";
    } else if constexpr (std::is_same_v<MemberType, std::array<size_t, 4>>) {
        return " (type: std::array<size_t, 4>)";
    } else if constexpr (std::is_same_v<MemberType, std::array<size_t, 6>>) {
        return " (type: std::array<size_t, 6>)";
    } else {
        return " (type: found but unknown)";
    }
}

// ThreadBlockDescriptor diagnostics
template <typename T>
consteval auto diagnose_thread_block_descriptor() -> std::string {
    if constexpr (!requires { T::thread_block; }) {
        return "      → T::thread_block member: [✗] (missing member)\n";
    } else {
        using TB = decltype(T::thread_block);
        std::string msg;
        
        if constexpr (requires(TB t) { t.block_size; }) {
            using BlockSizeType = decltype(std::declval<TB>().block_size);
            constexpr bool convertible = SizeType<BlockSizeType>;
            msg += "      → thread_block.block_size: " + std::string(CHECK_MARK(convertible)) + 
                   (convertible ? "" : std::string(get_type_info<BlockSizeType>())) + "\n";
        } else {
            msg += "      → thread_block.block_size: [✗] (missing member)\n";
        }
        
        if constexpr (requires(TB t) { t.tile_size.m; }) {
            using TileMType = decltype(std::declval<TB>().tile_size.m);
            constexpr bool convertible = SizeType<TileMType>;
            msg += "      → thread_block.tile_size.m: " + std::string(CHECK_MARK(convertible)) + 
                   (convertible ? "" : std::string(get_type_info<TileMType>())) + "\n";
        } else {
            msg += "      → thread_block.tile_size.m: [✗] (missing member)\n";
        }
        
        if constexpr (requires(TB t) { t.tile_size.n; }) {
            using TileNType = decltype(std::declval<TB>().tile_size.n);
            constexpr bool convertible = SizeType<TileNType>;
            msg += "      → thread_block.tile_size.n: " + std::string(CHECK_MARK(convertible)) + 
                   (convertible ? "" : std::string(get_type_info<TileNType>())) + "\n";
        } else {
            msg += "      → thread_block.tile_size.n: [✗] (missing member)\n";
        }
        
        if constexpr (requires(TB t) { t.tile_size.k; }) {
            using TileKType = decltype(std::declval<TB>().tile_size.k);
            constexpr bool convertible = SizeType<TileKType>;
            msg += "      → thread_block.tile_size.k: " + std::string(CHECK_MARK(convertible)) + 
                   (convertible ? "" : std::string(get_type_info<TileKType>())) + "\n";
        } else {
            msg += "      → thread_block.tile_size.k: [✗] (missing member)\n";
        }
        
        return msg;
    }
}

// GridwiseXdlGemmDescriptor diagnostics
template <typename T, typename XdlParams>
consteval auto diagnose_xdl_params() -> std::string {
    std::string msg;
    
    if constexpr (requires(XdlParams t) { t.m_per_xdl; }) {
        using MPerXdlType = decltype(std::declval<XdlParams>().m_per_xdl);
        constexpr bool convertible = SizeType<MPerXdlType>;
        msg += "          → xdl_params.m_per_xdl: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<MPerXdlType>())) + "\n";
    } else {
        msg += "          → xdl_params.m_per_xdl: [✗] (missing member)\n";
    }
    
    if constexpr (requires(XdlParams t) { t.n_per_xdl; }) {
        using NPerXdlType = decltype(std::declval<XdlParams>().n_per_xdl);
        constexpr bool convertible = SizeType<NPerXdlType>;
        msg += "          → xdl_params.n_per_xdl: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<NPerXdlType>())) + "\n";
    } else {
        msg += "          → xdl_params.n_per_xdl: [✗] (missing member)\n";
    }
    
    if constexpr (requires(XdlParams t) { t.m_xdl_per_wave; }) {
        using MXdlPerWaveType = decltype(std::declval<XdlParams>().m_xdl_per_wave);
        constexpr bool convertible = SizeType<MXdlPerWaveType>;
        msg += "          → xdl_params.m_xdl_per_wave: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<MXdlPerWaveType>())) + "\n";
    } else {
        msg += "          → xdl_params.m_xdl_per_wave: [✗] (missing member)\n";
    }
    
    if constexpr (requires(XdlParams t) { t.n_xdl_per_wave; }) {
        using NXdlPerWaveType = decltype(std::declval<XdlParams>().n_xdl_per_wave);
        constexpr bool convertible = SizeType<NXdlPerWaveType>;
        msg += "          → xdl_params.n_xdl_per_wave: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<NXdlPerWaveType>())) + "\n";
    } else {
        msg += "          → xdl_params.n_xdl_per_wave: [✗] (missing member)\n";
    }
    
    return msg;
}

// BlockTransferDescriptor diagnostics
template <typename T, typename BT>
consteval auto diagnose_block_transfer(const char* prefix) -> std::string {
    std::string msg;
    
    if constexpr (requires(BT t) { t.k0; }) {
        using K0Type = decltype(std::declval<BT>().k0);
        constexpr bool convertible = std::convertible_to<K0Type, size_t>;
        msg += std::string("          → ") + prefix + ".k0: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<K0Type>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".k0: [✗] (missing member)\n";
    }
    
    if constexpr (requires(BT t) { t.m_n; }) {
        using MNType = decltype(std::declval<BT>().m_n);
        constexpr bool convertible = std::convertible_to<MNType, size_t>;
        msg += std::string("          → ") + prefix + ".m_n: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<MNType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".m_n: [✗] (missing member)\n";
    }
    
    if constexpr (requires(BT t) { t.k1; }) {
        using K1Type = decltype(std::declval<BT>().k1);
        constexpr bool convertible = std::convertible_to<K1Type, size_t>;
        msg += std::string("          → ") + prefix + ".k1: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<K1Type>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".k1: [✗] (missing member)\n";
    }
    
    return msg;
}

// BlockTransferDescriptor4D diagnostics (requires k_batch_size)
template <typename T, typename BT>
consteval auto diagnose_block_transfer_4d(const char* prefix) -> std::string {
    std::string msg;
    
    if constexpr (requires(BT t) { t.k0; }) {
        using K0Type = decltype(std::declval<BT>().k0);
        constexpr bool convertible = std::convertible_to<K0Type, size_t>;
        msg += std::string("          → ") + prefix + ".k0: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<K0Type>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".k0: [✗] (missing member)\n";
    }
    
    if constexpr (requires(BT t) { t.m_n; }) {
        using MNType = decltype(std::declval<BT>().m_n);
        constexpr bool convertible = std::convertible_to<MNType, size_t>;
        msg += std::string("          → ") + prefix + ".m_n: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<MNType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".m_n: [✗] (missing member)\n";
    }
    
    if constexpr (requires(BT t) { t.k1; }) {
        using K1Type = decltype(std::declval<BT>().k1);
        constexpr bool convertible = std::convertible_to<K1Type, size_t>;
        msg += std::string("          → ") + prefix + ".k1: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<K1Type>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".k1: [✗] (missing member)\n";
    }
    
    // k_batch_size is required for Bwd descriptor
    if constexpr (requires(BT t) { t.k_batch_size; }) {
        using KBatchType = decltype(std::declval<BT>().k_batch_size);
        constexpr bool convertible = std::convertible_to<KBatchType, size_t>;
        msg += std::string("          → ") + prefix + ".k_batch_size: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<KBatchType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".k_batch_size: [✗] (missing member)\n";
    }
    
    return msg;
}

// LdsTransferDescriptor diagnostics
template <typename T, typename LT>
consteval auto diagnose_lds_transfer(const char* prefix) -> std::string {
    std::string msg;
    
    if constexpr (requires(LT t) { t.src_vector_dim; }) {
        using SrcVectorDimType = decltype(std::declval<LT>().src_vector_dim);
        constexpr bool convertible = std::convertible_to<SrcVectorDimType, size_t>;
        msg += std::string("          → ") + prefix + ".src_vector_dim: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<SrcVectorDimType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".src_vector_dim: [✗] (missing member)\n";
    }
    
    if constexpr (requires(LT t) { t.src_scalar_per_vector; }) {
        using SrcScalarType = decltype(std::declval<LT>().src_scalar_per_vector);
        constexpr bool convertible = std::convertible_to<SrcScalarType, size_t>;
        msg += std::string("          → ") + prefix + ".src_scalar_per_vector: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<SrcScalarType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".src_scalar_per_vector: [✗] (missing member)\n";
    }
    
    if constexpr (requires(LT t) { t.lds_dst_scalar_per_vector; }) {
        using LdsDstScalarType = decltype(std::declval<LT>().lds_dst_scalar_per_vector);
        constexpr bool convertible = std::convertible_to<LdsDstScalarType, size_t>;
        msg += std::string("          → ") + prefix + ".lds_dst_scalar_per_vector: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<LdsDstScalarType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".lds_dst_scalar_per_vector: [✗] (missing member)\n";
    }
    
    if constexpr (requires(LT t) { t.is_direct_load; }) {
        using IsDirectLoadType = decltype(std::declval<LT>().is_direct_load);
        constexpr bool convertible = std::convertible_to<IsDirectLoadType, bool>;
        msg += std::string("          → ") + prefix + ".is_direct_load: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<IsDirectLoadType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".is_direct_load: [✗] (missing member)\n";
    }
    
    if constexpr (requires(LT t) { t.lds_padding; }) {
        using LdsPaddingType = decltype(std::declval<LT>().lds_padding);
        constexpr bool convertible = std::convertible_to<LdsPaddingType, bool>;
        msg += std::string("          → ") + prefix + ".lds_padding: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<LdsPaddingType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".lds_padding: [✗] (missing member)\n";
    }
    
    return msg;
}

// ThreadClusterDescriptor diagnostics
template <typename T, typename TC>
consteval auto diagnose_thread_cluster(const char* prefix) -> std::string {
    std::string msg;
    
    if constexpr (requires(TC t) { t.m_block; }) {
        using MBlockType = decltype(std::declval<TC>().m_block);
        constexpr bool convertible = std::convertible_to<MBlockType, size_t>;
        msg += std::string("          → ") + prefix + ".m_block: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<MBlockType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".m_block: [✗] (missing member)\n";
    }
    
    if constexpr (requires(TC t) { t.m_wave_per_xdl; }) {
        using MWaveType = decltype(std::declval<TC>().m_wave_per_xdl);
        constexpr bool convertible = std::convertible_to<MWaveType, size_t>;
        msg += std::string("          → ") + prefix + ".m_wave_per_xdl: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<MWaveType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".m_wave_per_xdl: [✗] (missing member)\n";
    }
    
    if constexpr (requires(TC t) { t.n_block; }) {
        using NBlockType = decltype(std::declval<TC>().n_block);
        constexpr bool convertible = std::convertible_to<NBlockType, size_t>;
        msg += std::string("          → ") + prefix + ".n_block: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<NBlockType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".n_block: [✗] (missing member)\n";
    }
    
    if constexpr (requires(TC t) { t.n_wave_per_xdl; }) {
        using NWaveType = decltype(std::declval<TC>().n_wave_per_xdl);
        constexpr bool convertible = std::convertible_to<NWaveType, size_t>;
        msg += std::string("          → ") + prefix + ".n_wave_per_xdl: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<NWaveType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".n_wave_per_xdl: [✗] (missing member)\n";
    }
    
    return msg;
}

// AccessOrderDescriptor diagnostics
template <typename T, typename AO>
consteval auto diagnose_access_order(const char* prefix) -> std::string {
    std::string msg;
    
    if constexpr (requires(AO t) { t.order; }) {
        using OrderType = decltype(std::declval<AO>().order);
        constexpr bool convertible_3 = std::convertible_to<OrderType, std::array<size_t, 3>>;
        constexpr bool convertible_4 = std::convertible_to<OrderType, std::array<size_t, 4>>;
        constexpr bool convertible = convertible_3 || convertible_4;
        msg += std::string("          → ") + prefix + ".order: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<OrderType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".order: [✗] (missing member)\n";
    }
    
    return msg;
}

// EpilogueDescriptor diagnostics
template <typename T, typename E>
consteval auto diagnose_epilogue(const char* prefix) -> std::string {
    std::string msg;
    
    if constexpr (requires(E t) { t.m_xdl_per_wave_per_shuffle; }) {
        using MXdlType = decltype(std::declval<E>().m_xdl_per_wave_per_shuffle);
        constexpr bool convertible = std::convertible_to<MXdlType, size_t>;
        msg += std::string("          → ") + prefix + ".m_xdl_per_wave_per_shuffle: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<MXdlType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".m_xdl_per_wave_per_shuffle: [✗] (missing member)\n";
    }
    
    if constexpr (requires(E t) { t.n_per_wave_per_shuffle; }) {
        using NPerWaveType = decltype(std::declval<E>().n_per_wave_per_shuffle);
        constexpr bool convertible = std::convertible_to<NPerWaveType, size_t>;
        msg += std::string("          → ") + prefix + ".n_per_wave_per_shuffle: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<NPerWaveType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".n_per_wave_per_shuffle: [✗] (missing member)\n";
    }
    
    if constexpr (requires(E t) { t.scalar_per_vector; }) {
        using ScalarType = decltype(std::declval<E>().scalar_per_vector);
        constexpr bool convertible = std::convertible_to<ScalarType, size_t>;
        msg += std::string("          → ") + prefix + ".scalar_per_vector: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(get_type_info<ScalarType>())) + "\n";
    } else {
        msg += std::string("          → ") + prefix + ".scalar_per_vector: [✗] (missing member)\n";
    }
    
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
    if constexpr (!requires { { T::thread_block } -> ThreadBlockDescriptor; }) {
        return "      → T::thread_block member: [✗] (missing or wrong type)\n";
    } else {
        return "      → T::thread_block member: [✓]\n" + 
               detail::diagnose_thread_block_descriptor<T>();
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGridwiseFwdXdlGemm() -> std::string {
    std::string msg;
    
    if constexpr (!requires(T t) { { t.gridwise_gemm } -> GridwiseFwdXdlGemmDescriptor; }) {
        return "      → T::gridwise_gemm member: [✗] (missing or wrong type)\n";
    }
    
    msg += "      → T::gridwise_gemm member: [✓]\n";
    using GG = decltype(T::gridwise_gemm);
    
    if constexpr (requires(GG t) { t.ak1; }) {
        using AK1Type = decltype(std::declval<GG>().ak1);
        constexpr bool convertible = std::convertible_to<AK1Type, size_t>;
        msg += "      → gridwise_gemm.ak1: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<AK1Type>())) + "\n";
    } else {
        msg += "      → gridwise_gemm.ak1: [✗] (missing member)\n";
    }
    
    if constexpr (requires(GG t) { t.bk1; }) {
        using BK1Type = decltype(std::declval<GG>().bk1);
        constexpr bool convertible = std::convertible_to<BK1Type, size_t>;
        msg += "      → gridwise_gemm.bk1: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<BK1Type>())) + "\n";
    } else {
        msg += "      → gridwise_gemm.bk1: [✗] (missing member)\n";
    }
    
    if constexpr (requires(GG t) { t.xdl_params; }) {
        msg += "      → gridwise_gemm.xdl_params member: [✓]\n";
        msg += detail::diagnose_xdl_params<T, decltype(std::declval<GG>().xdl_params)>();
    } else {
        msg += "      → gridwise_gemm.xdl_params: [✗] (missing member)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGridwiseBwdXdlGemm() -> std::string {
    std::string msg;
    
    if constexpr (!requires(T t) { { t.gridwise_gemm } -> GridwiseBwdXdlGemmDescriptor; }) {
        return "      → T::gridwise_gemm member: [✗] (missing or wrong type)\n";
    }
    
    msg += "      → T::gridwise_gemm member: [✓]\n";
    using GG = decltype(T::gridwise_gemm);
    
    if constexpr (requires(GG t) { t.k1; }) {
        using K1Type = decltype(std::declval<GG>().k1);
        constexpr bool convertible = std::convertible_to<K1Type, size_t>;
        msg += "      → gridwise_gemm.k1: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<K1Type>())) + "\n";
    } else {
        msg += "      → gridwise_gemm.k1: [✗] (missing member)\n";
    }
    
    if constexpr (requires(GG t) { t.xdl_params; }) {
        msg += "      → gridwise_gemm.xdl_params member: [✓]\n";
        msg += detail::diagnose_xdl_params<T, decltype(std::declval<GG>().xdl_params)>();
    } else {
        msg += "      → gridwise_gemm.xdl_params: [✗] (missing member)\n";
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
    
    constexpr bool has_a = requires { { T::transfer.a.block_transfer } -> BlockTransferDescriptor; };
    msg += "      → T::transfer.a: " + std::string(CHECK_MARK(has_a)) + "\n";
    if constexpr (!has_a) {
        msg += "            → T::transfer.a.block_transfer: [✗] (missing or wrong type)\n";
    }
    
    constexpr bool has_b = requires { { T::transfer.b.block_transfer } -> BlockTransferDescriptor; };
    msg += "      → T::transfer.b: " + std::string(CHECK_MARK(has_b)) + "\n";
    if constexpr (!has_b) {
        msg += "            → T::transfer.b.block_transfer: [✗] (missing or wrong type)\n";
    }
    
    constexpr bool has_c = requires { { T::transfer.c.thread_cluster_dims } -> ThreadClusterDescriptor; };
    msg += "      → T::transfer.c: " + std::string(CHECK_MARK(has_c)) + "\n";
    if constexpr (!has_c) {
        msg += "            → T::transfer.c.thread_cluster_dims: [✗] (missing or wrong type)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesBlockTransfer4D() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a = requires { { T::transfer.a.block_transfer } -> BlockTransferDescriptor4D; };
    msg += "      → T::transfer.a: " + std::string(CHECK_MARK(has_a)) + "\n";
    if constexpr (!has_a) {
        msg += "            → T::transfer.a.block_transfer: [✗] (missing or wrong type)\n";
    } else {
        msg += detail::diagnose_block_transfer_4d<T, decltype(T::transfer.a.block_transfer)>("transfer.a.block_transfer");
    }
    
    constexpr bool has_b = requires { { T::transfer.b.block_transfer } -> BlockTransferDescriptor4D; };
    msg += "      → T::transfer.b: " + std::string(CHECK_MARK(has_b)) + "\n";
    if constexpr (!has_b) {
        msg += "            → T::transfer.b.block_transfer: [✗] (missing or wrong type)\n";
    } else {
        msg += detail::diagnose_block_transfer_4d<T, decltype(T::transfer.b.block_transfer)>("transfer.b.block_transfer");
    }
    
    constexpr bool has_c = requires { { T::transfer.c.thread_cluster_dims } -> ThreadClusterDescriptor; };
    msg += "      → T::transfer.c: " + std::string(CHECK_MARK(has_c)) + "\n";
    if constexpr (!has_c) {
        msg += "            → T::transfer.c.thread_cluster_dims: [✗] (missing or wrong type)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesThreadClusterAccessOrder() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    if constexpr (!has_transfer) {
        return "      → T::transfer member: [✗] (missing member)\n";
    }
    
    constexpr bool has_a = requires { T::transfer.a; };
    constexpr bool has_b = requires { T::transfer.b; };
    
    if constexpr (has_a && requires { T::transfer.a.block_transfer_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.a.block_transfer_access_order)>("transfer.a.block_transfer_access_order");
    } else if constexpr (has_a) {
        msg += "      → T::transfer.a.block_transfer_access_order: [✗] (missing member)\n";
    }
    
    if constexpr (has_b && requires { T::transfer.b.block_transfer_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.b.block_transfer_access_order)>("transfer.b.block_transfer_access_order");
    } else if constexpr (has_b) {
        msg += "      → T::transfer.b.block_transfer_access_order: [✗] (missing member)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesSourceAccessOrder() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    if constexpr (!has_transfer) {
        return "      → T::transfer member: [✗] (missing member)\n";
    }
    
    constexpr bool has_a = requires { T::transfer.a; };
    constexpr bool has_b = requires { T::transfer.b; };
    
    if constexpr (has_a && requires { T::transfer.a.src_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.a.src_access_order)>("transfer.a.src_access_order");
    } else if constexpr (has_a) {
        msg += "      → T::transfer.a.src_access_order: [✗] (missing member)\n";
    }
    
    if constexpr (has_b && requires { T::transfer.b.src_access_order; }) {
        msg += detail::diagnose_access_order<T, decltype(T::transfer.b.src_access_order)>("transfer.b.src_access_order");
    } else if constexpr (has_b) {
        msg += "      → T::transfer.b.src_access_order: [✗] (missing member)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesBlockGemm() -> std::string {
    std::string msg;
    
    if constexpr (!requires { { T::block_gemm } -> BlockGemmDescriptor; }) {
        return "      → T::block_gemm: [✗] (missing or wrong type)\n";
    }
    
    msg += "      → T::block_gemm member: [✓]\n";
    
    if constexpr (requires { T::block_gemm.pipeline_version; }) {
        using PipelineType = decltype(T::block_gemm.pipeline_version);
        constexpr bool convertible = std::convertible_to<PipelineType, PipelineVersion>;
        msg += "      → block_gemm.pipeline_version: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<PipelineType>())) + "\n";
    } else {
        msg += "      → block_gemm.pipeline_version: [✗] (missing member)\n";
    }
    
    if constexpr (requires { T::block_gemm.scheduler; }) {
        using SchedulerType = decltype(T::block_gemm.scheduler);
        constexpr bool convertible = std::convertible_to<SchedulerType, PipelineScheduler>;
        msg += "      → block_gemm.scheduler: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<SchedulerType>())) + "\n";
    } else {
        msg += "      → block_gemm.scheduler: [✗] (missing member)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesFwdConvSpecialization() -> std::string {
    if constexpr (requires { T::fwd_specialization; }) {
        using FwdSpecType = decltype(T::fwd_specialization);
        constexpr bool convertible = std::convertible_to<FwdSpecType, ConvSpecialization>;
        return "      → T::fwd_specialization: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<FwdSpecType>())) + "\n";
    } else {
        return "      → T::fwd_specialization: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesBwdWeightConvSpecialization() -> std::string {
    if constexpr (requires { T::bwd_weight_specialization; }) {
        using BwdSpecType = decltype(T::bwd_weight_specialization);
        constexpr bool convertible = std::convertible_to<BwdSpecType, ConvSpecialization>;
        return "      → T::bwd_weight_specialization: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<BwdSpecType>())) + "\n";
    } else {
        return "      → T::bwd_weight_specialization: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGemmSpecialization() -> std::string {
    if constexpr (requires { T::gemm_specialization; }) {
        using GemmSpecType = decltype(T::gemm_specialization);
        constexpr bool convertible = std::convertible_to<GemmSpecType, GemmSpecialization>;
        return "      → T::gemm_specialization: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<GemmSpecType>())) + "\n";
    } else {
        return "      → T::gemm_specialization: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesNumPrefetchStages() -> std::string {
    if constexpr (requires { T::num_gemm_k_prefetch_stages; }) {
        using NumPrefetchType = decltype(T::num_gemm_k_prefetch_stages);
        constexpr bool convertible = std::convertible_to<NumPrefetchType, size_t>;
        return "      → T::num_gemm_k_prefetch_stages: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<NumPrefetchType>())) + "\n";
    } else {
        return "      → T::num_gemm_k_prefetch_stages: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesNumGroupsToMerge() -> std::string {
    if constexpr (requires { T::num_groups_to_merge; }) {
        using NumGroupsType = decltype(T::num_groups_to_merge);
        constexpr bool convertible = std::convertible_to<NumGroupsType, size_t>;
        return "      → T::num_groups_to_merge: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<NumGroupsType>())) + "\n";
    } else {
        return "      → T::num_groups_to_merge: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesLoopScheduler() -> std::string {
    if constexpr (requires { T::loop_scheduler; }) {
        using LoopSchedulerType = decltype(T::loop_scheduler);
        constexpr bool convertible = std::convertible_to<LoopSchedulerType, PipelineScheduler>;
        return "      → T::loop_scheduler: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<LoopSchedulerType>())) + "\n";
    } else {
        return "      → T::loop_scheduler: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesLargeTensorSupport() -> std::string {
    std::string msg;
    if constexpr (requires { T::specialization; }) {
        using SpecType = decltype(T::specialization);
        constexpr bool convertible = std::convertible_to<SpecType, ConvAlgorithmSpecialization>;
        msg += "      → T::specialization: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<SpecType>())) + "\n";
        
        if constexpr (convertible) {
            constexpr bool is_large_tensor = (T::specialization == ConvAlgorithmSpecialization::LARGE_TENSOR);
            msg += "      → specialization == LARGE_TENSOR: " + std::string(CHECK_MARK(is_large_tensor)) + "\n";
        }
    } else {
        msg += "      → T::specialization: [✗] (missing member)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTwoStageSupport() -> std::string {
    std::string msg;
    if constexpr (requires { T::specialization; }) {
        using SpecType = decltype(T::specialization);
        constexpr bool convertible = std::convertible_to<SpecType, ConvAlgorithmSpecialization>;
        msg += "      → T::specialization: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<SpecType>())) + "\n";
        
        if constexpr (convertible) {
            constexpr bool is_two_stage = (T::specialization == ConvAlgorithmSpecialization::TWO_STAGE);
            msg += "      → specialization == TWO_STAGE: " + std::string(CHECK_MARK(is_two_stage)) + "\n";
        }
    } else {
        msg += "      → T::specialization: [✗] (missing member)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGenericInstance() -> std::string {
    std::string msg;
    if constexpr (requires { T::specialization; }) {
        msg += "      → T::specialization: [✗] (member should NOT exist for generic instance)\n";
        msg += "      → This concept requires the absence of the specialization member\n";
    }
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTransposeTransfer() -> std::string {
    std::string msg;
    
    if constexpr (requires { T::max_transpose_transfer_src_scalar_per_vector; }) {
        using SrcType = decltype(T::max_transpose_transfer_src_scalar_per_vector);
        constexpr bool convertible = std::convertible_to<SrcType, size_t>;
        msg += "      → T::max_transpose_transfer_src_scalar_per_vector: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<SrcType>())) + "\n";
    } else {
        msg += "      → T::max_transpose_transfer_src_scalar_per_vector: [✗] (missing member)\n";
    }
    
    if constexpr (requires { T::max_transpose_transfer_dst_scalar_per_vector; }) {
        using DstType = decltype(T::max_transpose_transfer_dst_scalar_per_vector);
        constexpr bool convertible = std::convertible_to<DstType, size_t>;
        msg += "      → T::max_transpose_transfer_dst_scalar_per_vector: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<DstType>())) + "\n";
    } else {
        msg += "      → T::max_transpose_transfer_dst_scalar_per_vector: [✗] (missing member)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGemmBatchOptions() -> std::string {
    if constexpr (requires { T::num_conv_groups_to_merge; }) {
        using NumGroupsType = decltype(T::num_conv_groups_to_merge);
        constexpr bool convertible = std::convertible_to<NumGroupsType, size_t>;
        return "      → T::num_conv_groups_to_merge: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<NumGroupsType>())) + "\n";
    } else {
        return "      → T::num_conv_groups_to_merge: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesGridwiseWmmaGemm() -> std::string {
    std::string msg;
    constexpr bool has_gridwise_gemm = requires(T t) { { t.gridwise_gemm } -> GridwiseWmmaGemmDescriptor; };
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
    if constexpr (!requires { { T::thread_block } -> TileThreadBlockDescriptor; }) {
        return "      → T::thread_block member: [✗] (missing or wrong type)\n";
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
consteval auto detailed_diagnostic_SpecifiesTileBlockGemm() -> std::string {
    std::string msg;
    constexpr bool has_block_gemm = requires { { T::block_gemm } -> TileBlockGemmDescriptor; };
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
    constexpr bool has_optimizations = requires { { T::optimizations } -> TileOptimizationsDescriptor; };
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
    constexpr bool has_thread_config = requires { { T::thread_config } -> DlThreadConfigDescriptor; };
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
    constexpr bool has_thread_cluster = requires { { T::thread_cluster } -> DlThreadClusterDescriptor; };
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
consteval auto detailed_diagnostic_SpecifiesDlFwdBlockTransfer() -> std::string {
    std::string msg;
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a = requires { { T::transfer.a } -> DlBlockTransferDescriptor4D; };
    constexpr bool has_b = requires { { T::transfer.b } -> DlBlockTransferDescriptor4D; };
    msg += "      → T::transfer.a: " + std::string(CHECK_MARK(has_a)) + "\n";
    
    if constexpr (has_a) {
        using ABT = decltype(T::transfer.a);
        constexpr bool has_thread_slice = requires(ABT t) { { t.thread_slice_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_thread_cluster = requires(ABT t) { { t.thread_cluster_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_cluster_arrange = requires(ABT t) { { t.thread_cluster_arrange_order } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_src_access = requires(ABT t) { { t.src_access_order } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_src_vector = requires(ABT t) { { t.src_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_src_contiguous = requires(ABT t) { { t.src_vector_tensor_contiguous_dim_order } -> std::convertible_to<std::array<size_t, 4>>; };
        constexpr bool has_dst_vector = requires(ABT t) { { t.dst_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>; };
        
        msg += "          → transfer.a.thread_slice_lengths (4D): " + std::string(CHECK_MARK(has_thread_slice)) + (has_thread_slice ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.thread_cluster_lengths (4D): " + std::string(CHECK_MARK(has_thread_cluster)) + (has_thread_cluster ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.thread_cluster_arrange_order (4D): " + std::string(CHECK_MARK(has_cluster_arrange)) + (has_cluster_arrange ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.src_access_order (4D): " + std::string(CHECK_MARK(has_src_access)) + (has_src_access ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.src_vector_tensor_lengths (4D): " + std::string(CHECK_MARK(has_src_vector)) + (has_src_vector ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.src_vector_tensor_contiguous_dim_order (4D): " + std::string(CHECK_MARK(has_src_contiguous)) + (has_src_contiguous ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.dst_vector_tensor_lengths (4D): " + std::string(CHECK_MARK(has_dst_vector)) + (has_dst_vector ? "\n" : " (missing or wrong type)\n");
    } else {
        msg += "              → T::transfer.a (4D): [✗] (missing or wrong type)\n";
    }
    
    msg += "      → T::transfer.b: " + std::string(CHECK_MARK(has_b)) + "\n";
    
    if constexpr (has_b) {
        msg += "              → T::transfer.b (4D): [✓] (similar fields as transfer.a)\n";
    } else {
        msg += "              → T::transfer.b (4D): [✗] (missing or wrong type)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesDlBwdBlockTransfer() -> std::string {
    std::string msg;
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a = requires { { T::transfer.a } -> DlBlockTransferDescriptor5D; };
    constexpr bool has_b = requires { { T::transfer.b } -> DlBlockTransferDescriptor5D; };
    msg += "      → T::transfer.a: " + std::string(CHECK_MARK(has_a)) + "\n";
    
    if constexpr (has_a) {
        using ABT = decltype(T::transfer.a);
        constexpr bool has_thread_slice = requires(ABT t) { { t.thread_slice_lengths } -> std::convertible_to<std::array<size_t, 5>>; };
        constexpr bool has_thread_cluster = requires(ABT t) { { t.thread_cluster_lengths } -> std::convertible_to<std::array<size_t, 5>>; };
        constexpr bool has_cluster_arrange = requires(ABT t) { { t.thread_cluster_arrange_order } -> std::convertible_to<std::array<size_t, 5>>; };
        constexpr bool has_src_access = requires(ABT t) { { t.src_access_order } -> std::convertible_to<std::array<size_t, 5>>; };
        constexpr bool has_src_vector = requires(ABT t) { { t.src_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 5>>; };
        constexpr bool has_src_contiguous = requires(ABT t) { { t.src_vector_tensor_contiguous_dim_order } -> std::convertible_to<std::array<size_t, 5>>; };
        constexpr bool has_dst_vector = requires(ABT t) { { t.dst_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 5>>; };
        
        msg += "          → transfer.a.thread_slice_lengths (5D): " + std::string(CHECK_MARK(has_thread_slice)) + (has_thread_slice ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.thread_cluster_lengths (5D): " + std::string(CHECK_MARK(has_thread_cluster)) + (has_thread_cluster ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.thread_cluster_arrange_order (5D): " + std::string(CHECK_MARK(has_cluster_arrange)) + (has_cluster_arrange ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.src_access_order (5D): " + std::string(CHECK_MARK(has_src_access)) + (has_src_access ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.src_vector_tensor_lengths (5D): " + std::string(CHECK_MARK(has_src_vector)) + (has_src_vector ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.src_vector_tensor_contiguous_dim_order (5D): " + std::string(CHECK_MARK(has_src_contiguous)) + (has_src_contiguous ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.a.dst_vector_tensor_lengths (5D): " + std::string(CHECK_MARK(has_dst_vector)) + (has_dst_vector ? "\n" : " (missing or wrong type)\n");
    } else {
        msg += "              → T::transfer.a (5D): [✗] (missing or wrong type)\n";
    }
    
    msg += "      → T::transfer.b: " + std::string(CHECK_MARK(has_b)) + "\n";
    
    if constexpr (has_b) {
        msg += "              → T::transfer.b (5D): [✓] (similar fields as transfer.a)\n";
    } else {
        msg += "              → T::transfer.b (5D): [✗] (missing or wrong type)\n";
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
    
    if constexpr (has_c && requires { T::transfer.c.src_dst_access_order; }) {
        using C = decltype(T::transfer.c);
        constexpr bool has_src_dst_access = requires(C t) { { t.src_dst_access_order } -> std::convertible_to<std::array<size_t, 6>>; };
        constexpr bool has_src_dst_vector_dim = requires(C t) { { t.src_dst_vector_dim } -> std::convertible_to<size_t>; };
        constexpr bool has_dst_scalar = requires(C t) { { t.dst_scalar_per_vector } -> std::convertible_to<size_t>; };
        
        msg += "          → transfer.c.src_dst_access_order: " + std::string(CHECK_MARK(has_src_dst_access)) + (has_src_dst_access ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.c.src_dst_vector_dim: " + std::string(CHECK_MARK(has_src_dst_vector_dim)) + (has_src_dst_vector_dim ? "\n" : " (missing or wrong type)\n");
        msg += "          → transfer.c.dst_scalar_per_vector: " + std::string(CHECK_MARK(has_dst_scalar)) + (has_dst_scalar ? "\n" : " (missing or wrong type)\n");
    } else if constexpr (has_c) {
        msg += "              → T::transfer.c (DlEpilogue): [✗] (missing required fields)\n";
    }
    
    return msg;
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesTileConvSpecialization() -> std::string {
    if constexpr (requires { T::specialization; }) {
        using SpecType = decltype(T::specialization);
        constexpr bool convertible = std::convertible_to<SpecType, TileConvSpecialization>;
        return "      → T::specialization: " + std::string(CHECK_MARK(convertible)) + 
               (convertible ? "" : std::string(detail::get_type_info<SpecType>())) + "\n";
    } else {
        return "      → T::specialization: [✗] (missing member)\n";
    }
}

template <typename T>
consteval auto detailed_diagnostic_SpecifiesLdsTransfer() -> std::string {
    std::string msg;
    
    constexpr bool has_transfer = requires { T::transfer; };
    msg += "      → T::transfer member: " + std::string(CHECK_MARK(has_transfer)) + "\n";
    
    if constexpr (!has_transfer) {
        return msg;
    }
    
    constexpr bool has_a = requires { { T::transfer.a.lds_transfer } -> LdsTransferDescriptor; };
    msg += "      → T::transfer.a: " + std::string(CHECK_MARK(has_a)) + "\n";
    if constexpr (!has_a) {
        msg += "            → T::transfer.a.lds_transfer: [✗] (missing or wrong type)\n";
    }
    
    constexpr bool has_b = requires { { T::transfer.b.lds_transfer } -> LdsTransferDescriptor; };
    msg += "      → T::transfer.b: " + std::string(CHECK_MARK(has_b)) + "\n";
    if constexpr (!has_b) {
        msg += "            → T::transfer.b.lds_transfer: [✗] (missing or wrong type)\n";
    }
    
    constexpr bool has_c = requires { { T::transfer.c.epilogue } -> EpilogueDescriptor; };
    msg += "      → T::transfer.c: " + std::string(CHECK_MARK(has_c)) + "\n";
    if constexpr (!has_c) {
        msg += "            → T::transfer.c.epilogue: [✗] (missing or wrong type)\n";
    }
    
    return msg;
}

} // namespace ck_tile::builder::diagnostics
