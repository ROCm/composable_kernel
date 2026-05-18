// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/fmha_problem.hpp"

#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>

namespace ck_tile {
namespace dispatcher {

struct FmhaKernelKey
{
    // Runtime signature -- corresponds to fmha_decl::FmhaSignature (build-time).
    // FmhaSignature uses strings for enums; Signature uses ints for matching speed.
    // When adding fields here, also update FmhaSignature and tie().
    struct Signature
    {
        FmhaKernelFamily family = FmhaKernelFamily::Fwd;
        std::string data_type;
        bool is_group_mode       = false;
        bool is_v_rowmajor       = true;
        bool has_logits_soft_cap = false;
        int mask_type            = 0;
        int bias_type            = 0;
        bool has_lse             = false;
        bool has_dropout         = false;
        int qscale_type          = 0;
        int rope_type            = 0;
        bool use_paged_kv        = false;
        bool do_fp8_static_quant = false;
        bool skip_min_seqlen_q   = false;
        bool has_sink            = false;
        bool has_dbias           = false;
        bool is_store_randval    = false;
        bool is_deterministic    = false;
        int kv_memory_layout     = 0;
        int kv_lookup_table      = 0;
        int page_size            = 1;
        std::uint16_t hdim_q     = 0;
        std::uint16_t hdim_v     = 0;
        int receipt              = -1;
    } signature;

    struct Algorithm
    {
        struct TileShape
        {
            std::uint16_t m0    = 0;
            std::uint16_t n0    = 0;
            std::uint16_t k0    = 0;
            std::uint16_t n1    = 0;
            std::uint16_t k1    = 0;
            std::uint16_t k0max = 0;
        } tile_shape;

        struct WaveShape
        {
            std::uint8_t m0 = 1;
            std::uint8_t n0 = 1;
            std::uint8_t k0 = 1;
            std::uint8_t m1 = 1;
            std::uint8_t n1 = 1;
            std::uint8_t k1 = 1;
            std::uint8_t m2 = 1;
            std::uint8_t n2 = 1;
            std::uint8_t k2 = 1;
        } wave_shape;

        struct WarpTileShape
        {
            std::uint16_t m0 = 0;
            std::uint16_t n0 = 0;
            std::uint16_t k0 = 0;
            std::uint16_t m1 = 0;
            std::uint16_t n1 = 0;
            std::uint16_t k1 = 0;
            std::uint16_t m2 = 0;
            std::uint16_t n2 = 0;
            std::uint16_t k2 = 0;
        } warp_tile_shape;

        std::string pipeline;
        bool pad_s                     = true;
        bool pad_sk                    = true;
        bool pad_d                     = true;
        bool pad_dv                    = true;
        bool use_trload                = false;
        std::uint8_t block_per_cu      = 1;
        std::uint8_t num_wave_groups   = 1;
        std::uint8_t max_splits_log2   = 0;
        std::uint16_t max_seq_len_q    = 0;
        std::uint16_t hdim_q_alignment = 0;
        std::uint16_t hdim_v_alignment = 0;
        std::int32_t selection_rank    = 0;
        std::string constraint_tag;
    } algorithm;

    std::string gfx_arch;

    [[nodiscard]] std::string encode_identifier() const
    {
        std::ostringstream oss;
        oss << "fmha_" << to_string(signature.family) << "_" << signature.data_type << "_"
            << (signature.is_group_mode ? "group" : "batch") << "_"
            << (signature.is_v_rowmajor ? "vr" : "vc") << "_hq" << signature.hdim_q << "_hv"
            << signature.hdim_v << "_p" << algorithm.pipeline << "_m" << signature.mask_type << "_b"
            << signature.bias_type << "_lse" << signature.has_lse << "_do" << signature.has_dropout
            << "_qs" << signature.qscale_type << "_rp" << signature.rope_type << "_pkv"
            << signature.use_paged_kv << "_sq" << signature.do_fp8_static_quant << "_sk"
            << signature.skip_min_seqlen_q << "_sink" << signature.has_sink << "_db"
            << signature.has_dbias << "_sr" << signature.is_store_randval << "_det"
            << signature.is_deterministic << "_km" << signature.kv_memory_layout << "_kl"
            << signature.kv_lookup_table << "_ps" << signature.page_size << "_t"
            << algorithm.tile_shape.m0 << "x" << algorithm.tile_shape.n0 << "x"
            << algorithm.tile_shape.k0 << "x" << algorithm.tile_shape.n1 << "x"
            << algorithm.tile_shape.k1 << "x" << algorithm.tile_shape.k0max << "_w0"
            << unsigned(algorithm.wave_shape.m0) << "x" << unsigned(algorithm.wave_shape.n0) << "x"
            << unsigned(algorithm.wave_shape.k0) << "_w1" << unsigned(algorithm.wave_shape.m1)
            << "x" << unsigned(algorithm.wave_shape.n1) << "x" << unsigned(algorithm.wave_shape.k1)
            << "_wt0" << algorithm.warp_tile_shape.m0 << "x" << algorithm.warp_tile_shape.n0 << "x"
            << algorithm.warp_tile_shape.k0 << "_wt1" << algorithm.warp_tile_shape.m1 << "x"
            << algorithm.warp_tile_shape.n1 << "x" << algorithm.warp_tile_shape.k1 << "_pads"
            << algorithm.pad_s << algorithm.pad_sk << algorithm.pad_d << algorithm.pad_dv << "_tr"
            << algorithm.use_trload << "_bpc" << unsigned(algorithm.block_per_cu) << "_wg"
            << unsigned(algorithm.num_wave_groups) << "_ms" << unsigned(algorithm.max_splits_log2)
            << "_mq" << algorithm.max_seq_len_q << "_aq" << algorithm.hdim_q_alignment << "_av"
            << algorithm.hdim_v_alignment << "_r" << algorithm.selection_rank << "_rc"
            << signature.receipt;
        return oss.str();
    }

    auto tie() const
    {
        return std::tie(signature.family,
                        signature.data_type,
                        signature.is_group_mode,
                        signature.is_v_rowmajor,
                        signature.has_logits_soft_cap,
                        signature.mask_type,
                        signature.bias_type,
                        signature.has_lse,
                        signature.has_dropout,
                        signature.qscale_type,
                        signature.rope_type,
                        signature.use_paged_kv,
                        signature.do_fp8_static_quant,
                        signature.skip_min_seqlen_q,
                        signature.has_sink,
                        signature.has_dbias,
                        signature.is_store_randval,
                        signature.is_deterministic,
                        signature.kv_memory_layout,
                        signature.kv_lookup_table,
                        signature.page_size,
                        signature.hdim_q,
                        signature.hdim_v,
                        algorithm.tile_shape.m0,
                        algorithm.tile_shape.n0,
                        algorithm.tile_shape.k0,
                        algorithm.tile_shape.n1,
                        algorithm.tile_shape.k1,
                        algorithm.tile_shape.k0max,
                        algorithm.wave_shape.m0,
                        algorithm.wave_shape.n0,
                        algorithm.wave_shape.k0,
                        algorithm.wave_shape.m1,
                        algorithm.wave_shape.n1,
                        algorithm.wave_shape.k1,
                        algorithm.wave_shape.m2,
                        algorithm.wave_shape.n2,
                        algorithm.wave_shape.k2,
                        algorithm.warp_tile_shape.m0,
                        algorithm.warp_tile_shape.n0,
                        algorithm.warp_tile_shape.k0,
                        algorithm.warp_tile_shape.m1,
                        algorithm.warp_tile_shape.n1,
                        algorithm.warp_tile_shape.k1,
                        algorithm.warp_tile_shape.m2,
                        algorithm.warp_tile_shape.n2,
                        algorithm.warp_tile_shape.k2,
                        algorithm.pipeline,
                        algorithm.pad_s,
                        algorithm.pad_sk,
                        algorithm.pad_d,
                        algorithm.pad_dv,
                        algorithm.use_trload,
                        algorithm.block_per_cu,
                        algorithm.num_wave_groups,
                        algorithm.max_splits_log2,
                        algorithm.max_seq_len_q,
                        algorithm.hdim_q_alignment,
                        algorithm.hdim_v_alignment,
                        algorithm.selection_rank,
                        algorithm.constraint_tag,
                        gfx_arch,
                        signature.receipt);
    }

    friend bool operator==(const FmhaKernelKey& lhs, const FmhaKernelKey& rhs)
    {
        return lhs.tie() == rhs.tie();
    }

    friend bool operator!=(const FmhaKernelKey& lhs, const FmhaKernelKey& rhs)
    {
        return !(lhs == rhs);
    }
};

} // namespace dispatcher
} // namespace ck_tile
