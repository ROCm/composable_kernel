// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename TilePartitioner_, typename FmhaPipeline_>
struct FmhaFwdAppendKVKernel
{
    using TilePartitioner                         = ck_tile::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;

    using QDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kApplyRoPE   = FmhaPipeline::RotaryEnum != BlockRotaryEmbeddingEnum::NONE;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<ck_tile::bf8_t> { static constexpr const char * name = "bf8"; };
    // clang-format on

    __host__ static std::string GetName()
    {
        // sync with generate.py
        // clang-format off

        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadSeqLenK) n += "sk";
            if (kPadHeadDimQ) n += "d";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_fwd_appendkv_d") + _TS_(FmhaPipeline::kTileSizeD) + "_" + _SS_(t2s<QDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_"
            "b" + _TS_(FmhaPipeline::kTileSizeS) + "x" + _TS_(FmhaPipeline::kTileSizeSk) + "x" + _TS_(FmhaPipeline::kTileSizeD) + "x" +
                  _TS_(FmhaPipeline::kTileSizeDv) + "_" + (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) +
            "v" + (std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor> ? "r" : "c") + (pn.empty() ? "" : "_" + pn) 
            + (!kApplyRoPE ? _SS_("") : (_SS_("_") + BlockRotaryEmbeddingEnumToStr<FmhaPipeline::RotaryEnum>::name));
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct EmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct CommonKargs
    {
        const void* q_ptr;
        void* k_ptr;
        const void* knew_ptr;
        void* v_ptr;
        const void* vnew_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t seqlen_knew;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_knew;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_vnew;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_knew;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_vnew;

        ck_tile::index_t batch_stride_knew;
        ck_tile::index_t batch_stride_vnew;
    };

    struct CommonRoPEKargs
    {
        const void* rotary_cos_ptr;
        const void* rotary_sin_ptr;
        ck_tile::index_t rotary_dim;
    };

    struct BatchModeKargs : CommonKargs,
                            std::conditional_t<kApplyRoPE, CommonRoPEKargs, EmptyKargs<0>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
    };

    struct GroupModeKargs : CommonKargs,
                            std::conditional_t<kApplyRoPE, CommonRoPEKargs, EmptyKargs<0>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, GroupModeKargs, BatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              void* k_ptr,
              const void* knew_ptr,
              void* v_ptr,
              const void* vnew_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t seqlen_knew,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              const void* rotary_cos_ptr,
              const void* rotary_sin_ptr,
              ck_tile::index_t rotary_dim,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_knew,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_vnew,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_knew,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_vnew,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_knew,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_vnew)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     knew_ptr,
                     v_ptr,
                     vnew_ptr,
                     seqlen_q,
                     seqlen_k,
                     seqlen_knew,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     stride_q,
                     stride_k,
                     stride_knew,
                     stride_v,
                     stride_vnew,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_knew,
                     nhead_stride_v,
                     nhead_stride_vnew,
                     batch_stride_knew,
                     batch_stride_vnew}, // args for common karg
                    {},                  // placeholder for rope
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v};

        if constexpr(kApplyRoPE)
        {
            kargs.rotary_cos_ptr = rotary_cos_ptr;
            kargs.rotary_sin_ptr = rotary_sin_ptr;
            kargs.rotary_dim     = rotary_dim;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              void* k_ptr,
              const void* knew_ptr,
              void* v_ptr,
              const void* vnew_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t seqlen_knew,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              const void* rotary_cos_ptr,
              const void* rotary_sin_ptr,
              ck_tile::index_t rotary_dim,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_knew,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_vnew,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_knew,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_vnew,
              ck_tile::index_t batch_stride_knew,
              ck_tile::index_t batch_stride_vnew)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     knew_ptr,
                     v_ptr,
                     vnew_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     seqlen_knew,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     stride_q,
                     stride_k,
                     stride_knew,
                     stride_v,
                     stride_vnew,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_knew,
                     nhead_stride_v,
                     nhead_stride_vnew,
                     batch_stride_knew,
                     batch_stride_vnew}, // args for common karg
                    {},                  // placeholder for rope
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};

        if constexpr(kApplyRoPE)
        {
            kargs.rotary_cos_ptr = rotary_cos_ptr;
            kargs.rotary_sin_ptr = rotary_sin_ptr;
            kargs.rotary_dim     = rotary_dim;
        }

        return kargs;
    }

    __host__ static constexpr auto GridSize(ck_tile::index_t batch_size,
                                            ck_tile::index_t nhead,
                                            ck_tile::index_t seqlen_q,
                                            ck_tile::index_t hdim_v)
    {
        return TilePartitioner::GridSize(batch_size, nhead, seqlen_q, hdim_v);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_sk, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_q, kargs.hdim_v);

        const index_t i_sk = __builtin_amdgcn_readfirstlane(i_tile_sk * FmhaPipeline::kTileSizeSk);
        // const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

#if defined(ENABLE_KERNEL_DEBUG_PRINT)
#define PRINTF(expr) printf("[POYENC][DEVICE] " #expr ": %2d\n", (expr));
#if 0
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == TID)
        {
            PRINTF(kargs.stride_k);
            PRINTF(kargs.nhead_stride_k);
            PRINTF(kargs.batch_stride_k);

            PRINTF(kargs.stride_knew);
            PRINTF(kargs.nhead_stride_knew);
            PRINTF(kargs.batch_stride_knew);

            PRINTF(kargs.stride_v);
            PRINTF(kargs.nhead_stride_v);
            PRINTF(kargs.batch_stride_v);

            PRINTF(kargs.stride_vnew);
            PRINTF(kargs.nhead_stride_vnew);
            PRINTF(kargs.batch_stride_vnew);
        }
#endif
#endif

        long_index_t batch_offset_q = 0;
        long_index_t batch_offset_k = 0;
        long_index_t batch_offset_knew =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_knew;
        long_index_t batch_offset_v = 0;
        long_index_t batch_offset_vnew =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_vnew;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

#if 0
            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }
#endif

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        KDataType* k_ptr =
            reinterpret_cast<KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const KDataType* knew_ptr =
            reinterpret_cast<const KDataType*>(kargs.knew_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_knew +
            batch_offset_knew;
        VDataType* v_ptr =
            reinterpret_cast<VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const VDataType* vnew_ptr =
            reinterpret_cast<const VDataType*>(kargs.vnew_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_vnew +
            batch_offset_vnew;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});

            return pad_tensor_view(
                q_dram_naive,
                make_tuple(number<FmhaPipeline::kTileSizeS>{}, number<FmhaPipeline::kTileSizeD>{}),
                sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_k + kargs.seqlen_knew, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kTileSizeSk>{}, number<FmhaPipeline::kTileSizeD>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();
        const auto knew_dram = [&]() {
            const auto knew_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                knew_ptr,
                make_tuple(kargs.seqlen_knew, kargs.hdim_q),
                make_tuple(kargs.stride_knew, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                knew_dram_naive,
                make_tuple(number<FmhaPipeline::kTileSizeSk>{}, number<FmhaPipeline::kTileSizeD>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.seqlen_k + kargs.seqlen_knew, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                const auto v_dram_transposed = transform_tensor_view(
                    v_dram_naive,
                    make_tuple(make_pass_through_transform(kargs.hdim_v),
                               make_pass_through_transform(kargs.seqlen_k + kargs.seqlen_knew)),
                    make_tuple(sequence<1>{}, sequence<0>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return pad_tensor_view(v_dram_transposed,
                                       make_tuple(number<FmhaPipeline::kTileSizeDv>{},
                                                  number<FmhaPipeline::kTileSizeSk>{}),
                                       sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen_k + kargs.seqlen_knew),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                return pad_tensor_view(v_dram_naive,
                                       make_tuple(number<FmhaPipeline::kTileSizeDv>{},
                                                  number<FmhaPipeline::kTileSizeSk>{}),
                                       sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
        }();
        const auto vnew_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto vnew_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    vnew_ptr,
                    make_tuple(kargs.seqlen_knew, kargs.hdim_v),
                    make_tuple(kargs.stride_vnew, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                const auto vnew_dram_transposed = transform_tensor_view(
                    vnew_dram_naive,
                    make_tuple(make_pass_through_transform(kargs.hdim_v),
                               make_pass_through_transform(kargs.seqlen_knew)),
                    make_tuple(sequence<1>{}, sequence<0>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return pad_tensor_view(vnew_dram_transposed,
                                       make_tuple(number<FmhaPipeline::kTileSizeDv>{},
                                                  number<FmhaPipeline::kTileSizeSk>{}),
                                       sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
            else
            {
                const auto vnew_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    vnew_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen_knew),
                    make_tuple(kargs.stride_vnew, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                return pad_tensor_view(vnew_dram_naive,
                                       make_tuple(number<FmhaPipeline::kTileSizeDv>{},
                                                  number<FmhaPipeline::kTileSizeSk>{}),
                                       sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
        }();
        constexpr auto rotary_cos_sin_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kTileSizeSk>{}, number<FmhaPipeline::kTileSizeD / 2>{});
        const auto rotary_cos_dram_window = [&]() {
            if constexpr(kApplyRoPE)
            {
                const auto rotary_cos_dram = [&]() {
                    const auto rotary_cos_dram_native =
                        make_naive_tensor_view<address_space_enum::global>(
                            reinterpret_cast<const KDataType*>(kargs.rotary_cos_ptr),
                            make_tuple(kargs.seqlen_k + kargs.seqlen_knew, kargs.rotary_dim / 2),
                            make_tuple(kargs.rotary_dim / 2, 1),
                            number<8>{},
                            number<1>{});

                    return pad_tensor_view(rotary_cos_dram_native,
                                           rotary_cos_sin_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(
                    rotary_cos_dram, rotary_cos_sin_dram_window_lengths, {0, 0});
            }
            else
            {
                return make_null_tile_window(rotary_cos_sin_dram_window_lengths);
            }
        }();
        const auto rotary_sin_dram_window = [&]() {
            if constexpr(kApplyRoPE)
            {
                const auto rotary_sin_dram = [&]() {
                    const auto rotary_sin_dram_native =
                        make_naive_tensor_view<address_space_enum::global>(
                            reinterpret_cast<const KDataType*>(kargs.rotary_sin_ptr),
                            make_tuple(kargs.seqlen_k + kargs.seqlen_knew, kargs.rotary_dim / 2),
                            make_tuple(kargs.rotary_dim / 2, 1),
                            number<8>{},
                            number<1>{});

                    return pad_tensor_view(rotary_sin_dram_native,
                                           rotary_cos_sin_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(
                    rotary_sin_dram, rotary_cos_sin_dram_window_lengths, {0, 0});
            }
            else
            {
                return make_null_tile_window(rotary_cos_sin_dram_window_lengths);
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kTileSizeS>{}, number<FmhaPipeline::kTileSizeD>{}),
            {0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<FmhaPipeline::kTileSizeSk>{}, number<FmhaPipeline::kTileSizeD>{}),
            {kargs.seqlen_k, 0});

        auto knew_dram_window = make_tile_window(
            knew_dram,
            make_tuple(number<FmhaPipeline::kTileSizeSk>{}, number<FmhaPipeline::kTileSizeD>{}),
            {i_sk, 0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<FmhaPipeline::kTileSizeDv>{}, number<FmhaPipeline::kTileSizeSk>{}),
            {0, kargs.seqlen_k});

        auto vnew_dram_window = make_tile_window(
            vnew_dram,
            make_tuple(number<FmhaPipeline::kTileSizeDv>{}, number<FmhaPipeline::kTileSizeSk>{}),
            {0, i_sk});

#if defined(ENABLE_KERNEL_DEBUG_PRINT)
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == TID)
        {
            printf("[POYENC][DEVICE] kargs.seqlen_k: %d\n", kargs.seqlen_k);
            printf("[POYENC][DEVICE] k_dram.get_length(0): %d\n",
                   k_dram.get_tensor_descriptor().get_length(number<0>{}));
            printf("[POYENC][DEVICE] k_dram.get_length(1): %d\n",
                   k_dram.get_tensor_descriptor().get_length(number<1>{}));
            printf("[POYENC][DEVICE] v_dram.get_length(0): %d\n",
                   v_dram.get_tensor_descriptor().get_length(number<0>{}));
            printf("[POYENC][DEVICE] v_dram.get_length(1): %d\n",
                   v_dram.get_tensor_descriptor().get_length(number<1>{}));
        }
#endif

        if constexpr(kApplyRoPE)
        {
            FmhaPipeline{}(q_dram_window,
                           k_dram_window,
                           knew_dram_window,
                           v_dram_window,
                           vnew_dram_window,
                           rotary_cos_dram_window,
                           rotary_sin_dram_window,
                           smem_ptr,
                           kargs.rotary_dim);
        }
        else
        {
            FmhaPipeline{}(q_dram_window,
                           k_dram_window,
                           knew_dram_window,
                           v_dram_window,
                           vnew_dram_window,
                           rotary_cos_dram_window,
                           rotary_sin_dram_window,
                           smem_ptr);
        }
    }
};

} // namespace ck_tile
