// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdJengaKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = FmhaPipeline::BiasEnum;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr bool kHasDropout       = FmhaPipeline::kHasDropout;
    static constexpr bool kDoFp8StaticQuant =
        (FmhaPipeline::Problem::QScaleEnum != ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE);
    static_assert(!FmhaPipeline::kIsGroupMode,
                  "Jenga sparse attention currently supports batch mode only.");
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "Jenga sparse attention does not support bias.");
    static_assert(!kStoreLSE, "Jenga sparse attention does not support LSE output.");
    static_assert(!kHasDropout, "Jenga sparse attention does not support dropout.");
    static_assert(!kHasLogitsSoftCap, "Jenga sparse attention does not support logits soft-cap.");
    static_assert(!kDoFp8StaticQuant,
                  "Jenga sparse attention does not support FP8 static quantization yet.");

    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    static constexpr bool kUseAsyncCopy = FmhaPipeline::Policy::AsyncCopy;

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct FmhaFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* block_relation_onehot_ptr;
        void* o_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;
        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;
    };

    struct FmhaFwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;
    };
    using Kargs = FmhaFwdBatchModeKargs;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                  const void* k_ptr,
                                                  const void* v_ptr,
                                                  const void* block_relation_onehot_ptr,
                                                  void* o_ptr,
                                                  ck_tile::index_t seqlen_q,
                                                  ck_tile::index_t seqlen_k,
                                                  ck_tile::index_t hdim_q,
                                                  ck_tile::index_t hdim_v,
                                                  ck_tile::index_t num_head_q,
                                                  ck_tile::index_t nhead_ratio_qk,
                                                  float scale_s,
                                                  ck_tile::index_t stride_q,
                                                  ck_tile::index_t stride_k,
                                                  ck_tile::index_t stride_v,
                                                  ck_tile::index_t stride_o,
                                                  ck_tile::index_t nhead_stride_q,
                                                  ck_tile::index_t nhead_stride_k,
                                                  ck_tile::index_t nhead_stride_v,
                                                  ck_tile::index_t nhead_stride_o,
                                                  ck_tile::index_t batch_stride_q,
                                                  ck_tile::index_t batch_stride_k,
                                                  ck_tile::index_t batch_stride_v,
                                                  ck_tile::index_t batch_stride_o,
                                                  ck_tile::index_t window_size_left,
                                                  ck_tile::index_t window_size_right,
                                                  ck_tile::index_t mask_type)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     block_relation_onehot_ptr,
                     o_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // FmhaFwdCommonKargs
                    {},               // FmhaFwdMaskKargs or FmhaFwdEmptyKargs<1>
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_)
    {
        return dim3(nhead_,
                    batch_size_,
                    ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                        ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1));
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

        const index_t i_block = blockIdx.z;
        const index_t i_nhead = blockIdx.x;
        const index_t i_batch = blockIdx.y;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        if constexpr(kHasMask)
        {
            // assume that num_tile_n1 is always 1
            return ck_tile::make_tuple(gridDim.z - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
        }
        else
        {
            return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
        }
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        // Extra LDS for staging block_relation_onehot (256 bools); keep 4B alignment for LDS loads.
        __shared__ char smem_ptr[GetSmemSize() + 256 * sizeof(int)];

        // if (threadIdx.x==0 && blockIdx.x==0 && blockIdx.z ==0) printf("smem size: %d",
        // int(GetSmemSize()));

        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q = 0;
        long_index_t batch_offset_k = 0;
        long_index_t batch_offset_v = 0;
        long_index_t batch_offset_o = 0;

        batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
        batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
        batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
        batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;

        // sparse mask
        const bool* block_relation_onehot_ptr =
            reinterpret_cast<const bool*>(kargs.block_relation_onehot_ptr) +
            static_cast<long_index_t>(i_batch * kargs.num_head_q + i_nhead) *
                ck_tile::integer_divide_ceil(kargs.seqlen_q, FmhaPipeline::kM0) *
                ck_tile::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0) +
            i_tile_m * ck_tile::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0);

        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});
            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK_, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.seqlen_k, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(kargs.seqlen_k)),
                                          make_tuple(sequence<1>{}, sequence<0>{}),
                                          make_tuple(sequence<0>{}, sequence<1>{}));

                constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
                return pad_tensor_view(
                    v_dram_transposed,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK_>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen_k),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                constexpr bool kPadHeadDimV_ = kUseAsyncCopy ? kPadHeadDimV : false;
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV_, kPadSeqLenK>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(number<FmhaPipeline::kM0>{},
                                      number<FmhaPipeline::kSubQKHeaddim>{});
                else
                    return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                             {i_n1, 0});

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        AttentionVariant variant;
        const auto variant_params = ck_tile::StandardAttentionParams<FmhaMask>{mask, kargs.scale_s};

        BlockIndices block_indices{i_batch, i_nhead, i_nhead / kargs.nhead_ratio_qk};

        auto o_acc_tile = FmhaPipeline{}(q_dram_window,
                                         k_dram_window,
                                         v_dram_window,
                                         block_relation_onehot_ptr,
                                         mask,
                                         kargs.scale_s,
                                         variant,
                                         variant_params,
                                         block_indices,
                                         smem_ptr);

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
